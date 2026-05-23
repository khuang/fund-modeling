"""
VC Fund Model — Pyodide/WebAssembly backend.
Exposes get_overview(), get_mc(), get_sensitivity() as Python callables
that JavaScript invokes via pyodide.globals.get('...')(...args).
Returns JSON strings containing metric dicts + base64-encoded PNG charts.
"""
import matplotlib
matplotlib.use('Agg')           # non-interactive backend — must come first

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.optimize import brentq
from dataclasses import dataclass
from typing import List
import io, base64, json, warnings
warnings.filterwarnings('ignore')

plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': '#fafafa',
    'axes.grid': True,
    'grid.alpha': 0.35,
    'grid.linewidth': 0.6,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'font.size': 10,
})

# ── Data classes ──────────────────────────────────────────────────────────────
@dataclass
class OutcomeBucket:
    label: str
    prob: float
    exit_val_lo_m: float
    exit_val_hi_m: float
    avg_dilutive_rounds: float

@dataclass
class FundConfig:
    name: str
    fund_size_m: float
    vintage_year: int
    entry_post_money_m: float
    dilution_per_round: float
    deployment_rate: float
    num_investments: int
    reserve_ratio: float
    check_min_m: float
    check_max_m: float
    follow_on_pct: float
    outcome_buckets: List[OutcomeBucket]
    avg_hold_yrs: float
    std_hold_yrs: float
    fund_life_yrs: int = 10
    invest_period_yrs: int = 5

# ── Outcome distributions ─────────────────────────────────────────────────────
BASE_BUCKETS = [
    OutcomeBucket('Total Loss',   0.17,    0.0,     0.0,  0.0),
    OutcomeBucket('Small Return', 0.31,    2.0,    50.0,  0.5),
    OutcomeBucket('Mid Return',   0.27,   50.0,   400.0,  2.0),
    OutcomeBucket('Outsize',      0.13,  400.0,  2000.0,  3.0),
    OutcomeBucket('Outlier',      0.06, 2000.0, 10000.0,  4.0),
]

BEAR_BUCKETS = [
    OutcomeBucket('Total Loss',   0.28,    0.0,    0.0,  0.0),
    OutcomeBucket('Small Return', 0.33,    2.0,   40.0,  0.5),
    OutcomeBucket('Mid Return',   0.24,   40.0,  250.0,  2.0),
    OutcomeBucket('Outsize',      0.12,  250.0, 1000.0,  3.0),
    OutcomeBucket('Outlier',      0.03, 1000.0, 5000.0,  4.0),
]

GROWTH_BUCKETS = [
    OutcomeBucket('Small Return',  0.15,   20.0,   80.0,  0.5),
    OutcomeBucket('Mid Return',    0.42,   80.0,  500.0,  1.5),
    OutcomeBucket('Good Return',   0.28,  500.0, 2000.0,  2.5),
    OutcomeBucket('Outsize',       0.15, 2000.0, 8000.0,  3.0),
]

GRADUATION_RATE = 0.37


def _parse_buckets(buckets_json: str) -> List[OutcomeBucket]:
    data = json.loads(buckets_json)
    return [OutcomeBucket(**b) for b in data]


# ── Core model functions ──────────────────────────────────────────────────────
def simulate_portfolio(cfg: FundConfig, seed=None) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n   = cfg.num_investments

    deployed    = cfg.fund_size_m * cfg.deployment_rate
    initial_cap = deployed * (1 - cfg.reserve_ratio)
    reserve_cap = deployed * cfg.reserve_ratio
    n_fo        = max(1, int(n * cfg.follow_on_pct))

    probs = np.array([b.prob for b in cfg.outcome_buckets], dtype=float)
    probs /= probs.sum()
    bucket_idx = rng.choice(len(cfg.outcome_buckets), size=n, p=probs)

    raw = np.exp(rng.uniform(np.log(cfg.check_min_m), np.log(cfg.check_max_m), size=n))
    initial_checks = raw / raw.sum() * initial_cap

    follow_on    = np.zeros(n)
    bucket_rank  = np.array([b.exit_val_hi_m for b in cfg.outcome_buckets])
    company_rank = bucket_rank[bucket_idx] + rng.uniform(0, 1, n)
    fo_idx       = np.argsort(company_rank)[::-1][:n_fo]
    raw_fo       = rng.uniform(0.5, 2.0, size=n_fo)
    follow_on[fo_idx] = raw_fo / raw_fo.sum() * reserve_cap
    total_invested = initial_checks + follow_on

    exit_vals  = np.zeros(n)
    rounds_arr = np.zeros(n)
    for i, bi in enumerate(bucket_idx):
        b = cfg.outcome_buckets[bi]
        if b.exit_val_hi_m == 0:
            continue
        lo            = max(b.exit_val_lo_m, 0.1)
        exit_vals[i]  = np.exp(rng.uniform(np.log(lo), np.log(b.exit_val_hi_m)))
        rounds_arr[i] = max(0.0, rng.normal(b.avg_dilutive_rounds, 0.5))

    initial_ownership = initial_checks / cfg.entry_post_money_m
    diluted_ownership = initial_ownership * (1 - cfg.dilution_per_round) ** rounds_arr
    gross_proceeds    = diluted_ownership * exit_vals
    multiples = np.where(total_invested > 0, gross_proceeds / total_invested, 0.0)

    inv_yr  = rng.uniform(0.5, float(cfg.invest_period_yrs), size=n)
    hold    = np.clip(rng.normal(cfg.avg_hold_yrs, cfg.std_hold_yrs, size=n),
                      2.0, float(cfg.fund_life_yrs - 1))
    exit_yr = np.clip(inv_yr + hold, 1.0, float(cfg.fund_life_yrs))

    return pd.DataFrame(dict(
        company=[f'Co-{i+1:02d}' for i in range(n)],
        outcome=[cfg.outcome_buckets[i].label for i in bucket_idx],
        exit_val_m=exit_vals, dilutive_rounds=rounds_arr,
        diluted_own_pct=diluted_ownership * 100,
        inv_year=inv_yr, exit_year=exit_yr,
        initial_check=initial_checks, follow_on=follow_on,
        total_invested=total_invested, multiple=multiples,
        gross_proceeds=gross_proceeds,
    ))


def build_cashflows(cfg: FundConfig, portfolio: pd.DataFrame) -> pd.DataFrame:
    yrs      = np.arange(cfg.fund_life_yrs + 1, dtype=float)
    invested = np.zeros(len(yrs))
    proceeds = np.zeros(len(yrs))

    # Vectorised — avoids iterrows bottleneck in WASM
    inv_yr_i  = np.minimum(portfolio['inv_year'].values.astype(int), cfg.invest_period_yrs)
    fo_yr_i   = np.minimum(inv_yr_i + 2, cfg.fund_life_yrs)
    exit_yr_i = np.minimum(portfolio['exit_year'].values.astype(int), cfg.fund_life_yrs)

    np.add.at(invested, inv_yr_i, portfolio['initial_check'].values)
    fo_vals = portfolio['follow_on'].values
    fo_mask = fo_vals > 0
    if fo_mask.any():
        np.add.at(invested, fo_yr_i[fo_mask], fo_vals[fo_mask])
    np.add.at(proceeds, exit_yr_i, portfolio['gross_proceeds'].values)

    undeployed = cfg.fund_size_m * (1 - cfg.deployment_rate)
    proceeds[cfg.fund_life_yrs] += undeployed

    return pd.DataFrame(dict(
        year=yrs, invested=invested, gross_proceeds=proceeds,
        undeployed_returned=np.where(yrs == cfg.fund_life_yrs, undeployed, 0),
    ))


def calc_irr(cashflows: np.ndarray) -> float:
    t = np.arange(len(cashflows), dtype=float)
    def npv(r): return np.sum(cashflows / (1 + r) ** t)
    try:    return brentq(npv, -0.9999, 20.0, maxiter=500)
    except: return np.nan


def _metrics(cfg, cf):
    deployed   = cfg.fund_size_m * cfg.deployment_rate
    undeployed = cfg.fund_size_m * (1 - cfg.deployment_rate)
    gross      = cf['gross_proceeds'].sum() - undeployed
    total_procs = gross + undeployed
    gross_cf   = (-cf['invested'] + (cf['gross_proceeds'] - cf['undeployed_returned'])).values
    return dict(
        fund=cfg.name, fund_size_m=cfg.fund_size_m,
        deployed_m=deployed, undeployed_returned_m=undeployed,
        gross_proceeds_m=gross, total_proceeds_m=total_procs,
        gross_moic=gross / deployed if deployed > 0 else np.nan,
        total_moic=total_procs / cfg.fund_size_m,
        dpi=total_procs / cfg.fund_size_m,
        gross_irr_pct=calc_irr(gross_cf) * 100,
    )


def run_fund(cfg: FundConfig, seed: int = 42) -> dict:
    port = simulate_portfolio(cfg, seed=seed)
    cf   = build_cashflows(cfg, port)
    return dict(cfg=cfg, portfolio=port, cashflows=cf, metrics=_metrics(cfg, cf))


def simulate_followon_portfolio(seed_port, seed_cfg, growth_cfg, seed=None):
    rng = np.random.default_rng(seed)
    deployed    = growth_cfg.fund_size_m * growth_cfg.deployment_rate
    initial_cap = deployed * (1 - growth_cfg.reserve_ratio)
    reserve_cap = deployed * growth_cfg.reserve_ratio
    n           = growth_cfg.num_investments
    n_fo        = max(1, int(n * growth_cfg.follow_on_pct))
    n_grad      = max(1, min(round(len(seed_port) * GRADUATION_RATE), n))
    n_co        = n - n_grad
    graduates   = seed_port.nlargest(n_grad, 'exit_val_m').reset_index(drop=True)

    raw    = np.exp(rng.uniform(np.log(growth_cfg.check_min_m),
                                np.log(growth_cfg.check_max_m), size=n))
    checks = raw / raw.sum() * initial_cap
    grad_checks, co_checks = checks[:n_grad], checks[n_grad:]

    rows = []
    for i in range(n_grad):
        row   = graduates.iloc[i]
        check = grad_checks[i]
        rem   = max(0.0, row['dilutive_rounds'] - 1.0)
        dil_own  = (check / growth_cfg.entry_post_money_m) * (1 - growth_cfg.dilution_per_round) ** rem
        proceeds = dil_own * row['exit_val_m']
        inv_yr   = rng.uniform(0.5, float(growth_cfg.invest_period_yrs))
        hold     = max(2.0, row['exit_year'] - row['inv_year'] - 1.5)
        exit_yr  = min(inv_yr + hold, float(growth_cfg.fund_life_yrs))
        rows.append(dict(
            company=f'Grad-{i+1:02d}', outcome=row['outcome'], source='seed_graduate',
            exit_val_m=row['exit_val_m'], dilutive_rounds=rem,
            diluted_own_pct=dil_own * 100, inv_year=inv_yr, exit_year=exit_yr,
            initial_check=check, follow_on=0.0, total_invested=check,
            gross_proceeds=proceeds, multiple=proceeds / check if check > 0 else 0.0,
        ))

    if n_co > 0:
        probs = np.array([b.prob for b in growth_cfg.outcome_buckets], dtype=float)
        probs /= probs.sum()
        bidx = rng.choice(len(growth_cfg.outcome_buckets), size=n_co, p=probs)
        for j in range(n_co):
            b     = growth_cfg.outcome_buckets[bidx[j]]
            check = co_checks[j]
            if b.exit_val_hi_m == 0:
                ev = rounds = 0.0
            else:
                lo    = max(b.exit_val_lo_m, 0.1)
                ev    = np.exp(rng.uniform(np.log(lo), np.log(b.exit_val_hi_m)))
                rounds = max(0.0, rng.normal(b.avg_dilutive_rounds, 0.5))
            dil_own  = (check / growth_cfg.entry_post_money_m) * (1 - growth_cfg.dilution_per_round) ** rounds
            proceeds = dil_own * ev
            inv_yr   = rng.uniform(0.5, float(growth_cfg.invest_period_yrs))
            hold     = np.clip(rng.normal(growth_cfg.avg_hold_yrs, growth_cfg.std_hold_yrs),
                               2.0, float(growth_cfg.fund_life_yrs - 1))
            exit_yr  = np.clip(inv_yr + hold, 1.0, float(growth_cfg.fund_life_yrs))
            rows.append(dict(
                company=f'CoInv-{j+1:02d}', outcome=b.label, source='co_investment',
                exit_val_m=ev, dilutive_rounds=rounds, diluted_own_pct=dil_own * 100,
                inv_year=inv_yr, exit_year=exit_yr, initial_check=check, follow_on=0.0,
                total_invested=check, gross_proceeds=proceeds,
                multiple=proceeds / check if check > 0 else 0.0,
            ))

    port = pd.DataFrame(rows)
    company_rank = port['exit_val_m'] + rng.uniform(0, 1, len(port))
    fo_idx   = company_rank.nlargest(n_fo).index
    raw_fo   = rng.uniform(0.5, 2.0, size=len(fo_idx))
    fo_alloc = raw_fo / raw_fo.sum() * reserve_cap
    for idx_val, fo_amt in zip(fo_idx, fo_alloc):
        port.loc[idx_val, 'follow_on']      += fo_amt
        port.loc[idx_val, 'total_invested'] += fo_amt
    return port


def run_growth_as_followon(seed_result, growth_cfg, seed=42):
    port = simulate_followon_portfolio(
        seed_result['portfolio'], seed_result['cfg'], growth_cfg, seed=seed)
    cf   = build_cashflows(growth_cfg, port)
    return dict(cfg=growth_cfg, portfolio=port, cashflows=cf, metrics=_metrics(growth_cfg, cf))


# ── Config builders ───────────────────────────────────────────────────────────
def _seed_cfg(s_fund_size, s_entry, s_n_inv, s_reserve_pct,
              s_chk_min, s_chk_max, buckets):
    return FundConfig(
        name='Seed Fund I', fund_size_m=float(s_fund_size), vintage_year=2026,
        entry_post_money_m=float(s_entry), dilution_per_round=0.22,
        deployment_rate=0.90, num_investments=int(s_n_inv),
        reserve_ratio=int(s_reserve_pct) / 100.0,
        check_min_m=float(s_chk_min), check_max_m=float(s_chk_max),
        follow_on_pct=0.15, outcome_buckets=buckets,
        avg_hold_yrs=7.0, std_hold_yrs=1.5,
    )

def _growth_cfg(g_fund_size, g_entry, g_n_inv, g_reserve_pct):
    return FundConfig(
        name='Growth Fund I', fund_size_m=float(g_fund_size), vintage_year=2026,
        entry_post_money_m=float(g_entry), dilution_per_round=0.22,
        deployment_rate=0.90, num_investments=int(g_n_inv),
        reserve_ratio=int(g_reserve_pct) / 100.0,
        check_min_m=2.50, check_max_m=10.00,
        follow_on_pct=0.35, outcome_buckets=GROWTH_BUCKETS,
        avg_hold_yrs=5.5, std_hold_yrs=1.5,
    )


# ── Chart helpers ─────────────────────────────────────────────────────────────
def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=120)
    buf.seek(0)
    data = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return data


def _plot_jcurve(seed_r, growth_r) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for ax, result, label, color in [
        (axes[0], seed_r,   'Seed Fund I',   '#2196F3'),
        (axes[1], growth_r, 'Growth Fund I', '#4CAF50'),
    ]:
        cf     = result['cashflows']
        cumnet = (-cf['invested'] + cf['gross_proceeds']).cumsum()
        ax.bar(cf['year'], -cf['invested'],       color='#EF5350', alpha=0.75, label='Capital deployed')
        ax.bar(cf['year'],  cf['gross_proceeds'], color=color,     alpha=0.75, label='Gross proceeds')
        ax.plot(cf['year'], cumnet, color='#212121', linewidth=2.0, label='Cumulative net')
        ax.axhline(0, color='#9E9E9E', linewidth=0.8, linestyle='--')
        m = result['metrics']
        ax.set_title(f'{label}  ·  DPI {m["dpi"]:.2f}x  ·  Gross IRR {m["gross_irr_pct"]:.1f}%',
                     fontweight='bold')
        ax.set_xlabel('Fund Year')
        ax.set_ylabel('$M')
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:.0f}M'))
        ax.legend(fontsize=8)
    plt.suptitle('LP Cash Flow J-Curves', fontsize=13, fontweight='bold')
    plt.tight_layout()
    return _fig_to_b64(fig)


def _plot_mc(irr_arr, dpi_arr, n_sims) -> str:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))

    clean = irr_arr[~np.isnan(irr_arr)]
    p10, med, p90 = np.percentile(clean, [10, 50, 90])
    ax1.hist(clean, bins=60, color='#2196F3', alpha=0.75, edgecolor='white', linewidth=0.3)
    for val, lbl, col in [
        (p10, f'P10: {p10:.1f}%',    '#FF5722'),
        (med, f'Median: {med:.1f}%', '#1565C0'),
        (p90, f'P90: {p90:.1f}%',    '#2E7D32'),
    ]:
        ax1.axvline(val, color=col, linewidth=2.0, linestyle='--', label=lbl)
    ax1.axvline(0, color='#212121', linewidth=0.8)
    ax1.set_xlabel('Gross IRR (%)')
    ax1.set_ylabel('Frequency')
    ax1.set_title(f'Gross IRR Distribution  ({n_sims:,} simulations)', fontweight='bold')
    ax1.legend(fontsize=9)

    dpi_clean = dpi_arr[~np.isnan(dpi_arr)]
    med_dpi   = float(np.median(dpi_clean))
    ax2.hist(dpi_clean, bins=60, color='#4CAF50', alpha=0.75, edgecolor='white', linewidth=0.3)
    ax2.axvline(med_dpi, color='#1B5E20', linewidth=2.0, linestyle='--',
                label=f'Median: {med_dpi:.2f}x')
    ax2.axvline(1.0, color='#9E9E9E', linewidth=1.0, linestyle=':', label='1.0x (break-even)')
    ax2.set_xlabel('DPI (x)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('DPI Distribution', fontweight='bold')
    ax2.legend(fontsize=9)

    plt.tight_layout()
    return _fig_to_b64(fig)


def _plot_sensitivity(entry_range, irr_p10_list, irr_med_list, irr_p90_list,
                      dpi_med_list, current_entry) -> str:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))

    ax1.fill_between(entry_range, irr_p10_list, irr_p90_list,
                     alpha=0.20, color='#2196F3', label='P10–P90 range')
    ax1.plot(entry_range, irr_med_list, 'o-', color='#2196F3',
             linewidth=2.0, markersize=5, label='Median IRR')
    ax1.axvline(current_entry, color='red', linestyle='--', linewidth=1.5,
                label=f'Current: ${current_entry}M')
    ax1.axhline(0, color='#9E9E9E', linewidth=0.6, linestyle=':')
    ax1.set_xlabel('Entry Post-Money Valuation ($M)')
    ax1.set_ylabel('Gross IRR (%)')
    ax1.set_title('Gross IRR vs Entry Valuation', fontweight='bold')
    ax1.legend(fontsize=9)

    ax2.plot(entry_range, dpi_med_list, 'o-', color='#4CAF50',
             linewidth=2.0, markersize=5, label='Median DPI')
    ax2.axvline(current_entry, color='red', linestyle='--', linewidth=1.5,
                label=f'Current: ${current_entry}M')
    ax2.axhline(1.0, color='#9E9E9E', linewidth=0.8, linestyle=':',
                label='1.0x (break-even)')
    ax2.set_xlabel('Entry Post-Money Valuation ($M)')
    ax2.set_ylabel('Median DPI (x)')
    ax2.set_title('DPI vs Entry Valuation', fontweight='bold')
    ax2.legend(fontsize=9)

    plt.suptitle('Impact of Entry Valuation on LP Returns (Seed Fund I)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    return _fig_to_b64(fig)


# ── Public API ────────────────────────────────────────────────────────────────
def get_overview(s_fund_size, s_entry, s_n_inv, s_reserve_pct, s_chk_min, s_chk_max,
                 g_fund_size, g_entry, g_n_inv, g_reserve_pct, buckets_json):
    """Single deterministic run (seed=42). Returns JSON."""
    bkts       = _parse_buckets(buckets_json)
    seed_cfg   = _seed_cfg(s_fund_size, s_entry, s_n_inv, s_reserve_pct,
                            s_chk_min, s_chk_max, bkts)
    growth_cfg = _growth_cfg(g_fund_size, g_entry, g_n_inv, g_reserve_pct)
    seed_r     = run_fund(seed_cfg, seed=42)
    growth_r   = run_growth_as_followon(seed_r, growth_cfg, seed=42)

    ms = seed_r['metrics']
    mg = growth_r['metrics']
    total_committed = float(s_fund_size) + float(g_fund_size)
    total_proceeds  = ms['total_proceeds_m'] + mg['total_proceeds_m']

    jcurve_b64 = _plot_jcurve(seed_r, growth_r)

    by_bucket = (
        seed_r['portfolio']
        .groupby('outcome')
        .agg(n=('company', 'count'),
             avg_exit_m=('exit_val_m', 'mean'),
             invested=('total_invested', 'sum'),
             proceeds=('gross_proceeds', 'sum'))
        .assign(moic=lambda d: (d['proceeds'] / d['invested'].replace(0, np.nan)).round(2))
        .sort_values('proceeds', ascending=False)
        .round(2)
        .reset_index()
    )

    src = (
        growth_r['portfolio']
        .groupby('source')
        .agg(n=('company', 'count'),
             avg_check=('initial_check', 'mean'),
             invested=('total_invested', 'sum'),
             proceeds=('gross_proceeds', 'sum'))
        .assign(moic=lambda d: (d['proceeds'] / d['invested'].replace(0, np.nan)).round(2))
        .round(2)
        .reset_index()
    )

    n_grad     = int((growth_r['portfolio']['source'] == 'seed_graduate').sum())
    n_co       = int((growth_r['portfolio']['source'] == 'co_investment').sum())
    n_writeoffs = int((seed_r['portfolio']['outcome'] == 'Total Loss').sum())

    return json.dumps({
        'seed': {
            'gross_irr':  round(ms['gross_irr_pct'], 1),
            'gross_moic': round(float(ms['gross_moic']), 2),
            'dpi':        round(ms['dpi'], 2),
        },
        'growth': {
            'gross_irr':  round(mg['gross_irr_pct'], 1),
            'gross_moic': round(float(mg['gross_moic']), 2),
            'dpi':        round(mg['dpi'], 2),
            'n_grad': n_grad,
            'n_co': n_co,
        },
        'combined': {
            'moic':            round(total_proceeds / total_committed, 2),
            'total_proceeds':  round(total_proceeds, 1),
            'total_committed': total_committed,
            'n_writeoffs':     n_writeoffs,
        },
        'jcurve': jcurve_b64,
        'seed_by_bucket':   by_bucket.to_dict('records'),
        'growth_by_source': src.to_dict('records'),
    })


def get_mc(s_fund_size, s_entry, s_n_inv, s_reserve_pct, s_chk_min, s_chk_max,
           buckets_json, n_sims):
    """Monte Carlo simulation. Returns JSON with percentile stats + chart."""
    bkts      = _parse_buckets(buckets_json)
    seed_cfg  = _seed_cfg(s_fund_size, s_entry, s_n_inv, s_reserve_pct,
                           s_chk_min, s_chk_max, bkts)
    deployed  = seed_cfg.fund_size_m * seed_cfg.deployment_rate
    undeployed = seed_cfg.fund_size_m * (1 - seed_cfg.deployment_rate)

    irr_list = []; dpi_list = []; moic_list = []
    for s in range(int(n_sims)):
        port  = simulate_portfolio(seed_cfg, seed=s)
        cf    = build_cashflows(seed_cfg, port)
        gross = cf['gross_proceeds'].sum() - undeployed
        total_procs = gross + undeployed
        gross_cf = (-cf['invested'] + (cf['gross_proceeds'] - cf['undeployed_returned'])).values
        irr_list.append(calc_irr(gross_cf) * 100)
        dpi_list.append(total_procs / seed_cfg.fund_size_m)
        moic_list.append(gross / deployed if deployed > 0 else np.nan)

    irr_arr  = np.array(irr_list, dtype=float)
    dpi_arr  = np.array(dpi_list, dtype=float)
    moic_arr = np.array(moic_list, dtype=float)
    clean    = irr_arr[~np.isnan(irr_arr)]

    chart_b64 = _plot_mc(irr_arr, dpi_arr, int(n_sims))

    percentiles = {}
    for pct in [5, 10, 25, 50, 75, 90, 95]:
        percentiles[f'P{pct}'] = {
            'irr':  round(float(np.nanpercentile(irr_arr, pct)), 1),
            'dpi':  round(float(np.nanpercentile(dpi_arr, pct)), 2),
            'moic': round(float(np.nanpercentile(moic_arr, pct)), 2),
        }

    return json.dumps({
        'stats': {
            'p10':          round(float(np.percentile(clean, 10)), 1),
            'p25':          round(float(np.percentile(clean, 25)), 1),
            'median':       round(float(np.median(clean)), 1),
            'p75':          round(float(np.percentile(clean, 75)), 1),
            'p90':          round(float(np.percentile(clean, 90)), 1),
            'pct_positive': round(float((irr_arr > 0).mean() * 100), 0),
        },
        'percentiles': percentiles,
        'chart': chart_b64,
    })


def get_sensitivity(s_fund_size, s_n_inv, s_reserve_pct, s_chk_min, s_chk_max,
                    buckets_json, n_sims, current_entry):
    """Entry valuation sweep $12–$50M. Returns JSON with chart + table."""
    entry_range = list(np.arange(12.0, 51.0, 3.0))
    bkts = _parse_buckets(buckets_json)

    irr_p10_list = []; irr_med_list = []; irr_p90_list = []; dpi_med_list = []

    for ev in entry_range:
        cfg = FundConfig(
            name='Sweep', fund_size_m=float(s_fund_size), vintage_year=2026,
            entry_post_money_m=float(ev), dilution_per_round=0.22,
            deployment_rate=0.90, num_investments=int(s_n_inv),
            reserve_ratio=int(s_reserve_pct) / 100.0,
            check_min_m=float(s_chk_min), check_max_m=float(s_chk_max),
            follow_on_pct=0.15, outcome_buckets=bkts,
            avg_hold_yrs=7.0, std_hold_yrs=1.5,
        )
        deployed   = cfg.fund_size_m * cfg.deployment_rate
        undeployed = cfg.fund_size_m * (1 - cfg.deployment_rate)
        irr_ev = []; dpi_ev = []
        for s in range(int(n_sims)):
            port  = simulate_portfolio(cfg, seed=s)
            cf    = build_cashflows(cfg, port)
            gross = cf['gross_proceeds'].sum() - undeployed
            gross_cf = (-cf['invested'] + (cf['gross_proceeds'] - cf['undeployed_returned'])).values
            irr_ev.append(calc_irr(gross_cf) * 100)
            dpi_ev.append((gross + undeployed) / cfg.fund_size_m)
        irr_arr_ev = np.array(irr_ev, dtype=float)
        irr_p10_list.append(float(np.nanpercentile(irr_arr_ev, 10)))
        irr_med_list.append(float(np.nanmedian(irr_arr_ev)))
        irr_p90_list.append(float(np.nanpercentile(irr_arr_ev, 90)))
        dpi_med_list.append(float(np.nanmedian(dpi_ev)))

    chart_b64 = _plot_sensitivity(entry_range, irr_p10_list, irr_med_list,
                                   irr_p90_list, dpi_med_list, float(current_entry))

    table = [
        {'entry_m': round(ev, 1), 'irr_p10': round(p10, 1),
         'irr_median': round(med, 1), 'irr_p90': round(p90, 1), 'dpi': round(dpi, 2)}
        for ev, p10, med, p90, dpi
        in zip(entry_range, irr_p10_list, irr_med_list, irr_p90_list, dpi_med_list)
    ]

    return json.dumps({'chart': chart_b64, 'table': table})


def _plot_optimizer_heatmaps(irr_mat, p10_mat, score_mat, moic_mat,
                              x_labels, y_labels, best_ri, best_ci):
    n_rows, n_cols = irr_mat.shape
    cell_px = max(1.1, 7.0 / max(n_cols, 1))
    half_w  = min(16, max(10, n_cols * cell_px * 2 + 3))
    half_h  = min(10, max(3,  n_rows * max(0.55, 3.5 / max(n_rows, 1)) + 2))
    fig, axes = plt.subplots(2, 2, figsize=(half_w * 2, half_h * 2))
    ann_fs  = max(6, min(9, int(65 / max(n_rows, n_cols, 1))))

    specs = [
        (irr_mat,   'Median Gross IRR (%)',                    'YlGn',   axes[0][0], '.0f', '%'),
        (p10_mat,   'P10 Gross IRR (%)  ← downside',          'RdYlGn', axes[0][1], '.0f', '%'),
        (moic_mat,  'Median TVPI (x)',                         'YlOrRd', axes[1][0], '.2f', 'x'),
        (score_mat, 'Risk-Adj Score\n(0.6×Median + 0.4×P10)', 'Blues',  axes[1][1], '.0f', ''),
    ]
    for mat, title, cmap, ax, fmt, suffix in specs:
        masked = np.ma.masked_invalid(mat)
        im     = ax.imshow(masked, cmap=cmap, aspect='auto')

        ax.set_xticks(np.arange(-0.5, mat.shape[1], 1), minor=True)
        ax.set_yticks(np.arange(-0.5, mat.shape[0], 1), minor=True)
        ax.grid(which='minor', color='white', linewidth=1.5)
        ax.tick_params(which='minor', size=0)

        ax.set_xticks(range(len(x_labels)))
        ax.set_xticklabels(x_labels, fontsize=max(7, 10 - n_cols // 3), rotation=30, ha='right')
        ax.set_yticks(range(len(y_labels)))
        ax.set_yticklabels(y_labels, fontsize=max(7, 10 - n_rows // 3))
        ax.set_xlabel('Fund Size', fontsize=9)
        ax.set_ylabel('# Investments', fontsize=9)
        ax.set_title(title, fontweight='bold', fontsize=10, pad=8)

        vmin = float(np.nanmin(mat)); vmax = float(np.nanmax(mat))
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat[i, j]
                if np.isnan(v):
                    continue
                norm = (v - vmin) / (vmax - vmin) if vmax > vmin else 0.5
                text_color = 'white' if norm > 0.60 else '#222'
                ax.text(j, i, f'{v:{fmt}}{suffix}',
                        ha='center', va='center', fontsize=ann_fs,
                        color=text_color, clip_on=True)

        rect = plt.Rectangle((best_ci - 0.5, best_ri - 0.5), 1, 1,
                              linewidth=3, edgecolor='red', facecolor='none', zorder=6)
        ax.add_patch(rect)
        plt.colorbar(im, ax=ax, shrink=0.85)

    plt.suptitle(
        'Portfolio Optimizer — Fund Size × # Investments\n'
        '(red border = highest risk-adjusted score)',
        fontsize=12, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return _fig_to_b64(fig)


def _plot_boxwhisker(irr_dist, fund_sizes, n_inv_list, best):
    blue    = '#1a237e'
    red_col = '#c62828'
    light   = '#e8eaf6'
    best_ni = best['n_inv']

    fig, ax = plt.subplots(figsize=(max(8, len(n_inv_list) * 0.9 + 2), 5),
                           facecolor='white')

    ni_data = [[v for fs in fund_sizes for v in irr_dist.get((fs, ni), [])]
               for ni in n_inv_list]
    bp = ax.boxplot(ni_data, labels=[str(ni) for ni in n_inv_list],
                    patch_artist=True,
                    medianprops=dict(color=red_col, linewidth=2),
                    whiskerprops=dict(linewidth=1.2, color='#555'),
                    capprops=dict(linewidth=1.2, color='#555'),
                    flierprops=dict(marker='.', markersize=3, alpha=0.35, color='#999'))
    for patch, ni in zip(bp['boxes'], n_inv_list):
        patch.set_facecolor(red_col if ni == best_ni else light)
        patch.set_edgecolor(blue)
        patch.set_alpha(0.85 if ni == best_ni else 0.65)

    ax.axhline(0, color='#9e9e9e', linewidth=0.8, linestyle='--')
    ax.set_title(
        'Gross IRR Distribution by # Investments  (all fund sizes pooled)\n'
        'red = optimal # investments from grid search',
        fontweight='bold', fontsize=10)
    ax.set_xlabel('# Investments')
    ax.set_ylabel('Gross IRR (%)')
    for lbl in ax.get_xticklabels():
        if lbl.get_text() == str(best_ni):
            lbl.set_color(red_col); lbl.set_fontweight('bold')

    plt.tight_layout()
    return _fig_to_b64(fig)


def _plot_pareto(rows, best, fund_sizes):
    # IRR is scale-invariant with fund_size (check, ownership, proceeds all
    # scale proportionally), so ax1 must colour by n_inv — the only dimension
    # that actually moves IRR coordinates.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ni_unique = sorted(set(r['n_inv'] for r in rows))
    palette   = list(plt.cm.tab20.colors)   # 20 plain Python (R,G,B) tuples
    ni_color  = {ni: palette[i % 20] for i, ni in enumerate(ni_unique)}
    best_fs, best_ni = best['fund_size'], best['n_inv']

    # ── ax1: IRR_p10 vs IRR_median, one point per n_inv ─────────────────────
    # All fund_sizes for the same n_inv land at the same coordinate; show the
    # middle fund_size as the representative point to avoid false overlap.
    for ni in ni_unique:
        subset = [r for r in rows if r['n_inv'] == ni]
        rep    = subset[len(subset) // 2]   # median fund_size representative
        col    = ni_color[ni]
        is_best = (ni == best_ni)
        ax1.scatter(rep['irr_p10'], rep['irr_median'],
                    s=220 if is_best else 80,
                    color=col,
                    marker='*' if is_best else 'o',
                    edgecolors='red' if is_best else '#333',
                    linewidths=1.5 if is_best else 0.5,
                    zorder=10 if is_best else 5,
                    label=f'{ni} cos')
        if is_best:
            ax1.annotate(
                f'  ★ {ni} cos  (${best_fs}M fund)\n  avg ${best["avg_check"]}M check',
                (rep['irr_p10'], rep['irr_median']), fontsize=8.5, color='red')

    ax1.legend(title='# investments', fontsize=7, title_fontsize=8,
               loc='lower right', ncol=max(1, len(ni_unique) // 8))
    ax1.set_xlabel('P10 Gross IRR (%)  ← downside risk')
    ax1.set_ylabel('Median Gross IRR (%)')
    ax1.set_title('IRR by # Investments\n(one point per n_inv; fund size doesn\'t shift IRR)',
                  fontweight='bold')
    ax1.axvline(0, color='#9e9e9e', linewidth=0.7, linestyle=':')

    # ── ax2: score vs avg_check, all fund_size × n_inv cells ────────────────
    # avg_check = fund_size × 0.9 × (1−reserve) / n_inv, so it DOES vary with
    # fund_size → meaningful x-axis variation here.
    for ni in ni_unique:
        col = ni_color[ni]
        pts = [(r['avg_check'], r['score'])
               for r in rows if r['n_inv'] == ni
               and not (r['fund_size'] == best_fs and r['n_inv'] == best_ni)]
        if pts:
            xs, ys = zip(*pts)
            ax2.scatter(xs, ys, s=60, color=col, alpha=0.8, zorder=5, label=f'{ni} cos')

    ax2.scatter(best['avg_check'], best['score'],
                s=220, color=ni_color[best_ni], marker='*',
                edgecolors='red', linewidths=1.5, zorder=10)
    ax2.annotate(f'  ★ ${best_fs}M / {best_ni} cos',
                 (best['avg_check'], best['score']), fontsize=8.5, color='red')

    ax2.legend(title='# investments', fontsize=7, title_fontsize=8,
               loc='upper right', ncol=max(1, len(ni_unique) // 8))
    ax2.set_xlabel('Avg Initial Check Size ($M)')
    ax2.set_ylabel('Risk-Adjusted Score  (0.6×Median + 0.4×P10 IRR)')
    ax2.set_title('Score vs Avg Check Size\n(fund size drives x-axis spread per n_inv)',
                  fontweight='bold')

    plt.suptitle('Portfolio Construction Optimizer', fontsize=12, fontweight='bold')
    plt.tight_layout()
    return _fig_to_b64(fig)


def get_optimizer(entry_val, reserve_pct, buckets_json, n_sims,
                  fs_min=75, fs_max=250, ni_min=10, ni_max=70, n_steps=7):
    """Grid search over fund_size × n_investments.

    Check range is derived from avg_check (min = 0.5×, max = 2×) so the
    construction is self-consistent as the grid varies.
    Score = 0.6 × median_irr + 0.4 × P10_irr  (risk-adjusted return).
    Returns JSON with best config, full results table, and charts.
    """
    fs_min  = max(25,  int(fs_min)); fs_max = max(fs_min + 5, int(fs_max))
    ni_min  = max(5,   int(ni_min)); ni_max = max(ni_min + 1, int(ni_max))
    n_steps = max(3, min(20, int(n_steps)))

    # Round fund sizes to nearest $5M; geomspace for n_inv to spread low end
    FUND_SIZES = sorted(set(int(round(v / 5) * 5)
                            for v in np.linspace(fs_min, fs_max, n_steps)))
    N_INV_LIST = sorted(set(max(5, int(round(v)))
                            for v in np.geomspace(ni_min, ni_max, n_steps)))

    bkts     = _parse_buckets(buckets_json)
    n_sims   = int(n_sims)
    res_frac = int(reserve_pct) / 100.0

    rows     = []
    irr_dist = {}   # (fund_size, n_inv) -> list[float] for box plots

    for fund_size in FUND_SIZES:
        for n_inv in N_INV_LIST:
            deployed  = fund_size * 0.90
            init_cap  = deployed * (1 - res_frac)
            avg_check = init_cap / n_inv
            chk_min   = max(0.1, avg_check * 0.5)
            chk_max   = avg_check * 2.0

            cfg = FundConfig(
                name=f'{fund_size}M/{n_inv}',
                fund_size_m=float(fund_size), vintage_year=2026,
                entry_post_money_m=float(entry_val),
                dilution_per_round=0.22, deployment_rate=0.90,
                num_investments=n_inv, reserve_ratio=res_frac,
                check_min_m=chk_min, check_max_m=chk_max,
                follow_on_pct=0.15, outcome_buckets=bkts,
                avg_hold_yrs=7.0, std_hold_yrs=1.5,
            )
            dep    = cfg.fund_size_m * cfg.deployment_rate
            undep  = cfg.fund_size_m * (1 - cfg.deployment_rate)
            irr_list = []; dpi_list = []; moic_list = []

            for s in range(n_sims):
                port  = simulate_portfolio(cfg, seed=s)
                cf    = build_cashflows(cfg, port)
                gross = cf['gross_proceeds'].sum() - undep
                total = gross + undep
                gcf   = (-cf['invested'] + (cf['gross_proceeds'] - cf['undeployed_returned'])).values
                irr_list.append(calc_irr(gcf) * 100)
                dpi_list.append(total / cfg.fund_size_m)
                moic_list.append(gross / dep if dep > 0 else np.nan)

            irr_arr  = np.array(irr_list, dtype=float)
            irr_med  = float(np.nanmedian(irr_arr))
            irr_p10  = float(np.nanpercentile(irr_arr, 10))
            dpi_med  = float(np.nanmedian(dpi_list))
            moic_med = float(np.nanmedian(moic_list))
            score    = 0.6 * irr_med + 0.4 * irr_p10

            irr_dist[(fund_size, n_inv)] = irr_list
            rows.append({
                'fund_size':   fund_size,
                'n_inv':       n_inv,
                'avg_check':   round(avg_check, 2),
                'irr_median':  round(irr_med, 1),
                'irr_p10':     round(irr_p10, 1),
                'dpi_median':  round(dpi_med, 2),
                'moic_median': round(moic_med, 2),
                'score':       round(score, 1),
            })

    best = max(rows, key=lambda r: r['score'])

    # Build matrices for heatmaps (rows = n_inv, cols = fund_size)
    n_r = len(N_INV_LIST); n_c = len(FUND_SIZES)
    irr_mat   = np.full((n_r, n_c), np.nan)
    p10_mat   = np.full((n_r, n_c), np.nan)
    score_mat = np.full((n_r, n_c), np.nan)
    moic_mat  = np.full((n_r, n_c), np.nan)

    for r in rows:
        ri = N_INV_LIST.index(r['n_inv'])
        ci = FUND_SIZES.index(r['fund_size'])
        irr_mat[ri, ci]   = r['irr_median']
        p10_mat[ri, ci]   = r['irr_p10']
        score_mat[ri, ci] = r['score']
        moic_mat[ri, ci]  = r['moic_median']

    x_labels = [f'${f}M' for f in FUND_SIZES]
    y_labels = [f'{n}' for n in N_INV_LIST]
    best_ri  = N_INV_LIST.index(best['n_inv'])
    best_ci  = FUND_SIZES.index(best['fund_size'])

    heatmap_b64   = _plot_optimizer_heatmaps(
        irr_mat, p10_mat, score_mat, moic_mat, x_labels, y_labels, best_ri, best_ci)
    pareto_b64    = _plot_pareto(rows, best, FUND_SIZES)
    boxwhisk_b64  = _plot_boxwhisker(irr_dist, FUND_SIZES, N_INV_LIST, best)

    return json.dumps({
        'best':       best,
        'results':    rows,
        'heatmap':    heatmap_b64,
        'pareto':     pareto_b64,
        'boxwhisker': boxwhisk_b64,
    })


print('VC Fund Model loaded. Functions available: get_overview, get_mc, get_optimizer')
