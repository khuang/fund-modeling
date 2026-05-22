"""
VC Fund Model — Interactive Streamlit App
Deploy: streamlit run streamlit_app.py
Streamlit Community Cloud: connect GitHub repo → share.streamlit.io
"""
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy.optimize import brentq
from dataclasses import dataclass
from typing import List
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="VC Fund Model",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("tab10")

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
    OutcomeBucket("Total Loss",   0.17,    0.0,     0.0,  0.0),
    OutcomeBucket("Small Return", 0.31,    2.0,    50.0,  0.5),
    OutcomeBucket("Mid Return",   0.27,   50.0,   400.0,  2.0),
    OutcomeBucket("Outsize",      0.13,  400.0,  2000.0,  3.0),
    OutcomeBucket("Outlier",      0.06, 2000.0, 10000.0,  4.0),
]

BEAR_BUCKETS = [
    OutcomeBucket("Total Loss",   0.28,    0.0,    0.0,  0.0),
    OutcomeBucket("Small Return", 0.33,    2.0,   40.0,  0.5),
    OutcomeBucket("Mid Return",   0.24,   40.0,  250.0,  2.0),
    OutcomeBucket("Outsize",      0.12,  250.0, 1000.0,  3.0),
    OutcomeBucket("Outlier",      0.03, 1000.0, 5000.0,  4.0),
]

GROWTH_BUCKETS = [
    OutcomeBucket("Small Return",  0.15,   20.0,   80.0,  0.5),
    OutcomeBucket("Mid Return",    0.42,   80.0,  500.0,  1.5),
    OutcomeBucket("Good Return",   0.28,  500.0, 2000.0,  2.5),
    OutcomeBucket("Outsize",       0.15, 2000.0, 8000.0,  3.0),
]

GRADUATION_RATE = 0.37

CONSTRUCTIONS_DEF = [
    ("High Conviction",    27, 0.40, 1.50, 4.50, "base"),
    ("Ultra Conviction",   12, 0.35, 4.00, 9.00, "base"),
    ("Spray & Pray",       55, 0.25, 0.50, 1.50, "base"),
    ("Pro-Rata Maximizer", 27, 0.55, 1.50, 4.50, "base"),
    ("Market Stress Test", 27, 0.40, 1.50, 4.50, "bear"),
]
ENTRY_VALS_GRID = [15.0, 20.0, 27.5, 35.0, 45.0]

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
            exit_vals[i] = rounds_arr[i] = 0.0
        else:
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
        company=[f"Co-{i+1:02d}" for i in range(n)],
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

    for _, row in portfolio.iterrows():
        yr = int(min(row["inv_year"], cfg.invest_period_yrs))
        invested[yr] += row["initial_check"]
        if row["follow_on"] > 0:
            fo_yr = int(min(yr + 2, cfg.fund_life_yrs))
            invested[fo_yr] += row["follow_on"]

    for _, row in portfolio.iterrows():
        yr = int(min(row["exit_year"], cfg.fund_life_yrs))
        proceeds[yr] += row["gross_proceeds"]

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


def run_fund(cfg: FundConfig, seed: int = 42) -> dict:
    port = simulate_portfolio(cfg, seed=seed)
    cf   = build_cashflows(cfg, port)
    deployed   = cfg.fund_size_m * cfg.deployment_rate
    undeployed = cfg.fund_size_m * (1 - cfg.deployment_rate)
    gross      = cf["gross_proceeds"].sum() - undeployed
    total_procs = gross + undeployed
    gross_cf   = (-cf["invested"] + (cf["gross_proceeds"] - cf["undeployed_returned"])).values
    return dict(
        cfg=cfg, portfolio=port, cashflows=cf,
        metrics=dict(
            fund=cfg.name, fund_size_m=cfg.fund_size_m,
            deployed_m=deployed, undeployed_returned_m=undeployed,
            entry_post_money_m=cfg.entry_post_money_m,
            gross_proceeds_m=gross, total_proceeds_m=total_procs,
            gross_moic=gross / deployed if deployed > 0 else np.nan,
            total_moic=total_procs / cfg.fund_size_m,
            dpi=total_procs / cfg.fund_size_m,
            gross_irr_pct=calc_irr(gross_cf) * 100,
        ),
    )


def simulate_followon_portfolio(seed_port, seed_cfg, growth_cfg, seed=None):
    rng = np.random.default_rng(seed)
    deployed    = growth_cfg.fund_size_m * growth_cfg.deployment_rate
    initial_cap = deployed * (1 - growth_cfg.reserve_ratio)
    reserve_cap = deployed * growth_cfg.reserve_ratio
    n           = growth_cfg.num_investments
    n_fo        = max(1, int(n * growth_cfg.follow_on_pct))
    n_grad      = max(1, min(round(len(seed_port) * GRADUATION_RATE), n))
    n_co        = n - n_grad
    graduates   = seed_port.nlargest(n_grad, "exit_val_m").reset_index(drop=True)

    raw    = np.exp(rng.uniform(np.log(growth_cfg.check_min_m),
                                np.log(growth_cfg.check_max_m), size=n))
    checks = raw / raw.sum() * initial_cap
    grad_checks, co_checks = checks[:n_grad], checks[n_grad:]

    rows = []
    for i in range(n_grad):
        row   = graduates.iloc[i]
        check = grad_checks[i]
        rem   = max(0.0, row["dilutive_rounds"] - 1.0)
        init_own = check / growth_cfg.entry_post_money_m
        dil_own  = init_own * (1 - growth_cfg.dilution_per_round) ** rem
        proceeds = dil_own * row["exit_val_m"]
        inv_yr   = rng.uniform(0.5, float(growth_cfg.invest_period_yrs))
        hold     = max(2.0, row["exit_year"] - row["inv_year"] - 1.5)
        exit_yr  = min(inv_yr + hold, float(growth_cfg.fund_life_yrs))
        rows.append(dict(
            company=f"Grad-{i+1:02d}", outcome=row["outcome"], source="seed_graduate",
            exit_val_m=row["exit_val_m"], dilutive_rounds=rem,
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
            init_own = check / growth_cfg.entry_post_money_m
            dil_own  = init_own * (1 - growth_cfg.dilution_per_round) ** rounds
            proceeds = dil_own * ev
            inv_yr   = rng.uniform(0.5, float(growth_cfg.invest_period_yrs))
            hold     = np.clip(rng.normal(growth_cfg.avg_hold_yrs, growth_cfg.std_hold_yrs),
                               2.0, float(growth_cfg.fund_life_yrs - 1))
            exit_yr  = np.clip(inv_yr + hold, 1.0, float(growth_cfg.fund_life_yrs))
            rows.append(dict(
                company=f"CoInv-{j+1:02d}", outcome=b.label, source="co_investment",
                exit_val_m=ev, dilutive_rounds=rounds, diluted_own_pct=dil_own * 100,
                inv_year=inv_yr, exit_year=exit_yr, initial_check=check, follow_on=0.0,
                total_invested=check, gross_proceeds=proceeds,
                multiple=proceeds / check if check > 0 else 0.0,
            ))

    port = pd.DataFrame(rows)
    company_rank = port["exit_val_m"] + rng.uniform(0, 1, len(port))
    fo_idx   = company_rank.nlargest(n_fo).index
    raw_fo   = rng.uniform(0.5, 2.0, size=len(fo_idx))
    fo_alloc = raw_fo / raw_fo.sum() * reserve_cap
    for idx_val, fo_amt in zip(fo_idx, fo_alloc):
        port.loc[idx_val, "follow_on"]      += fo_amt
        port.loc[idx_val, "total_invested"] += fo_amt
    return port


def run_growth_as_followon(seed_result, growth_cfg, seed=42):
    port      = simulate_followon_portfolio(
        seed_result["portfolio"], seed_result["cfg"], growth_cfg, seed=seed)
    cf        = build_cashflows(growth_cfg, port)
    deployed  = growth_cfg.fund_size_m * growth_cfg.deployment_rate
    undeployed = growth_cfg.fund_size_m * (1 - growth_cfg.deployment_rate)
    gross      = cf["gross_proceeds"].sum() - undeployed
    total_procs = gross + undeployed
    gross_cf   = (-cf["invested"] + (cf["gross_proceeds"] - cf["undeployed_returned"])).values
    return dict(
        cfg=growth_cfg, portfolio=port, cashflows=cf,
        metrics=dict(
            fund=growth_cfg.name, fund_size_m=growth_cfg.fund_size_m,
            deployed_m=deployed, undeployed_returned_m=undeployed,
            entry_post_money_m=growth_cfg.entry_post_money_m,
            gross_proceeds_m=gross, total_proceeds_m=total_procs,
            gross_moic=gross / deployed if deployed > 0 else np.nan,
            total_moic=total_procs / growth_cfg.fund_size_m,
            dpi=total_procs / growth_cfg.fund_size_m,
            gross_irr_pct=calc_irr(gross_cf) * 100,
        ),
    )


# ── Cached computations (primitive params only → Streamlit can hash these) ───
@st.cache_data(show_spinner=False)
def cached_monte_carlo(
    fund_size, entry_val, n_invest, reserve_pct, chk_min, chk_max,
    buckets_key, n_sims
):
    buckets = BASE_BUCKETS if buckets_key == "base" else BEAR_BUCKETS
    cfg = FundConfig(
        name="Seed Fund", fund_size_m=fund_size, vintage_year=2026,
        entry_post_money_m=entry_val, dilution_per_round=0.22,
        deployment_rate=0.90, num_investments=n_invest,
        reserve_ratio=reserve_pct / 100.0, check_min_m=chk_min, check_max_m=chk_max,
        follow_on_pct=0.15, outcome_buckets=buckets,
        avg_hold_yrs=7.0, std_hold_yrs=1.5,
    )
    deployed   = cfg.fund_size_m * cfg.deployment_rate
    undeployed = cfg.fund_size_m * (1 - cfg.deployment_rate)
    rows = []
    for s in range(n_sims):
        port  = simulate_portfolio(cfg, seed=s)
        cf    = build_cashflows(cfg, port)
        gross = cf["gross_proceeds"].sum() - undeployed
        total_procs = gross + undeployed
        gross_cf = (-cf["invested"] + (cf["gross_proceeds"] - cf["undeployed_returned"])).values
        rows.append(dict(
            gross_moic=gross / deployed if deployed > 0 else np.nan,
            total_moic=total_procs / cfg.fund_size_m,
            dpi=total_procs / cfg.fund_size_m,
            gross_irr_pct=calc_irr(gross_cf) * 100,
        ))
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def cached_entry_sweep(
    fund_size, n_invest, reserve_pct, chk_min, chk_max, buckets_key, n_sims
):
    entry_range = np.arange(12, 52, 2.5).tolist()
    buckets = BASE_BUCKETS if buckets_key == "base" else BEAR_BUCKETS
    rows = []
    for ev in entry_range:
        cfg = FundConfig(
            name="Sweep", fund_size_m=fund_size, vintage_year=2026,
            entry_post_money_m=ev, dilution_per_round=0.22,
            deployment_rate=0.90, num_investments=n_invest,
            reserve_ratio=reserve_pct / 100.0, check_min_m=chk_min, check_max_m=chk_max,
            follow_on_pct=0.15, outcome_buckets=buckets,
            avg_hold_yrs=7.0, std_hold_yrs=1.5,
        )
        deployed   = cfg.fund_size_m * cfg.deployment_rate
        undeployed = cfg.fund_size_m * (1 - cfg.deployment_rate)
        irr_list = []; dpi_list = []
        for s in range(n_sims):
            port  = simulate_portfolio(cfg, seed=s)
            cf    = build_cashflows(cfg, port)
            gross = cf["gross_proceeds"].sum() - undeployed
            total_procs = gross + undeployed
            gross_cf = (-cf["invested"] + (cf["gross_proceeds"] - cf["undeployed_returned"])).values
            irr_list.append(calc_irr(gross_cf) * 100)
            dpi_list.append(total_procs / cfg.fund_size_m)
        rows.append(dict(
            entry_val=ev,
            irr_p10=float(np.nanpercentile(irr_list, 10)),
            irr_median=float(np.nanmedian(irr_list)),
            irr_p90=float(np.nanpercentile(irr_list, 90)),
            dpi_median=float(np.nanmedian(dpi_list)),
        ))
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def cached_construction_grid(fund_size, n_sims_grid):
    def _run(args):
        cname, n, res, clo, chi, bkey, ev, fsz = args
        bkts = BASE_BUCKETS if bkey == "base" else BEAR_BUCKETS
        cfg = FundConfig(
            name=f"{cname}|{ev}", fund_size_m=fsz, vintage_year=2026,
            entry_post_money_m=ev, dilution_per_round=0.22,
            deployment_rate=0.90, num_investments=n, reserve_ratio=res,
            check_min_m=clo, check_max_m=chi, follow_on_pct=0.15,
            outcome_buckets=bkts, avg_hold_yrs=7.0, std_hold_yrs=1.5,
        )
        deployed   = cfg.fund_size_m * cfg.deployment_rate
        undeployed = cfg.fund_size_m * (1 - cfg.deployment_rate)
        irr_list = []; dpi_list = []; moic_list = []
        for s in range(n_sims_grid):
            port  = simulate_portfolio(cfg, seed=s)
            cf    = build_cashflows(cfg, port)
            gross = cf["gross_proceeds"].sum() - undeployed
            total_procs = gross + undeployed
            gross_cf = (-cf["invested"] + (cf["gross_proceeds"] - cf["undeployed_returned"])).values
            irr_list.append(calc_irr(gross_cf) * 100)
            dpi_list.append(total_procs / cfg.fund_size_m)
            moic_list.append(gross / deployed if deployed > 0 else np.nan)
        return (cname, ev,
                float(np.nanmedian(irr_list)), float(np.nanpercentile(irr_list, 10)),
                float(np.nanmedian(dpi_list)), float(np.nanmedian(moic_list)))

    args_list = [
        (cname, n, res, clo, chi, bkey, ev, fund_size)
        for (cname, n, res, clo, chi, bkey) in CONSTRUCTIONS_DEF
        for ev in ENTRY_VALS_GRID
    ]
    results = []
    with ThreadPoolExecutor(max_workers=min(8, len(args_list))) as pool:
        futs = [pool.submit(_run, a) for a in args_list]
        for f in as_completed(futs):
            results.append(f.result())

    cn = [c[0] for c in CONSTRUCTIONS_DEF]
    el = [f"${v}M" for v in ENTRY_VALS_GRID]
    irr_tab  = pd.DataFrame(index=cn, columns=el, dtype=float)
    p10_tab  = pd.DataFrame(index=cn, columns=el, dtype=float)
    dpi_tab  = pd.DataFrame(index=cn, columns=el, dtype=float)
    moic_tab = pd.DataFrame(index=cn, columns=el, dtype=float)
    for cname, ev, irr_med, irr_p10, dpi_med, moic_med in results:
        lbl = f"${ev}M"
        irr_tab.loc[cname, lbl]  = irr_med
        p10_tab.loc[cname, lbl]  = irr_p10
        dpi_tab.loc[cname, lbl]  = dpi_med
        moic_tab.loc[cname, lbl] = moic_med
    return irr_tab, p10_tab, dpi_tab, moic_tab


# ── Chart helpers ─────────────────────────────────────────────────────────────
def plot_jcurve(seed_result, growth_result):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for ax, result, label, color in [
        (axes[0], seed_result,   "Seed Fund I",   "#2196F3"),
        (axes[1], growth_result, "Growth Fund I", "#4CAF50"),
    ]:
        cf     = result["cashflows"]
        cumnet = (-cf["invested"] + cf["gross_proceeds"]).cumsum()
        ax.bar(cf["year"], -cf["invested"],       color="#EF5350", alpha=0.7, label="Capital deployed")
        ax.bar(cf["year"],  cf["gross_proceeds"], color=color,     alpha=0.7, label="Gross proceeds")
        ax.plot(cf["year"], cumnet, color="black", linewidth=2, label="Cumulative net")
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        m = result["metrics"]
        ax.set_title(f"{label}  |  DPI {m['dpi']:.2f}x  |  Gross IRR {m['gross_irr_pct']:.1f}%")
        ax.set_xlabel("Fund Year")
        ax.set_ylabel("$M")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:.0f}M"))
        ax.legend(fontsize=8)
    plt.tight_layout()
    return fig


def plot_irr_distribution(mc_df, entry_val, n_sims):
    fig, ax = plt.subplots(figsize=(8, 4))
    clean = mc_df["gross_irr_pct"].dropna()
    p10, med, p90 = float(clean.quantile(0.10)), float(clean.median()), float(clean.quantile(0.90))
    ax.hist(clean, bins=50, color="#2196F3", alpha=0.75, edgecolor="white", linewidth=0.4)
    for val, lbl, color in [
        (p10, f"P10  {p10:.1f}%",    "#FF5722"),
        (med, f"Median {med:.1f}%",  "#1565C0"),
        (p90, f"P90  {p90:.1f}%",    "#2E7D32"),
    ]:
        ax.axvline(val, color=color, linewidth=2.0, linestyle="--", label=lbl)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Gross IRR (%)")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Seed Fund I — Gross IRR Distribution\n"
                 f"(${entry_val}M entry · {n_sims:,} simulations)")
    ax.legend(fontsize=9)
    plt.tight_layout()
    return fig


def plot_dpi_distribution(mc_df, n_sims):
    fig, ax = plt.subplots(figsize=(8, 4))
    dpi_clean = mc_df["dpi"].dropna()
    med_dpi = float(dpi_clean.median())
    ax.hist(dpi_clean, bins=50, color="#4CAF50", alpha=0.75, edgecolor="white", linewidth=0.4)
    ax.axvline(med_dpi, color="#1B5E20", linewidth=2.0, linestyle="--",
               label=f"Median {med_dpi:.2f}x")
    ax.axvline(1.0, color="gray", linewidth=1.0, linestyle=":", label="1.0x (break-even)")
    ax.set_xlabel("DPI (x)")
    ax.set_ylabel("Frequency")
    ax.set_title(f"DPI Distribution\n({n_sims:,} simulations)")
    ax.legend(fontsize=9)
    plt.tight_layout()
    return fig


def plot_entry_sweep(sweep_df, current_entry):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))

    ax1.fill_between(sweep_df["entry_val"], sweep_df["irr_p10"], sweep_df["irr_p90"],
                     alpha=0.20, color="#2196F3", label="P10–P90 range")
    ax1.plot(sweep_df["entry_val"], sweep_df["irr_median"], "o-", color="#2196F3",
             linewidth=2.0, markersize=5, label="Median IRR")
    ax1.axvline(current_entry, color="red", linestyle="--", linewidth=1.5,
                label=f"Current: ${current_entry}M")
    ax1.axhline(0, color="gray", linewidth=0.6, linestyle=":")
    ax1.set_xlabel("Entry Post-Money Valuation ($M)")
    ax1.set_ylabel("Gross IRR (%)")
    ax1.set_title("Gross IRR vs Entry Valuation")
    ax1.legend(fontsize=9)

    ax2.plot(sweep_df["entry_val"], sweep_df["dpi_median"], "o-", color="#4CAF50",
             linewidth=2.0, markersize=5, label="Median DPI")
    ax2.axvline(current_entry, color="red", linestyle="--", linewidth=1.5,
                label=f"Current: ${current_entry}M")
    ax2.axhline(1.0, color="gray", linewidth=0.8, linestyle=":", label="1.0x (break-even)")
    ax2.set_xlabel("Entry Post-Money Valuation ($M)")
    ax2.set_ylabel("Median DPI (x)")
    ax2.set_title("DPI vs Entry Valuation")
    ax2.legend(fontsize=9)

    plt.suptitle("Impact of Entry Valuation on LP Returns (Seed Fund I)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    return fig


def plot_grid_heatmaps(irr_tab, p10_tab, dpi_tab, moic_tab):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    specs = [
        (irr_tab,  "Median Gross IRR (%)",  "YlGn",   axes[0, 0], ".1f", "%",  None),
        (p10_tab,  "P10 Gross IRR (%)",     "RdYlGn", axes[0, 1], ".1f", "%",  0),
        (dpi_tab,  "Median DPI (x)",        "YlOrRd", axes[1, 0], ".2f", "x",  None),
        (moic_tab, "Median Gross MOIC (x)", "Blues",  axes[1, 1], ".2f", "x",  None),
    ]
    for table, title, cmap, ax, fmt, suffix, center in specs:
        annot = table.map(lambda v: f"{v:{fmt[1:]}}{suffix}")
        sns.heatmap(
            table.astype(float), ax=ax, cmap=cmap, annot=annot, fmt="s",
            annot_kws={"size": 10}, linewidths=0.5, linecolor="white",
            center=center, cbar_kws={"shrink": 0.8},
        )
        ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
        ax.set_xlabel("Entry Post-Money Valuation", fontsize=9)
        ax.set_ylabel("")
        ax.tick_params(axis="x", labelsize=9)
        ax.tick_params(axis="y", labelsize=9, rotation=0)
    plt.suptitle("Portfolio Construction × Entry Valuation",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    return fig


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Parameters")

    st.subheader("Seed Fund I")
    s_fund_size  = st.slider("Fund size", 50, 300, 140, step=5, format="$%dM")
    s_entry      = st.slider("Entry post-money", 10.0, 60.0, 27.5, step=2.5, format="$%.1fM")
    s_n_inv      = st.slider("# investments", 5, 100, 27)
    s_reserve_pct = st.slider("Reserve ratio", 5, 70, 10, step=5, format="%d%%",
                               help="Fraction of deployed capital held for follow-on")
    s_chk_min    = st.slider("Min initial check", 0.25, 5.0, 2.0, step=0.25, format="$%.2fM")
    s_chk_max    = st.slider("Max initial check", 1.0, 15.0, 7.0, step=0.5, format="$%.1fM")
    if s_chk_min >= s_chk_max:
        s_chk_max = s_chk_min + 0.5
        st.warning("Check max auto-adjusted to be above min.")

    deployed_s     = s_fund_size * 0.90
    initial_cap_s  = deployed_s * (1 - s_reserve_pct / 100.0)
    avg_check_s    = initial_cap_s / s_n_inv
    init_own_s     = avg_check_s / s_entry * 100
    st.caption(
        f"Avg check: **${avg_check_s:.2f}M** · Initial ownership: **{init_own_s:.1f}%** · "
        f"Reserve: **${deployed_s * s_reserve_pct / 100:.0f}M**"
    )

    st.subheader("Growth Fund I *(follow-on only)*")
    g_fund_size  = st.slider("Fund size ", 50, 300, 140, step=5, format="$%dM")
    g_entry      = st.slider("Entry post-money ", 30.0, 150.0, 78.7, step=2.5, format="$%.1fM")
    g_n_inv      = st.slider("# investments ", 5, 50, 22)
    g_reserve_pct = st.slider("Reserve ratio ", 10, 60, 30, step=5, format="%d%%")

    st.subheader("Simulation")
    n_sims = st.select_slider(
        "# simulations", options=[200, 500, 1000, 2000], value=500,
        help="More simulations = smoother distributions, slower compute"
    )
    market = st.radio(
        "Market scenario",
        ["Base Case (17% loss rate)", "Bear Market (28% loss rate)"],
        help="Bear: compressed exit valuations, fewer outliers"
    )
    buckets_key = "bear" if "Bear" in market else "base"

    st.divider()
    st.caption("All charts update automatically when you change any parameter.")

# ── Build configs from sidebar params ────────────────────────────────────────
_buckets = BASE_BUCKETS if buckets_key == "base" else BEAR_BUCKETS

SEED_CFG = FundConfig(
    name="Seed Fund I", fund_size_m=s_fund_size, vintage_year=2026,
    entry_post_money_m=s_entry, dilution_per_round=0.22,
    deployment_rate=0.90, num_investments=s_n_inv,
    reserve_ratio=s_reserve_pct / 100.0,
    check_min_m=s_chk_min, check_max_m=s_chk_max,
    follow_on_pct=0.15, outcome_buckets=_buckets,
    avg_hold_yrs=7.0, std_hold_yrs=1.5,
)

GROWTH_CFG = FundConfig(
    name="Growth Fund I", fund_size_m=g_fund_size, vintage_year=2026,
    entry_post_money_m=g_entry, dilution_per_round=0.22,
    deployment_rate=0.90, num_investments=g_n_inv,
    reserve_ratio=g_reserve_pct / 100.0,
    check_min_m=2.50, check_max_m=10.00,
    follow_on_pct=0.35, outcome_buckets=GROWTH_BUCKETS,
    avg_hold_yrs=5.5, std_hold_yrs=1.5,
)

# ── Page header ───────────────────────────────────────────────────────────────
st.title("📈 VC Fund Model — AI-Focused, 2026")
st.caption(
    f"Seed Fund I · ${s_fund_size}M · {s_n_inv} investments · ${s_entry}M entry  "
    f"| Growth Fund I · ${g_fund_size}M · {g_n_inv} investments · ${g_entry}M entry  "
    f"| {n_sims:,} simulations · {market.split(' (')[0]}"
)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_overview, tab_mc, tab_sensitivity, tab_grid = st.tabs([
    "📊 Overview",
    "🎲 Monte Carlo",
    "📉 Entry Sensitivity",
    "🔲 Construction Grid",
])

# ─────────────────────────────────────────────────────────────────────────────
# Tab 1 — Overview (single deterministic run, seed=42)
# ─────────────────────────────────────────────────────────────────────────────
with tab_overview:
    seed_result   = run_fund(SEED_CFG, seed=42)
    growth_result = run_growth_as_followon(seed_result, GROWTH_CFG, seed=42)

    ms = seed_result["metrics"]
    mg = growth_result["metrics"]
    total_committed = s_fund_size + g_fund_size
    total_proceeds  = ms["total_proceeds_m"] + mg["total_proceeds_m"]
    combined_moic   = total_proceeds / total_committed

    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("Seed Fund I")
        st.metric("Gross IRR",  f"{ms['gross_irr_pct']:.1f}%")
        st.metric("Gross MOIC (on deployed)", f"{ms['gross_moic']:.2f}x")
        st.metric("DPI",        f"{ms['dpi']:.2f}x")
        st.caption(
            f"Avg check: ${avg_check_s:.2f}M  ·  "
            f"Initial ownership: {init_own_s:.1f}%  ·  "
            f"Reserve: ${deployed_s * s_reserve_pct / 100:.0f}M"
        )

    with c2:
        st.subheader("Growth Fund I")
        st.metric("Gross IRR",  f"{mg['gross_irr_pct']:.1f}%")
        st.metric("Gross MOIC (on deployed)", f"{mg['gross_moic']:.2f}x")
        st.metric("DPI",        f"{mg['dpi']:.2f}x")
        n_grad = sum(1 for s in growth_result["portfolio"]["source"] if s == "seed_graduate")
        n_co   = sum(1 for s in growth_result["portfolio"]["source"] if s == "co_investment")
        st.caption(
            f"{n_grad} seed graduates + {n_co} co-investments  ·  "
            f"Entry: ${g_entry}M"
        )

    with c3:
        st.subheader("Combined LP")
        st.metric("Combined MOIC", f"{combined_moic:.2f}x")
        st.metric("Total proceeds", f"${total_proceeds:.0f}M")
        st.metric("Total committed", f"${total_committed:.0f}M")
        n_seed_write = seed_result["portfolio"]["outcome"].value_counts().get("Total Loss", 0)
        st.caption(
            f"Seed write-offs: {n_seed_write} / {s_n_inv}  ·  "
            f"Graduation rate: {GRADUATION_RATE:.0%} → {n_grad} growth investments"
        )

    st.divider()

    col_l, col_r = st.columns(2)
    with col_l:
        st.subheader("Seed portfolio by outcome")
        by_bucket = (
            seed_result["portfolio"]
            .groupby("outcome")
            .agg(n=("company", "count"),
                 avg_exit_val_m=("exit_val_m", "mean"),
                 total_invested=("total_invested", "sum"),
                 gross_proceeds=("gross_proceeds", "sum"))
            .assign(moic=lambda d: d["gross_proceeds"] / d["total_invested"].replace(0, np.nan))
            .sort_values("gross_proceeds", ascending=False)
            .round(2)
        )
        st.dataframe(by_bucket, use_container_width=True)

    with col_r:
        st.subheader("Growth portfolio by source")
        src = (
            growth_result["portfolio"]
            .groupby("source")
            .agg(n=("company", "count"),
                 avg_check=("initial_check", "mean"),
                 total_invested=("total_invested", "sum"),
                 gross_proceeds=("gross_proceeds", "sum"))
            .assign(moic=lambda d: d["gross_proceeds"] / d["total_invested"].replace(0, np.nan))
            .round(2)
        )
        st.dataframe(src, use_container_width=True)

    st.divider()
    st.subheader("J-Curves — Cash Flows Over Fund Life")
    fig_jcurve = plot_jcurve(seed_result, growth_result)
    st.pyplot(fig_jcurve)
    plt.close(fig_jcurve)


# ─────────────────────────────────────────────────────────────────────────────
# Tab 2 — Monte Carlo
# ─────────────────────────────────────────────────────────────────────────────
with tab_mc:
    with st.spinner(f"Running {n_sims:,} Monte Carlo simulations…"):
        mc_df = cached_monte_carlo(
            s_fund_size, s_entry, s_n_inv, s_reserve_pct,
            s_chk_min, s_chk_max, buckets_key, n_sims
        )

    clean = mc_df["gross_irr_pct"].dropna()
    p10  = float(clean.quantile(0.10))
    p25  = float(clean.quantile(0.25))
    med  = float(clean.median())
    p75  = float(clean.quantile(0.75))
    p90  = float(clean.quantile(0.90))
    pct_positive = float((clean > 0).mean() * 100)

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("P10 IRR",    f"{p10:.1f}%",  help="Worst decile outcome")
    c2.metric("P25 IRR",    f"{p25:.1f}%")
    c3.metric("Median IRR", f"{med:.1f}%")
    c4.metric("P75 IRR",    f"{p75:.1f}%")
    c5.metric("P90 IRR",    f"{p90:.1f}%",  help="Best decile outcome")
    c6.metric("% Positive", f"{pct_positive:.0f}%", help="Fraction of runs with IRR > 0")

    st.divider()

    col_l, col_r = st.columns(2)
    with col_l:
        fig_irr = plot_irr_distribution(mc_df, s_entry, n_sims)
        st.pyplot(fig_irr)
        plt.close(fig_irr)

    with col_r:
        fig_dpi = plot_dpi_distribution(mc_df, n_sims)
        st.pyplot(fig_dpi)
        plt.close(fig_dpi)

    with st.expander("Percentile table"):
        pct_rows = []
        for pct in [5, 10, 25, 50, 75, 90, 95]:
            pct_rows.append({
                "Percentile": f"P{pct}",
                "Gross IRR (%)": f"{mc_df['gross_irr_pct'].quantile(pct / 100):.1f}%",
                "DPI (x)":       f"{mc_df['dpi'].quantile(pct / 100):.2f}x",
                "Gross MOIC (x)":f"{mc_df['gross_moic'].quantile(pct / 100):.2f}x",
            })
        st.dataframe(
            pd.DataFrame(pct_rows).set_index("Percentile"),
            use_container_width=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Tab 3 — Entry Valuation Sensitivity
# ─────────────────────────────────────────────────────────────────────────────
with tab_sensitivity:
    n_sweep = min(n_sims, 300)
    st.info(
        f"Sweeping entry valuation $12M–$50M with all other Seed Fund I params held fixed. "
        f"Running {n_sweep} simulations per data point. Current entry: **${s_entry}M**."
    )
    with st.spinner("Computing sensitivity sweep…"):
        sweep_df = cached_entry_sweep(
            s_fund_size, s_n_inv, s_reserve_pct,
            s_chk_min, s_chk_max, buckets_key, n_sweep
        )

    fig_sweep = plot_entry_sweep(sweep_df, s_entry)
    st.pyplot(fig_sweep)
    plt.close(fig_sweep)

    current_irr = sweep_df.loc[
        (sweep_df["entry_val"] - s_entry).abs().idxmin(), "irr_median"
    ]
    base_irr = sweep_df.loc[
        (sweep_df["entry_val"] - 27.5).abs().idxmin(), "irr_median"
    ]
    drag = base_irr - current_irr
    if abs(s_entry - 27.5) > 1.0:
        direction = "above" if s_entry > 27.5 else "below"
        st.info(
            f"At ${s_entry}M (${abs(s_entry - 27.5):.1f}M {direction} the $27.5M base), "
            f"median IRR is **{current_irr:.1f}%** vs {base_irr:.1f}% at base. "
            f"{'Entry premium costs' if drag > 0 else 'Entry discount adds'} "
            f"**{abs(drag):.1f} pp** of median IRR."
        )

    with st.expander("View sensitivity table"):
        tbl = sweep_df.copy()
        tbl.columns = ["Entry ($M)", "P10 IRR (%)", "Median IRR (%)", "P90 IRR (%)", "Median DPI (x)"]
        tbl = tbl.round(2)
        tbl["Entry ($M)"] = tbl["Entry ($M)"].map(lambda x: f"${x:.1f}M")
        st.dataframe(tbl, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# Tab 4 — Construction Grid
# ─────────────────────────────────────────────────────────────────────────────
with tab_grid:
    st.write(
        "Five portfolio constructions × five entry valuations — "
        "every cell answers: *what do returns look like if we build this way and pay that price?*  \n"
        f"Fund size: **${s_fund_size}M** · Simulations per cell: **{min(n_sims, 500):,}**"
    )
    with st.expander("Construction definitions"):
        st.markdown("""
| Construction | # Companies | Reserve | Check range | Notes |
|---|---|---|---|---|
| High Conviction | 27 | 40% | $1.5–4.5M | Base case |
| Ultra Conviction | 12 | 35% | $4–9M | Fewer, larger bets |
| Spray & Pray | 55 | 25% | $0.5–1.5M | Maximum diversification |
| Pro-Rata Maximizer | 27 | 55% | $1.5–4.5M | Heavy follow-on reserve |
| Market Stress Test | 27 | 40% | $1.5–4.5M | Bear outcome distribution |

Entry valuations tested: $15M · $20M · $27.5M · $35M · $45M
""")

    run_grid = st.button("▶  Run Construction Grid", type="primary")

    if run_grid:
        st.session_state["grid_done"] = True

    if st.session_state.get("grid_done"):
        n_grid = min(n_sims, 500)
        with st.spinner(f"Running 25 scenarios × {n_grid} simulations in parallel…"):
            irr_tab, p10_tab, dpi_tab, moic_tab = cached_construction_grid(s_fund_size, n_grid)

        fig_grid = plot_grid_heatmaps(irr_tab, p10_tab, dpi_tab, moic_tab)
        st.pyplot(fig_grid)
        plt.close(fig_grid)

        with st.expander("View numeric tables"):
            st.subheader("Median Gross IRR (%)")
            st.dataframe(irr_tab.round(1), use_container_width=True)
            st.subheader("P10 Gross IRR (%) — Downside")
            st.dataframe(p10_tab.round(1), use_container_width=True)
            st.subheader("Median DPI (x)")
            st.dataframe(dpi_tab.round(2), use_container_width=True)
            st.subheader("Median Gross MOIC (x)")
            st.dataframe(moic_tab.round(2), use_container_width=True)
    else:
        st.caption("Click **▶ Run Construction Grid** to compute all 25 scenarios.")
        st.caption(
            "This runs ~12,500 simulations in parallel and takes 15–30 seconds. "
            "Results are cached — subsequent runs with the same fund size are instant."
        )
