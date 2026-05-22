"""Appends Section 7 (parallel portfolio construction comparison) to fund_model.ipynb."""
import nbformat as nbf

with open('fund_model.ipynb') as f:
    nb = nbf.read(f, as_version=4)

def md(src):  return nbf.v4.new_markdown_cell(src)
def code(src): return nbf.v4.new_code_cell(src)

new_cells = []

new_cells.append(md("""\
## Section 7 — Parallel Portfolio Construction Comparison

Six scenarios are simulated concurrently via `ThreadPoolExecutor`.
Each runs 1,000 Monte Carlo trials; results are compared on net IRR, DPI, and GP carry.

| Scenario | Investments | Reserve | Check Size | Notes |
|----------|------------|---------|-----------|-------|
| Base Case | 55 | 40% | $0.5–2M | Current config |
| Concentrated | 22 | 35% | $2–5M | Fewer, larger bets |
| Spray & Pray | 85 | 25% | $0.15–0.75M | Maximum diversification |
| High Reserve | 55 | 55% | $0.5–2M | Follow more winners |
| AI-Tilted | 55 | 40% | $0.5–2M | Lower loss rate, higher outlier |
| Bear Case | 55 | 40% | $0.5–2M | Higher loss rate (30%) |
"""))

new_cells.append(code("""\
from copy import deepcopy
import time

# ── Define scenarios ──────────────────────────────────────────────────────
def make_scenario(name, num_investments, reserve_ratio,
                  check_min, check_max, outcome_buckets=None) -> FundConfig:
    cfg = deepcopy(SEED_FUND)
    cfg.name = name
    cfg.num_investments = num_investments
    cfg.reserve_ratio = reserve_ratio
    cfg.check_min_m = check_min
    cfg.check_max_m = check_max
    if outcome_buckets is not None:
        cfg.outcome_buckets = outcome_buckets
    return cfg


BASE_BUCKETS = SEED_FUND.outcome_buckets

AI_BUCKETS = [
    OutcomeBucket('Total Loss',   0.15, 0.00, 0.00),  # lower loss rate
    OutcomeBucket('Small Return', 0.28, 0.10, 1.50),
    OutcomeBucket('Mid Return',   0.25, 1.50, 5.00),
    OutcomeBucket('Outsize',      0.20, 5.00, 15.0),
    OutcomeBucket('Outlier',      0.12, 15.0, 50.0),  # higher outlier rate
]

BEAR_BUCKETS = [
    OutcomeBucket('Total Loss',   0.30, 0.00, 0.00),  # higher loss rate
    OutcomeBucket('Small Return', 0.32, 0.10, 1.50),
    OutcomeBucket('Mid Return',   0.22, 1.50, 5.00),
    OutcomeBucket('Outsize',      0.12, 5.00, 15.0),
    OutcomeBucket('Outlier',      0.04, 15.0, 50.0),
]

SCENARIOS = [
    make_scenario('Base Case',     55, 0.40, 0.50, 2.00),
    make_scenario('Concentrated',  22, 0.35, 2.00, 5.00),
    make_scenario('Spray & Pray',  85, 0.25, 0.15, 0.75),
    make_scenario('High Reserve',  55, 0.55, 0.50, 2.00),
    make_scenario('AI-Tilted',     55, 0.40, 0.50, 2.00, AI_BUCKETS),
    make_scenario('Bear Case',     55, 0.40, 0.50, 2.00, BEAR_BUCKETS),
]

print(f"Defined {len(SCENARIOS)} scenarios:")
for s in SCENARIOS:
    print(f"  {s.name:<18} {s.num_investments} investments, "
          f"{s.reserve_ratio:.0%} reserve, "
          f"${s.check_min_m}–${s.check_max_m}M checks")
"""))

new_cells.append(code("""\
# ── Worker function ───────────────────────────────────────────────────────
# ThreadPoolExecutor is used instead of ProcessPoolExecutor because spawned
# processes can't pickle functions defined inside a notebook kernel.
# NumPy releases the GIL during array operations, so threads run in parallel
# for the heavy simulation work.
def _run_scenario_mc(args):
    \"\"\"Runs Monte Carlo for one scenario.\"\"\"
    cfg, n_sims = args
    mech = fund_mechanics(cfg)
    rows = []
    for s in range(n_sims):
        port = simulate_portfolio(cfg, mech, seed=s)
        cf   = build_cashflows(cfg, port, mech)
        gross = cf['gross_distributions'].sum()
        total_lp = cf['total_calls'].sum()
        wf = apply_waterfall(total_lp, mech['gp_commit'], gross,
                             cfg.fund_life_yrs, cfg.carry_rate, cfg.hurdle_rate)
        lp_pct = wf['lp_distributions'] / gross if gross > 0 else 0
        lp_dist = cf['gross_distributions'] * lp_pct
        net_cf_vals = (-cf['total_calls'] + lp_dist).values
        rows.append(dict(
            scenario=cfg.name,
            gross_moic=gross / cf['investments'].sum() if cf['investments'].sum() > 0 else float('nan'),
            net_dpi=wf['lp_distributions'] / total_lp,
            net_irr_pct=irr(net_cf_vals) * 100,
            gp_carry_m=wf['gp_carry'],
        ))
    return pd.DataFrame(rows)


# ── Run all scenarios in parallel ─────────────────────────────────────────
from concurrent.futures import ThreadPoolExecutor, as_completed

N_SIMS = 1000
t0 = time.time()

args_list = [(cfg, N_SIMS) for cfg in SCENARIOS]

scenario_results = {}
with ThreadPoolExecutor(max_workers=len(SCENARIOS)) as pool:
    futures = {pool.submit(_run_scenario_mc, args): args[0].name
               for args in args_list}
    for future in as_completed(futures):
        name = futures[future]
        scenario_results[name] = future.result()
        print(f"  ✓ {name} done")

elapsed = time.time() - t0
print(f"\\nAll {len(SCENARIOS)} scenarios complete in {elapsed:.1f}s "
      f"({N_SIMS} sims each = {len(SCENARIOS)*N_SIMS:,} total runs)")
"""))

new_cells.append(code("""\
# ── Summary table ─────────────────────────────────────────────────────────
import pandas as pd

summary_rows = []
for name in [s.name for s in SCENARIOS]:
    df = scenario_results[name]
    clean = df['net_irr_pct'].dropna()
    summary_rows.append(dict(
        Scenario=name,
        Investments=next(s.num_investments for s in SCENARIOS if s.name == name),
        Reserve=f"{next(s.reserve_ratio for s in SCENARIOS if s.name == name):.0%}",
        **{
            'Net IRR P10':    f"{clean.quantile(0.10):.1f}%",
            'Net IRR Median': f"{clean.median():.1f}%",
            'Net IRR P90':    f"{clean.quantile(0.90):.1f}%",
            'Net DPI Median': f"{df['net_dpi'].median():.2f}x",
            'GP Carry P50':   f"${df['gp_carry_m'].median():.0f}M",
        }
    ))

summary_df = pd.DataFrame(summary_rows).set_index('Scenario')
print("\\nPortfolio Construction Comparison — 1,000 Monte Carlo runs each")
print(summary_df.to_string())
"""))

new_cells.append(code("""\
# ── Visualization: IRR distributions across all scenarios ─────────────────
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

scenario_order = [s.name for s in SCENARIOS]
colors = plt.cm.tab10(np.linspace(0, 0.9, len(scenario_order)))

# ── Plot 1: Overlapping IRR distributions ────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 5))
for name, color in zip(scenario_order, colors):
    clean = scenario_results[name]['net_irr_pct'].dropna()
    ax.hist(clean, bins=60, alpha=0.45, label=name, color=color, density=True)
    ax.axvline(clean.median(), color=color, linewidth=1.8, linestyle='--')

ax.set_xlabel('Net IRR (%)')
ax.set_ylabel('Density')
ax.set_title('Net IRR Distribution by Portfolio Construction (1,000 sims each)')
ax.legend(loc='upper right', fontsize=9)
ax.axvline(0, color='black', linewidth=0.8)
plt.tight_layout()
plt.savefig('scenario_irr_overlay.png', dpi=150, bbox_inches='tight')
plt.show()
"""))

new_cells.append(code("""\
# ── Plot 2: Box plot comparison ───────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

metrics = [
    ('net_irr_pct', 'Net IRR (%)', axes[0]),
    ('net_dpi',     'Net DPI (x)', axes[1]),
    ('gp_carry_m',  'GP Carry ($M)', axes[2]),
]

for col, ylabel, ax in metrics:
    data  = [scenario_results[n][col].dropna().values for n in scenario_order]
    bp = ax.boxplot(data, patch_artist=True, notch=False,
                    medianprops=dict(color='black', linewidth=2),
                    whiskerprops=dict(linewidth=1.2),
                    flierprops=dict(marker='o', markersize=2, alpha=0.3))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_xticklabels(scenario_order, rotation=35, ha='right', fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_title(ylabel)
    if col == 'gp_carry_m':
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'${v:.0f}M'))

plt.suptitle('Portfolio Construction Scenarios — LP Outcome Distributions', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('scenario_boxplots.png', dpi=150, bbox_inches='tight')
plt.show()
"""))

new_cells.append(code("""\
# ── Plot 3: Risk/return scatter (P10 vs Median IRR) ───────────────────────
fig, ax = plt.subplots(figsize=(9, 6))

for name, color in zip(scenario_order, colors):
    clean = scenario_results[name]['net_irr_pct'].dropna()
    p10   = clean.quantile(0.10)
    med   = clean.median()
    ax.scatter(p10, med, s=120, color=color, zorder=5, label=name)
    ax.annotate(name, (p10, med), textcoords='offset points',
                xytext=(6, 4), fontsize=9)

ax.set_xlabel('P10 Net IRR (%)  ← downside risk')
ax.set_ylabel('Median Net IRR (%)')
ax.set_title('Risk / Return by Portfolio Construction\\n(higher-right = better)')
ax.legend(fontsize=8, loc='lower right')

# Add diagonal reference line
lo = min(scenario_results[n]['net_irr_pct'].quantile(0.10) for n in scenario_order) - 2
hi = max(scenario_results[n]['net_irr_pct'].median() for n in scenario_order) + 2
ax.plot([lo, hi], [lo, hi], '--', color='gray', linewidth=0.8, alpha=0.5)

plt.tight_layout()
plt.savefig('scenario_risk_return.png', dpi=150, bbox_inches='tight')
plt.show()

print("\\n✓ Scenario charts saved: scenario_irr_overlay.png, scenario_boxplots.png, scenario_risk_return.png")
"""))

nb.cells.extend(new_cells)

with open('fund_model.ipynb', 'w') as f:
    nbf.write(nb, f)

print(f"Appended Section 7 ({len(new_cells)} cells) to fund_model.ipynb")
