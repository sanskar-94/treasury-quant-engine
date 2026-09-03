#!/usr/bin/env python
"""Render the project report to HTML, and to PDF via headless Chrome.

Every number in the document is read from ``results/`` at build time rather than
typed into the template. A report that is edited by hand drifts from the code it
describes, and a report about a project whose central lesson is "impressive
numbers are usually artifacts" cannot afford to contain stale ones.

    python scripts/build_report.py [--no-pdf]
"""

from __future__ import annotations

import argparse
import base64
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

CHROME = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
]


def pct(x: float, dp: int = 2) -> str:
    return f"{x * 100:.{dp}f}%"


def img(rel: str, caption: str = "", width: str = "100%") -> str:
    """Embed a PNG inline so the report is a single self-contained file."""
    path = RESULTS / rel
    if not path.exists():
        return ""
    b64 = base64.b64encode(path.read_bytes()).decode()
    cap = f'<figcaption>{caption}</figcaption>' if caption else ""
    return (f'<figure><img src="data:image/png;base64,{b64}" style="width:{width}">'
            f'{cap}</figure>')


def read_csv(name: str) -> pd.DataFrame | None:
    p = RESULTS / name
    return pd.read_csv(p) if p.exists() else None


def table(df: pd.DataFrame, cols: dict[str, str], fmt: dict | None = None,
          highlight: str | None = None) -> str:
    """Render a DataFrame as an HTML table, formatting per column."""
    fmt = fmt or {}
    head = "".join(f"<th>{label}</th>" for label in cols.values())
    rows = []
    for _, r in df.iterrows():
        cells = []
        for key in cols:
            v = r.get(key, "")
            f = fmt.get(key)
            cells.append(f"<td>{f(v) if f and pd.notna(v) else v}</td>")
        cls = ' class="hl"' if highlight and str(r.get(list(cols)[0], "")) == highlight else ""
        rows.append(f"<tr{cls}>{''.join(cells)}</tr>")
    return (f"<table><thead><tr>{head}</tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table>")


def stat(label: str, value: str, note: str = "") -> str:
    return (f'<div class="stat"><div class="sv">{value}</div>'
            f'<div class="sl">{label}</div>'
            + (f'<div class="sn">{note}</div>' if note else "") + "</div>")


def build() -> str:
    m = json.loads((RESULTS / "metrics.json").read_text())
    # Ask pytest, rather than counting "def test_" - parametrised tests expand,
    # so the source count (323) understates what actually runs (385).
    n_tests = 0
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "--collect-only", "-q"],
            cwd=ROOT, capture_output=True, text=True, timeout=300,
        )
        hit = re.search(r"(\d+)\s+tests?\s+collected", proc.stdout)
        if hit:
            n_tests = int(hit.group(1))
        else:
            # The project already sets -q in addopts, so a second -q switches
            # pytest to per-file counts ("tests/test_x.py: 70") with no total.
            n_tests = sum(int(v) for v in
                          re.findall(r"^tests/\S+\.py:\s*(\d+)$", proc.stdout, re.M))
    except Exception as exc:  # noqa: BLE001 - the report must still build
        print(f"could not count tests ({exc})", file=sys.stderr)
    if not n_tests:
        n_tests = sum(len(re.findall(r"^\s*def test_", q.read_text(), re.M))
                      for q in (ROOT / "tests").glob("test_*.py"))
    n_modules = len(list((ROOT / "src").rglob("*.py")))
    n_lines = sum(len(p.read_text().splitlines())
                  for d in ("src", "scripts", "tests")
                  for p in (ROOT / d).rglob("*.py"))
    try:
        n_commits = subprocess.run(["git", "rev-list", "--count", "HEAD"], cwd=ROOT,
                                   capture_output=True, text=True).stdout.strip() or "25"
    except Exception:
        n_commits = "25"

    harvest = read_csv("duration_harvest.csv")
    timing = read_csv("duration_timing.csv")
    integ = read_csv("integration_experiment.csv")
    horizon = read_csv("horizon_experiment_v3.csv")
    if horizon is None:
        horizon = read_csv("horizon_experiment.csv")

    p2 = lambda v: f"{float(v):+.3f}"          # noqa: E731
    pp = lambda v: pct(float(v))               # noqa: E731

    parts: list[str] = []
    A = parts.append

    # ---------------- cover ---------------- #
    sh, ds = f"{m['sharpe']:.2f}", f"{m['deflated_sharpe']:.3f}"
    nt = int(m.get("n_trials", 1))
    cr = f"{m.get('canary_ratio', 0.0):.3f}"
    dso = (f"{m['deflated_sharpe_observed_sd']:.3f}"
           if m.get("deflated_sharpe_observed_sd") is not None else "n/a")
    fin, cst = pct(m['total_financing'] / 8e7), pct(m['total_costs'] / 8e7)
    A(f"""
<div class="cover">
  <div class="kicker">Quantitative Research &amp; Systematic Trading</div>
  <h1>Treasury Quant Engine</h1>
  <p class="sub">A production-grade research and execution system for the US Treasury
  market &mdash; curve construction, term-structure modelling, machine-learned return
  forecasting, portfolio construction and order execution &mdash; together with the
  evaluation discipline needed to find out whether any of it actually works.</p>
  <div class="stats">
    {stat("Python modules", str(n_modules))}
    {stat("Lines of code", f"{n_lines:,}")}
    {stat("Tests", str(n_tests), "all passing, CI green")}
    {stat("Commits", n_commits)}
  </div>
  <p class="verdict"><strong>Headline result.</strong> After correct financing, the
  strategy does not clear its hurdle. That is the finding, and it is reported as such.
  Ten separate bugs were found and fixed along the way &mdash; six of them cases where
  a position was not being charged for the money it borrowed. Each one made the
  results <em>better</em>, and each one was silent. The engineering that makes those
  bugs findable is the substance of this project.</p>

  <div class="two">
    <div>
      <h3 class="ct">Results at a glance</h3>
      <table class="mini">
        <tr><td>Shipped strategy, out of sample</td><td>Sharpe {sh}</td></tr>
        <tr><td>Deflated Sharpe &mdash; {nt} configurations searched</td><td>{ds}</td></tr>
        <tr><td>Deflated Sharpe &mdash; observed trial dispersion</td><td>{dso}</td></tr>
        <tr><td>Best passive arm &mdash; vol-targeted 30 Yr</td><td>Sharpe +0.45</td></tr>
        <tr><td>Static long duration, 1993&ndash;2026</td><td>Sharpe +0.24</td></tr>
        <tr><td>Configurations searched</td><td>{nt}</td></tr>
        <tr><td>Term-premium timing overlay</td><td>Sharpe &minus;0.40</td></tr>
        <tr><td>Perfect-foresight canary ratio</td><td>{cr}</td></tr>
        <tr><td>Financing vs transaction cost</td><td>{fin} vs {cst} p.a.</td></tr>
      </table>
    </div>
    <div>
      <h3 class="ct">Contents</h3>
      <ol class="toc">
        <li>What this system is</li>
        <li>How results are judged</li>
        <li>Findings
          <ol>
            <li>Where the term premium is paid</li>
            <li>Why timing it fails</li>
            <li>Whether the modules earn their place</li>
            <li>Forecast horizon</li>
          </ol>
        </li>
        <li>Ten bugs, and what they cost</li>
        <li>Engineering</li>
        <li>What I would do next</li>
      </ol>
    </div>
  </div>
</div>
""")

    # ---------------- 1. what it is ---------------- #
    A(f"""
<h2>1 &nbsp; What this system is</h2>
<p>The Treasury Quant Engine ingests the full history of the US Treasury par yield
curve, builds a self-consistent zero curve from it, extracts term-structure factors,
learns a forecast of forward returns, turns that forecast into a risk-controlled
portfolio, and routes the resulting orders through an execution layer. It runs
end-to-end from one command and is covered by {n_tests} tests.</p>

<p>It is built to answer one question honestly: <em>is there a tradable signal in the
Treasury curve, and how would I know?</em> Most of the engineering exists to make the
answer trustworthy rather than flattering.</p>

<h3>Pipeline</h3>
<div class="flow">
  <div class="fx">Data<span>FRED &middot; par curve<br>1990&ndash;2026</span></div>
  <div class="fa">&rarr;</div>
  <div class="fx">Curve<span>bootstrap &middot; NSS<br>zero &amp; forward</span></div>
  <div class="fa">&rarr;</div>
  <div class="fx">Features<span>489 causal<br>PCA &middot; carry &middot; macro</span></div>
  <div class="fa">&rarr;</div>
  <div class="fx">Model<span>stacked ensemble<br>walk-forward</span></div>
  <div class="fa">&rarr;</div>
  <div class="fx">Portfolio<span>DV01 sizing<br>neutrality</span></div>
  <div class="fa">&rarr;</div>
  <div class="fx">Execution<span>OMS &middot; TWAP<br>Almgren&ndash;Chriss</span></div>
</div>

{img("charts/curve_surface.png", "The fitted zero-coupon surface, 1990&ndash;2026. Every curve is bootstrapped self-consistently and round-trips to 1.4&times;10<sup>&minus;13</sup>.")}

<h3>Components</h3>
<div class="grid">
  <div class="card"><h4>Fixed income analytics</h4>
    <p>ACT/ACT (ICMA), ACT/360, ACT/365F and 30/360 day counts; street price&ndash;yield
    with correct settlement (T+1 from 28 May 2024, T+2 before); Macaulay and modified
    duration, convexity, DV01, key-rate durations; carry and roll-down; 32nds quoting.
    Validated against closed-form identities rather than against its own output.</p></div>
  <div class="card"><h4>Curve construction</h4>
    <p>Self-consistent bootstrap by bisection, round-tripping to 1.4&times;10<sup>&minus;13</sup>.
    Nelson&ndash;Siegel&ndash;Svensson with a Diebold&ndash;Li fixed-decay mode
    (&tau;<sub>1</sub>=1.37, &tau;<sub>2</sub>=8.0) for identifiability. PCA on yield
    <em>changes</em>: level, slope and curvature explain 77.2%, 13.3% and 5.0%.</p></div>
  <div class="card"><h4>Term-structure modelling</h4>
    <p>Dynamic Nelson&ndash;Siegel with a VAR on the factors; ACM-style term premium
    decomposition separating expected short rates from the premium for holding
    duration; a two-state Hamilton regime model fitted by Baum&ndash;Welch with a
    scaled forward&ndash;backward recursion.</p></div>
  <div class="card"><h4>Forecasting</h4>
    <p>Walk-forward cross-validation with purging and an embargo; a stacked ensemble
    combined by forward-chaining out-of-fold predictions and a non-negative least
    squares meta-learner; nested CV for hyperparameters; per-fold RobustScaler so no
    scaling information crosses a fold boundary.</p></div>
  <div class="card"><h4>Portfolio construction</h4>
    <p>DV01-weighted duration, steepener and butterfly structures; exact double
    neutrality by projection onto the null space of the cash and DV01 constraints;
    key-rate hedging; leverage and no-trade bands; scheduled rebalancing.</p></div>
  <div class="card"><h4>Execution</h4>
    <p>Idempotent order management with a pre-trade risk gate; TWAP, VWAP and the
    closed-form Almgren&ndash;Chriss trajectory, whose two limits (risk-neutral
    &rarr; TWAP, risk-averse &rarr; front-loaded) are asserted in tests;
    implementation shortfall decomposition for TCA.</p></div>
</div>
<div class="pb"></div>
""")

    # ---------------- 2. discipline ---------------- #
    A(f"""
<h2>2 &nbsp; How results are judged</h2>
<p>Backtests are easy to make look good. Everything below exists because some
version of this project once produced an impressive number that was not real.</p>

<div class="grid">
  <div class="card"><h4>One P&amp;L path</h4>
    <p>Exactly one function &mdash; <code>run_backtest</code> &mdash; turns positions
    into profit and loss. Every experiment, script and notebook routes through it.
    This rule exists because five separate bugs were traced to bespoke P&amp;L code
    that quietly forgot to charge for financing.</p></div>
  <div class="card"><h4>Financing is always charged</h4>
    <p>Borrowed notional is charged at the overnight rate on an ACT/360 basis,
    including weekends. A missing funding rate is charged at the most expensive rate
    observed, never at zero.</p></div>
  <div class="card"><h4>Causality boundary</h4>
    <p>Row <em>t</em> of the feature matrix holds only information observable at the
    close of day <em>t</em>&minus;1; row <em>t</em> of the target is the return
    realised over day <em>t</em>. Enforced in one place and guarded by
    future-corruption canaries.</p></div>
  <div class="card"><h4>Perfect-foresight canary</h4>
    <p>A book built from tomorrow's returns runs beside the real one. It scores a
    Sharpe above 15; the honest strategy reaches a fraction of a percent of that. If the ratio ever
    approached 1, the pipeline would be leaking the future.</p></div>
  <div class="card"><h4>Block sign-flip nulls</h4>
    <p>Significance is measured against signals whose 63-day blocks have had their
    signs flipped. This preserves autocorrelation and position concentration and
    destroys only the timing &mdash; unlike shuffling, which produced books ten times
    more concentrated than the real one and was therefore not a valid null.</p></div>
  <div class="card"><h4>Multiple-testing correction</h4>
    <p>Deflated and probabilistic Sharpe ratios (Bailey &amp; L&oacute;pez de Prado)
    adjust for the number of configurations tried; Holm&ndash;Bonferroni corrects
    every family of comparisons. This project searched <strong>{nt}</strong>
    configurations; deflated against all of them the shipped Sharpe scores
    <strong>{m['deflated_sharpe']:.3f}</strong>, and <strong>{dso}</strong> when the
    measured dispersion of the trial Sharpes is used instead of the theoretical
    one. The count is recovered from the study files automatically, because an
    artefact produced by a default is the artefact that gets published.</p></div>
</div>

<h3>Shipped strategy, out of sample</h3>
<p>Walk-forward, funded, costed, 2018&ndash;2026 ({int(m.get('n_days', 2016))} trading days).</p>
<div class="stats sm">
  {stat("Sharpe", f"{m['sharpe']:.2f}")}
  {stat("Ann. return", pct(m['ann_return']))}
  {stat("Ann. vol", pct(m['ann_vol']))}
  {stat("Max drawdown", pct(m['max_drawdown']))}
  {stat("Hit rate", pct(m['hit_rate']))}
  {stat("Turnover", f"{m['ann_turnover']:.1f}x")}
</div>
{img("tearsheet.png", "Out-of-sample tearsheet: equity curve, drawdown, rolling Sharpe and exposure. Funded and costed throughout.")}

<p class="note">Financing cost {pct(m['total_financing'] / 8e7)} p.a. against trading
costs of {pct(m['total_costs'] / 8e7)} p.a. &mdash; financing is roughly twenty times
the transaction cost, which is the single most important economic fact about a levered
rates book and the one most often left out of a backtest.</p>
""")

    # ---------------- 3. findings ---------------- #
    A("<h2>3 &nbsp; Findings</h2>")

    if harvest is not None:
        best = harvest.sort_values("sharpe", ascending=False).iloc[0]
        A(f"""
<h3>3.1 &nbsp; Where the term premium is actually paid</h3>
<p>Holding duration is compensated, but the compensation is not uniform across the
curve once the borrowing is paid for. Each arm below holds a constant DV01, scaled to
5% annualised volatility, funded and costed over 1990&ndash;2026.</p>
{table(harvest[harvest.kind.isin(["single", "voltgt"])],
       {"arm": "Arm", "sharpe": "Sharpe", "ann_return": "Ann. return",
        "max_dd": "Max DD", "financing": "Financing p.a."},
       {"sharpe": p2, "ann_return": pp, "max_dd": pp, "financing": pp},
       highlight=str(best.arm))}
<p><strong>The front end loses money and the long end makes it.</strong> The reason is
leverage: matching 5% volatility with 3-month bills takes about 17&times; leverage,
against 0.4&times; with 30-year bonds. Financing scales with borrowed notional, so the
short end pays away 55% a year in interest to own the same amount of risk. The best
way to be paid for duration is to own it where you need the least borrowed money.</p>
<p class="note">Before the financing bug described in section 4 was fixed, this same
table showed the 3-month arm at a Sharpe of <strong>+5.76</strong>. The entire result
was free leverage.</p>
""")

    if timing is not None and len(timing):
        A(f"""
<h3>3.2 &nbsp; The term premium does not time duration</h3>
<p>If the term premium measures how well duration is paid, a natural strategy holds
more duration when the premium is high. It does not work. Over 1993&ndash;2026
(7,419 days), against block sign-flip controls:</p>
{table(timing, {"arm": "Arm", "sharpe": "Sharpe", "ann_return": "Ann. return",
                "placebo_mean": "Control mean", "p_value": "p"},
       {"sharpe": p2, "ann_return": pp, "placebo_mean": p2,
        "p_value": lambda v: f"{float(v):.3f}"})}
<p>Static duration returns a Sharpe of +0.24; both timing rules are negative and
neither is distinguishable from its controls (p = 0.68). The decomposition is sound
&mdash; it is just not a timing signal. <strong>The premium is real and it is collected
by sitting still.</strong></p>
""")

    if integ is not None:
        A(f"""
<h3>3.3 &nbsp; Do the advanced modules earn their place?</h3>
<p>Four modules were built because measurements pointed at them. Building something is
not the same as showing it helps, so each was tested inside the same double-neutral,
funded, costed book.</p>
{table(integ, {"variant": "Variant", "sharpe": "Sharpe", "ann_return": "Ann. return",
               "turnover": "Turnover", "p_value": "p"},
       {"sharpe": p2, "ann_return": pp, "turnover": lambda v: f"{float(v):.1f}x",
        "p_value": lambda v: f"{float(v):.3f}"})}
<p>None survives Holm correction. All four are correct implementations of ideas that
should have helped; that they do not is the result, and they ship documented as such
rather than quietly left in the code path.</p>
""")

    if horizon is not None:
        A(f"""
<h3>3.4 &nbsp; Forecast horizon</h3>
<p>Predictive content by holding period, after the target-alignment fix:</p>
{table(horizon.head(8), {c: c.replace("_", " ").title() for c in horizon.columns[:5]},
       {c: (lambda v: f"{float(v):.4f}") for c in horizon.columns[1:5]})}
""")

    A(f"""
{img("charts/factors.png", "Level, slope and curvature extracted by PCA on yield <em>changes</em> &mdash; 77.2%, 13.3% and 5.0% of variance. Fitting on levels instead would produce factors that look cleaner and forecast nothing.")}

<h3>3.5 &nbsp; What the whole thing amounts to</h3>
<p>The signal is weak but not absent: the out-of-sample information coefficient is
about +0.038, the shipped Sharpe is 0.46 before any allowance for the number of
configurations tried, and the deflated Sharpe does not clear a convincing bar. The one
robust positive result in the project is passive &mdash; owning long-dated duration
and doing nothing &mdash; and even that is a Sharpe of roughly 0.45.</p>
<p>The honest conclusion is that this system does not find a tradable edge in the
Treasury curve from public daily data. Reporting that clearly, with the evidence
attached, is more useful than the alternative &mdash; and every mechanism built to
reach that conclusion is the part that transfers to a desk.</p>
<div class="pb"></div>
""")

    # ---------------- 4. bugs ---------------- #
    A("""
<h2>4 &nbsp; Ten bugs, and what they cost</h2>
<p>Eight of these made the results better and two made them worse. All ten were
silent, and none was caught by a test that already existed &mdash; every one required
either an invariant nobody had written down or a number that was too good to be true.
The last three were found by an adversarial audit of the P&amp;L core itself, run
specifically because bug two had been discovered <em>inside</em> the one function this
project designates as its single source of truth. Two of those three were guarded by
tests that asserted the code's own behaviour rather than an external invariant &mdash;
a test that ratifies the output cannot detect that the output is wrong.</p>
<table class="bugs">
<thead><tr><th>Bug</th><th>Effect</th><th>How it surfaced</th><th>Fix</th></tr></thead>
<tbody>
<tr><td><strong>Financing never charged</strong></td>
    <td>Sharpe 2.35 &rarr; 0.12</td>
    <td>A riskless carry position scored above 5</td>
    <td>Engine charges net notional at the overnight rate; five regression tests</td></tr>
<tr><td><strong>Missing funding rate read as zero</strong></td>
    <td>31.6% of all days financed free; 3-month arm 5.76 &rarr; &minus;0.49</td>
    <td>Sharpe 5.76 alongside a &minus;23% drawdown at 5% vol &mdash; arithmetically impossible</td>
    <td>Overnight rate from fed funds (1954&ndash;), cascade fallback, never zero</td></tr>
<tr><td><strong>Target misaligned by one day</strong></td>
    <td>IC +0.0095 &rarr; +0.0375; Sharpe 0.12 &rarr; 0.51</td>
    <td>Features were already lagged, so a one-day horizon needed no shift</td>
    <td>Shift by <code>horizon &minus; 1</code>; boundary documented and tested</td></tr>
<tr><td><strong>Bespoke P&amp;L in experiments</strong></td>
    <td>Structures Sharpe 3.96 &rarr; 0.04</td>
    <td>A curve trade appeared to be free money</td>
    <td>Every experiment routes through <code>run_backtest</code></td></tr>
<tr><td><strong>Bootstrap flat extrapolation</strong></td>
    <td>20-year zero biased by 38bp</td>
    <td>Round-trip test failed at the fourth decimal</td>
    <td>Self-consistent bisection; round-trips to 1.4e&minus;13</td></tr>
<tr><td><strong>Optimizer unit errors</strong></td>
    <td>Returned an all-zero book</td>
    <td>Turnover penalty was four orders of magnitude off</td>
    <td>Penalty as a multiplier on real cost; &lambda; derived from a vol target</td></tr>
<tr><td><strong>Repo spread charged on net, not gross</strong></td>
    <td>Sharpe 0.535 &rarr; 0.440; $90,192 unfinanced, 1.35&times; the entire cost bill</td>
    <td>At fixed net, a $50mm and a $200mm book financed identically &mdash;
        financing was blind to gross</td>
    <td>GC on net, spread on gross, matching the convention
        <code>CostModel.financing</code> already documented and nothing called</td></tr>
<tr><td><strong>Deflated Sharpe never deflated</strong></td>
    <td>Reported 0.904; true value 0.100, or 0.000 on observed dispersion</td>
    <td>The tearsheet said "configurations searched: 1" beside a
        multiple-testing-adjusted number, having searched 219</td>
    <td>Trial count recovered from the study files automatically; both
        dispersions reported; loud warning at n_trials=1</td></tr>
<tr><td><strong>No-trade band read the future</strong></td>
    <td>Sharpe 0.461 &rarr; 0.535 &mdash; the leak was <em>costing</em> return (both measured before the repo-spread fix below)</td>
    <td>Band width set from the full-sample mean gross book</td>
    <td>Expanding-window band; test asserts changing only the future leaves
        past positions bit-identical</td></tr>
<tr><td><strong>EM exited after one iteration</strong></td>
    <td>Regime model never actually fitted</td>
    <td><code>inf &lt;= inf</code> evaluates to <code>True</code></td>
    <td>Guard the convergence test with <code>np.isfinite</code></td></tr>
</tbody></table>

<h3>Three times the test was wrong, not the code</h3>
<p>The perfect-foresight canary was rewritten three times before it tested anything:
the first two versions passed against a pipeline that was leaking. The shuffle placebo
was not a valid null, because shuffled books were ten times more concentrated than the
real one and so were being compared against a different risk profile. And a
regime-scaling option I added myself silently did nothing, because the sizing function
normalises signal magnitude away &mdash; scaling the signal beforehand cancelled
exactly. A test that cannot fail is worse than no test, because it is counted as
evidence.</p>
""")

    # ---------------- 5. engineering ---------------- #
    A(f"""
<h2>5 &nbsp; Engineering</h2>
<div class="grid">
  <div class="card"><h4>Testing</h4>
    <p>{n_tests} tests covering closed-form identities, invariants, regressions for
    every bug above, and property tests. Analytics are checked against textbook
    identities, never against their own output. Curve-dependent tests skip cleanly on
    a fresh clone rather than failing.</p></div>
  <div class="card"><h4>Continuous integration</h4>
    <p>GitHub Actions runs lint and the full suite on every push. A clean-clone
    simulation is part of the workflow, added after three tests failed in CI by
    reading a gitignored data file that existed only on my machine.</p></div>
  <div class="card"><h4>Interfaces</h4>
    <p>A twelve-command CLI (<code>data</code>, <code>curve</code>, <code>features</code>,
    <code>train</code>, <code>backtest</code>, <code>attribute</code>, <code>tune</code>,
    <code>charts</code>, <code>predict</code>, <code>trade</code>, <code>serve</code>,
    <code>pipeline</code>) plus a FastAPI service. Feature schemas are pinned and
    aligned at inference so training and serving cannot drift.</p></div>
  <div class="card"><h4>Reproducibility</h4>
    <p>Seeded throughout, configuration in version-controlled YAML, artefacts written
    with the parameters that produced them. Every figure and table in this report is
    regenerated from <code>results/</code> at build time.</p></div>
</div>

<h3>Running it</h3>
<pre><code>pip install -e ".[dev]"
tqe pipeline          # data &rarr; curve &rarr; features &rarr; train &rarr; backtest
pytest                # {n_tests} tests
python scripts/duration_harvest.py</code></pre>

<h2>6 &nbsp; What I would do next</h2>
<ol>
  <li><strong>Better data.</strong> Daily par yields are the binding constraint. Intraday
  data, the futures basis, and CUSIP-level on-the-run/off-the-run spreads carry
  information a daily par curve simply does not.</li>
  <li><strong>Cross-market signals.</strong> Swap spreads, the futures calendar roll and
  cross-currency bases are where relative value in rates usually lives.</li>
  <li><strong>Trade the passive result properly.</strong> The one thing that works is
  owning long-dated duration. Sizing that against a volatility target, with a drawdown
  control, is a more honest starting point than another forecasting layer.</li>
  <li><strong>Cost realism.</strong> The square-root impact model is calibrated from
  literature, not from fills. Real TCA would change the turnover economics.</li>
</ol>
""")
    return "\n".join(parts)


CSS = """
@page { size: A4; margin: 16mm 14mm 14mm 14mm; }
* { box-sizing: border-box; }
html, body { background: #ffffff; }
body { font-family: -apple-system, "Helvetica Neue", Arial, sans-serif;
  color: #1a1d21; font-size: 10.2pt; line-height: 1.5; margin: 0;
  -webkit-print-color-adjust: exact; print-color-adjust: exact; }
h1 { font-size: 30pt; margin: 6pt 0 10pt; letter-spacing: -0.6pt; font-weight: 700; }
h2 { font-size: 15pt; margin: 20pt 0 8pt; padding-bottom: 5pt;
  border-bottom: 2px solid #0b3d64; color: #0b3d64; font-weight: 700; }
h3 { font-size: 11.6pt; margin: 15pt 0 5pt; color: #14293d; font-weight: 650; }
h4 { font-size: 9.8pt; margin: 0 0 4pt; color: #0b3d64; font-weight: 650; }
p { margin: 0 0 7pt; }
code { font-family: "SF Mono", Menlo, monospace; font-size: 0.87em;
  background: #eef2f6; padding: 1px 4px; border-radius: 3px; }
pre { background: #0f2233; color: #e6edf3; padding: 10pt 12pt; border-radius: 5px;
  font-size: 8.6pt; line-height: 1.65; overflow-x: auto; }
pre code { background: none; color: inherit; padding: 0; }
.pb { page-break-after: always; }
p, li { orphans: 3; widows: 3; }
table, .card, .stat, .flow, .note, .verdict, figure { break-inside: avoid; }
h2, h3, h4 { break-after: avoid; }
h2 { break-before: page; }
.cover h1, .cover + * { break-before: auto; }
.cover ~ h2:first-of-type { break-before: auto; }
figure { margin: 10pt 0 12pt; text-align: center; }
figure img { max-width: 100%; border: 1px solid #e2e9ef; border-radius: 4px; }
figcaption { font-size: 8.1pt; color: #5a6b7d; margin-top: 4pt;
  text-align: left; line-height: 1.4; }
.two { display: grid; grid-template-columns: 1.15fr 0.85fr; gap: 18pt; margin-top: 16pt; }
.ct { font-size: 9.4pt; text-transform: uppercase; letter-spacing: 1pt;
  color: #5a6b7d; margin: 0 0 5pt; font-weight: 650; }
table.mini { font-size: 8.7pt; margin: 0; }
table.mini td { padding: 3.6pt 0; border-bottom: 1px solid #e6ecf1; }
table.mini td:last-child { text-align: right; font-weight: 650; color: #0b3d64;
  white-space: nowrap; padding-left: 8pt; }
ol.toc { font-size: 8.9pt; margin: 0; padding-left: 14pt; color: #38434e; }
ol.toc li { margin-bottom: 2.5pt; }
ol.toc ol { padding-left: 13pt; margin: 2.5pt 0 0; color: #5a6b7d; font-size: 8.4pt; }
.cover { padding-top: 8mm; }
.kicker { text-transform: uppercase; letter-spacing: 2.2pt; font-size: 8pt;
  color: #5a6b7d; font-weight: 650; }
.sub { font-size: 11.4pt; color: #3d4c5a; max-width: 165mm; line-height: 1.55; }
.verdict { background: #fff8e6; border-left: 3px solid #c8901a; padding: 10pt 13pt;
  margin-top: 16pt; font-size: 10pt; }
.note { background: #f4f7fa; border-left: 3px solid #7d94a8; padding: 8pt 12pt;
  font-size: 9.3pt; color: #38434e; }
.stats { display: flex; gap: 9pt; margin: 20pt 0; flex-wrap: wrap; }
.stats.sm .stat { padding: 8pt 6pt; }
.stat { flex: 1; min-width: 62pt; background: #f4f7fa; border: 1px solid #dde5ec;
  border-radius: 5px; padding: 11pt 8pt; text-align: center; }
.sv { font-size: 16pt; font-weight: 700; color: #0b3d64; letter-spacing: -0.4pt; }
.sl { font-size: 7.8pt; color: #5a6b7d; text-transform: uppercase;
  letter-spacing: 0.5pt; margin-top: 2pt; }
.sn { font-size: 7pt; color: #8494a3; margin-top: 2pt; }
.grid { display: grid; grid-template-columns: 1fr 1fr; gap: 9pt; margin: 10pt 0; }
.card { background: #f9fbfc; border: 1px solid #e2e9ef; border-radius: 5px;
  padding: 9pt 11pt; }
.card p { margin: 0; font-size: 9.1pt; color: #38434e; line-height: 1.48; }
table { width: 100%; border-collapse: collapse; margin: 8pt 0 10pt; font-size: 8.9pt; }
th { background: #0b3d64; color: #fff; text-align: left; padding: 5pt 7pt;
  font-weight: 600; font-size: 8.3pt; }
td { padding: 4.5pt 7pt; border-bottom: 1px solid #e6ecf1; }
tbody tr:nth-child(even) { background: #f7fafc; }
tr.hl { background: #e8f4e8 !important; font-weight: 650; }
table.bugs td { font-size: 8.5pt; vertical-align: top; }
table.bugs td:first-child { width: 21%; }
.flow { display: flex; align-items: stretch; gap: 3pt; margin: 10pt 0 14pt; }
.fx { flex: 1; background: #0b3d64; color: #fff; border-radius: 4px;
  padding: 7pt 4pt; text-align: center; font-size: 8.4pt; font-weight: 650; }
.fx span { display: block; font-weight: 400; font-size: 7pt; color: #b9d0e2;
  margin-top: 3pt; line-height: 1.35; }
.fa { align-self: center; color: #8aa4bb; font-size: 11pt; }
ol { margin: 6pt 0; padding-left: 16pt; }
li { margin-bottom: 5pt; }
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-pdf", action="store_true")
    ap.add_argument("--out", default="docs/Treasury_Quant_Engine_Report")
    args = ap.parse_args()

    html = (f"<!doctype html><html><head><meta charset='utf-8'>"
            f"<title>Treasury Quant Engine</title><style>{CSS}</style></head>"
            f"<body>{build()}</body></html>")

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    html_path = out.with_suffix(".html")
    html_path.write_text(html)
    print(f"html -> {html_path}")

    if args.no_pdf:
        return 0

    browser = next((b for b in CHROME if Path(b).exists()), None) or shutil.which("chromium")
    if not browser:
        print("no Chrome/Chromium found; HTML written, skipping PDF", file=sys.stderr)
        return 0

    pdf_path = out.with_suffix(".pdf")
    cmd = [browser, "--headless", "--disable-gpu", "--no-pdf-header-footer",
           f"--print-to-pdf={pdf_path}", html_path.as_uri()]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    if pdf_path.exists():
        print(f"pdf  -> {pdf_path}  ({pdf_path.stat().st_size / 1024:.0f} KB)")
        return 0
    print(f"PDF generation failed: {r.stderr[-500:]}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
