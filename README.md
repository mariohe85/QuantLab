# QuantLab

Local Django factor research workspace built on the `FactorsToday-V2-Proxy-Thematic` model:
locally constructed factor returns, historical stock exposures, portfolio factor analysis,
screens, optimization, monitoring, and monthly backtests.

## Boundaries

- Current S&P 500 membership is used historically, so results are survivorship-biased.
- Yahoo is an unofficial public source. Snapshots are checksummed for reproducibility, not
  institutional data quality.
- Every factor is rebuilt locally from adjusted closes and a versioned ticker list. Five style
  factors (SmallSize, Momentum, LowVolatility, BetaFactor, Liquidity) are ETF stand-ins for
  stock-universe characteristics and carry basis risk; `PeripheryCore` has no ETF analogue and
  is omitted.
- The ten thematic baskets have fixed membership. That is reproducible but can embed hindsight
  and stale-name bias.
- The UI reports actual factor counts per model level rather than implying broader coverage.
- Stock history uses the current S&P 500 membership and therefore cannot provide unbiased
  historical constituent results.

## Run

If PowerShell refuses a `.ps1` with "not digitally signed", allow scripts for this
session only, then start QuantLab:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\setup.ps1
.\start_quantlab.ps1
```

`-Scope Process` lasts until you close that terminal. It does not change the machine policy.
Alternatively, run a script without changing policy at all:

```powershell
powershell -ExecutionPolicy Bypass -File .\start_quantlab.ps1
```

`setup.ps1` creates `.venv`, installs `requirements.txt`, creates `data/db.sqlite3`, applies all
migrations, and runs Django's system check. It is safe to rerun for normal environment updates.
It does not download data unless explicitly requested.

The launcher starts the durable worker and web UI at <http://127.0.0.1:8086>. Django admin is
at <http://127.0.0.1:8086/admin/> (`admin` / `quantlab` unless `QUANTLAB_ADMIN_USER` or
`QUANTLAB_ADMIN_PASSWORD` is set). They can also be run separately with
`.\.venv\Scripts\python.exe manage.py quant_worker` and
`.\.venv\Scripts\python.exe manage.py runserver`.
`bootstrap_quantlab` remains available for deterministic offline testing, but it also creates
sample optimizer, screen, risk, and backtest records and is not part of the clean first-run flow.

### First run from an empty database

Three stages must run in order: prices, then factor returns, then the per-stock decomposition.
`setup.ps1` can run all three, using the IDs produced by each preceding stage:

```powershell
# Creates the environment and reproduces prices, 56 factor returns, four model
# levels, 24 month-ends of full-universe stock fits, and cash-rate history.
.\setup.ps1 -RebuildResearch
```

The rebuild defaults to prices from `2018-01-01` through tomorrow (Yahoo's end date is exclusive),
a 756-trading-day estimation window, and 24 month-ends. On this machine, prices take a few minutes
and the full decomposition takes roughly 30 minutes. Optional overrides include:

```powershell
.\setup.ps1 -RebuildResearch -StartDate 2018-01-01 -EndDate 2026-09-08 -Months 24 -Workers 12
```

If stage 1 fails with `CertificateVerifyError`, your machine cannot verify Yahoo's TLS chain.
This trusts the network, so only do it on a network you trust:

```powershell
.\setup.ps1 -RebuildResearch -SkipYahooTlsVerification
```

The snapshot's provenance records that TLS was not verified.

After stage 3, Factors, Signals, Stock, and Stock Selection are all populated. Portfolios,
optimizer studies, and backtests are created from the UI and need no bootstrap step.
`doc.md` documents the same three stages as queued jobs, every `monthly_exposures` parameter,
and the daily incremental refresh.

The underlying Yahoo/Wikipedia sources are live. Running this later reproduces the same model
methodology and pipeline, but cannot guarantee byte-identical factor values if Yahoo revises
adjusted history or Wikipedia's current S&P 500 membership changes.

To clear all local research rows while preserving the schema and repository files:

```powershell
python manage.py reset_quantlab --yes
```

## Platform

The default page is Factors. Factors, Stock, and Portfolio are separate top-level workspaces that
share one canonical model dataset and as-of context:

- **Factors** builds and validates the factor library, then shows family filters, returns,
  z-scores, provenance, coverage, and cumulative history.
- **Stock** searches the S&P 500 universe and compares nested Base, Base + Sector,
  Base + Sector + Industry, and All Factors models with full HAC inference, plus the month-end
  history of each fit and a per-factor beta time series.
- **Portfolio** has a Holdings view with a stock-by-factor matrix, plus Current Portfolio Exposure
  for weighted tilts, factor/specific/total risk, and contribution views.

**Signals** is the cross-sectional companion to the Factors workspace. It ranks every stock with
a latest model fit by the selected factor's beta z-score, supports factor-family, model, direction,
and sector views, and opens company beta/fit history. Missing 24-month company history is queued
on demand through the same durable exposure job.

**Stock Selection** combines multiple factor betas into a normalized weighted score. It
supports factor directions, top-X limits, sector/coverage/fit filters, exclusions, and manual
additions. Saved selections can create equal-weight portfolios and open them directly in
Portfolio or Optimizer. The old top-level Screens and Portfolios URLs redirect here.

Data, Stock, Portfolio, Signals, Stock Selection, Optimizer, Risk, Backtests, and Run History remain
available as independent pages. The legacy `/stocks/`
redirects there.

See `doc.md` for the runbook that produces factor returns and stock decomposition history.

`data/db.sqlite3` contains normalized universes, snapshots, securities, adjusted closes, factor definitions/builds/
observations/equations, monthly stock models and inference, portfolios and holdings, monthly
portfolio risk, limits and breach lifecycle, screens/results, optimization studies, backtest
rebalances/artifacts, and durable jobs. Market-data prices and provenance are SQL-only.

## Methodology

The `FactorsToday-V2-Proxy-Thematic` model builds 56 factors in a strict hierarchy: 15 base
factors (Market, five macro spreads, nine ETF style pairs), then 11 sectors, 10 industries,
10 countries, and 10 fixed thematic baskets. Each layer is stripped against the complete
preceding stack. Daily baskets are compounded to Friday, stripped with rolling 156-week OLS,
carried back to daily residuals, and scaled using prior 60-day volatility to a 10% target with
a 5× cap. Market remains unscaled. Full methodology, worked examples, and limitations are in
`factorv2.md`.

Each immutable model fingerprint has one canonical dataset. Daily runs append or correct factor
observations in that dataset; a methodology change creates a preserved new version. Execution
history remains in durable Jobs rather than appearing as separate Build 1/Build 2 datasets.
The dataset stores each factor's recipe and provenance badge along with equations, freshness,
and checksummed output.

Monthly stock models use trailing three years by default, ElasticNetCV selection, and conditional
HAC OLS inference; fixed-model OLS/HAC is available when classical inference is required. The
four catalogs are nested: Base (15), Base + Sector (26), Base + Sector + Industry (36), and
All Factors + Themes (56). Each stock/model/month has one row:
the current month's `as_of` advances to the latest trading date while prior months remain final.
Stored output includes beta, SE, t, p,
confidence interval, alpha, adjusted R², residual volatility, active factors, and coverage.
Portfolio decomposition persists factor/specific/total variance and attribution identities.

The Optimizer is a Standard allocator. It supports minimum variance, maximum return, maximum
Sharpe, and maximum return minus ½ risk (default: expected return − ½ variance) with either the PSD factor risk model
`BΩB′ + D` or Ledoit-Wolf stock covariance. The recommended return model prices stock betas using
editable annual factor-premium assumptions, adds a risk-free rate, and excludes noisy regression
alpha. Each assumption is a prior: every run blends it toward the trailing factor mean in
proportion to how precisely that window measures the mean, so backtest rebalances re-estimate
premiums causally instead of reusing one hardcoded number. **Equal Sharpe** instead assumes
every name has the same Sharpe ratio (`μ − r_f = k·σ`), which makes Maximum Sharpe the
Maximum Diversification Portfolio `w ∝ Σ⁻¹σ` and ignores historical means. Historical shrinkage
and user-supplied returns remain available. Maximum Sharpe uses excess return over the 13-week Treasury bill yield of
the as-of date (`manage.py fetch_risk_free_rates`), and diagnostics warn when flat forecasts reduce
a return objective to minimum variance. Controls are percentage-based, with absolute or
current-weight-relative stock limits and compact add-only factor and sector constraints.
Successful studies compare original versus optimized model estimates at the study date.
Stock Selection portfolios can queue an Approximate 24-month replay of that month’s ranked
names: equal-weight Original versus Optimized versus SPY.
After the 156-week warmup, each month’s optimize uses trailing volatility through that date.
Infeasible studies remain labelled `infeasible` rather than silently falling back to equal weight.

## Verify

```powershell
python manage.py check
python manage.py makemigrations --check
python manage.py test
```

Tests do not require network access.

## SQLite price storage

This `QuantLabv2` clone stores adjusted close prices and all research results in
`data/db.sqlite3`. No application workflow creates external market-data or result
artifacts. The original `QuantLab` directory remains the untouched baseline.
