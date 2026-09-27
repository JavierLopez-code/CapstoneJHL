# Systematic Multi-Asset Portfolio with Regime-Scaled Exposure

**Capstone Project — Initial Report & Exploratory Data Analysis (Module 20.1)**
**Author:** Javier Hernan Lopez

---

## Research Question

Can a monthly-rebalanced, multi-asset ETF portfolio achieve better **risk-adjusted returns** (Sharpe, Sortino, maximum drawdown) than passively holding the S&P 500 — participating in long-term market growth with materially lower volatility and controlled drawdowns?

The objective is explicitly *not* to beat SPY in raw return. Success is defined on risk-adjusted metrics, declared before any data was touched.

## Data

| Source | Content |
|---|---|
| [yfinance](https://github.com/ranaroussi/yfinance) | Daily dividend-adjusted prices for 7 liquid ETFs: SPY, VEA, XLE, XLU (equity), IEF, SHY (Treasuries), GLD (gold), each from its inception (VEA: 2007) |
| [FRED](https://fred.stlouisfed.org/) | Market stress indicators: VIX, VIX3M, 10y–2y Treasury spread (T10Y2Y) |

Daily data is used for monitoring and indicators; returns are **aggregated to weekly for covariance estimation**, because daily returns understate the SPY–VEA correlation due to the international close-time lag (a data-frequency artifact, not an economic fact).

## Repository Structure

```
notebooks/
  01_data.ipynb          — download, cleaning, adjusted-returns validation
  02_exploration.ipynb   — EDA: ACF diagnostics + K-means clustering of assets
  03_covariance.ipynb    — rolling covariance: sample + Ledoit-Wolf shrinkage
  04_ewma.ipynb          — covariance: EWMA variants (λ = 0.94 / 0.97 / 0.99)
  05_optimization.ipynb  — baseline models: MinVar-with-caps vs Risk Parity
data/                    — raw and processed data (separate folders)
outputs/                 — point-in-time monthly outputs (parquet/csv,
                           date column = "as computed at")
```

Every notebook opens with a markdown cell stating the specification for that step. All computed objects (covariances, weights) are **point-in-time**: each month's output uses only data available at that date and is stored as computed — never overwritten with recomputed history. Without this discipline, any later backtest would be fiction.

## Data Cleaning & Validation (Notebook 01)

- Prices adjusted for dividends and splits; adjustment verified by signature: IEF shows ~3.6% annualized total return where unadjusted prices would show ~1% (the difference is the dividend yield — its presence confirms the adjustment is correct).
- Missing values handled by each asset's own inception (no artificial backfill); non-trading days aligned across assets before computing returns; duplicate timestamps checked and absent.
- Correlation sanity checks (mandatory before anything downstream): SPY–VEA ≈ 0.88 on weekly returns, SPY–IEF ≈ −0.22, SHY ≈ flat vs all risk assets (its 0.78 correlation with IEF is expected — both are Treasuries), GLD weakly linked to both families. Any violation here is treated as a data bug first, an economic finding second.
- Outlier analysis: the largest daily moves (±15%+) inspected and cross-checked against known market events — all fall on crisis sessions of 2008 (GFC) and 2020 (COVID). They are genuine market moves, not data errors — so they are **kept**: clipping or winsorizing them would erase exactly the risk this project is designed to manage.

## Feature Engineering

Raw prices are transformed into the variables the system actually consumes:

- **Weekly log returns** per asset (covariance input) and squared returns (volatility proxy).
- **Rolling 3-year covariance matrices**, later shrunk (see below) — the central engineered object, 193 point-in-time monthly matrices.
- **Market-condition features** for the upcoming regime detector: realized volatility, trailing returns, and average cross-asset equity correlation.
- **12-1 momentum** (12-month return skipping the most recent month), to be used only as a capped post-optimization tilt.

## EDA — Key Findings (Notebook 02)

**1. Return direction is (economically) unpredictable; volatility is not.**
ACF of weekly returns: Ljung-Box is statistically significant for several assets, but the largest |autocorrelation| is ≈ 0.07 — under 1% of next-week variance, below transaction-cost thresholds. *Statistically detectable ≠ economically exploitable.* ACF of squared returns: |acf| up to ≈ 0.26 with p-values as low as 10⁻¹²²; volatility clusters persistently for 4–6 weeks.

**Design consequence (the project's empirical foundation):** the optimizer never receives expected-return forecasts. The system estimates and budgets *risk*; momentum enters only as a capped tilt after optimization, never inside it.

**2. Diversification is real, not nominal.**
K-means on the assets' co-movement profiles — unsupervised, no labels — recovers exactly the intended behavior families: k=2 splits {equity} vs {IEF, SHY, GLD}; k=3 isolates {SPY, VEA, XLE, XLU} · {IEF, SHY} · {GLD alone}. The dendrogram places XLU as the most distant equity member (the defensive) and gold as a genuine third behavior.

## Baseline Model (Notebooks 03–05)

**Model.** The baseline is a **regularized risk model feeding a portfolio optimizer**:

1. **Ledoit-Wolf shrinkage** of the rolling 3-year weekly covariance matrix — conceptually, Ridge regression applied to covariance estimation: noisy pairwise estimates are shrunk toward a structured target, with the shrinkage intensity *self-calibrated from the data* (observed range 0.03–0.19, peaking at 0.186 in March 2020 — a chaotic sample earns less trust and more structure). An EWMA variant runs in parallel as a competing estimator (λ ∈ {0.94, 0.97, 0.99}; the headline λ = 0.97 has a ~5-month half-life).
2. **Two allocation models compete on those matrices**: minimum-variance with weight caps, and Risk Parity (equal risk contributions). Both are computed monthly, point-in-time, producing a full historical series of portfolio weights.

This is the baseline that Module 24 will extend (momentum tilt, regime-based exposure scaling) and evaluate in a full walk-forward backtest with transaction costs.

**Evaluation metric and rationale.** The declared headline metrics are the **Sharpe ratio** (risk-adjusted efficiency, for comparison against SPY and a 60/40 benchmark) and **maximum drawdown / Calmar ratio** (the project's stated objective: controlled drawdowns). Rationale: the research question is explicitly about risk-adjusted performance, so raw return or accuracy-style metrics would not measure success; Sharpe captures return per unit of volatility, while max drawdown captures the tail experience an investor actually lives through — the two together cover "efficiency" and "pain". Because return direction proved unpredictable (EDA finding 1), metrics that reward forecasting skill would be misaligned with a system designed to manage risk.

**Interim model validation (available now, before the backtest):**
- The shrinkage intensity behaves as theory predicts — rising when the sample is noisy (2020) — confirming the regularization is doing its job rather than acting as a fixed fudge factor.
- The weight series responds to real changes in the risk structure: when the SPY–IEF correlation, negative for a decade, flipped positive during 2022 (bonds falling with equities), Risk Parity **cut the IEF weight from a 46% average in 2021 to 38.6% in 2023, drifting further to 33% today**, with no human intervention — the model registered that bonds had stopped providing diversification and reduced reliance on them. The EWMA estimator's SPY–IEF correlation first turned positive in June 2022, exactly 6 months before the rolling-window estimate (December 2022), illustrating the memory-length trade-off the backtest will adjudicate.

The full quantitative comparison (baseline vs extended system vs SPY and 60/40, with costs and statistical significance via block bootstrap) is the work of Module 24.

## Changes Since Module 16

The research question, data sources, and evaluation plan are unchanged. Two methodological refinements, both motivated by the EDA:

1. **Covariance is estimated on weekly (not daily) returns**, after quantifying the close-time-lag bias in daily SPY–VEA correlation.
2. **Any return-forecasting component was ruled out as an optimizer input** by the ACF evidence; momentum is restricted to a capped post-optimization tilt.

## Next Steps (Module 24)

- Regime detector (K-means on market-condition features) scaling total exposure, with freed capital parked in SHY.
- Momentum tilt and assembly of the full monthly pipeline.
- Walk-forward backtest with 5 bps one-way transaction costs vs SPY and 60/40, plus per-crisis event studies (2011, 2018, 2020, 2022).
- Statistical validation: moving-block bootstrap confidence intervals on Sharpe differences — distinguishing real effects from sampling luck. A rigorous negative finding is considered as valuable an outcome as a positive one.
