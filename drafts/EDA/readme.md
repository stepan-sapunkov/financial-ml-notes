# EDA Toolkit for Time Series

## Short Overview

`eda.py` is a notebook-friendly exploratory data analysis toolkit for financial and crypto time series.

It provides tools for:

- target construction and forecast horizon diagnostics;
- data quality checks;
- stationarity, autocorrelation, and rolling moment diagnostics;
- distribution, tail, and volatility clustering diagnostics;
- seasonality, persistence, and structural break diagnostics;
- feature-target hypothesis checks;
- feature-feature redundancy checks;
- trading and risk diagnostics.

The module can be used through the `EDA` class:

```python
from eda import EDA
```

or as a standalone function library:

```python
import eda
```

The design goal is to reduce common time-series mistakes, especially look-ahead bias. Forward returns are aligned at decision time `t`, rolling statistics use trailing windows, and HAC/Newey-West standard errors are used where overlapping forward returns matter.

## Installation and Dependencies

Required dependencies:

```text
pandas
numpy
scipy
statsmodels
scikit-learn
matplotlib
```

Install them with:

```bash
pip install pandas numpy scipy statsmodels scikit-learn matplotlib
```

Optional dependency:

```text
ruptures      # optional, used by structural_breaks for multiple break detection
```

Install it with:

```bash
pip install ruptures
```

If `ruptures` is not installed, `structural_breaks` returns an empty break table with a warning instead of failing silently.

## Import Examples

If `eda.py` is inside `username/EDA/`, add that directory to `sys.path` from a notebook:

```python
import sys
sys.path.append("username/EDA")

from eda import EDA
```

Class API:

```python
eda = EDA(df, time_col="datetime", price_col="close")
```

Standalone API:

```python
import eda

target = eda.forward_return(df["close"], horizon=5)
result = eda.target_selection(df["close"], horizons=[1, 5, 10], plot=False)
```

## Expected Data Format

The module expects a `pd.DataFrame` or `pd.Series`.

A typical crypto dataset looks like this:

| datetime | close | volume | spread | feature_1 |
|---|---:|---:|---:|---:|
| 2024-01-01 00:00:00 | 42100.0 | 12.4 | 0.01 | 0.52 |
| 2024-01-01 00:01:00 | 42120.5 | 10.8 | 0.01 | 0.49 |

Main expectations:

- `df` should be a `pd.DataFrame`.
- Time should be either the index or a column passed as `time_col`.
- `time_col` is converted to `pd.DatetimeIndex` and sorted by time inside `EDA`.
- The price column is usually `"close"`.
- Feature columns should be numeric when used in feature diagnostics.
- A target column can be supplied if the target is already built.
- Missing values and infinite values are usually converted to `NaN` and dropped for tests that require finite samples.
- Rolling outputs preserve the original time index where possible.
- Sorted time order matters because forward returns and rolling windows depend on row order.

Example:

```python
eda = EDA(
    df,
    time_col="datetime",
    price_col="close",
    target_col=None,
)
```

## Important Time-Series Conventions

If a feature is observed at time `t`, it must be compared only with a target that is known after `t`. It must not be compared with future feature values.

For log forward returns, the module uses:

<p align="center">
  <img src="https://latex.codecogs.com/svg.image?r_{t,h}=\log(P_{t+h})-\log(P_t)" />
</p>

For simple forward returns, the module uses:

<p align="center">
  <img src="https://latex.codecogs.com/svg.image?r_{t,h}=\frac{P_{t+h}}{P_t}-1" />
</p>

Where:

- `P_t` is the price at time `t`;
- `h` is the forecast horizon in rows or bars;
- the return is indexed at decision time `t`;
- the last `h` rows are `NaN` because `P_{t+h}` is unavailable;
- rolling statistics are trailing and use current and past observations only.

Overlapping returns matter. A 10-bar forward return overlaps heavily with the next 9 observations. For test diagnostics, use `step=h` when you want a non-overlapping subsample. For feature-target IC tests, `feature_target_report` enforces at least `h - 1` HAC/Newey-West lags when it builds forward targets internally.

## Quick Start

```python
from eda import EDA

eda = EDA(df, time_col="datetime", price_col="close")

target_result = eda.target_selection(
    horizons=[1, 5, 10, 20],
    cost=0.0005,
    plot=True,
)

diag_result = eda.data_diagnostics(
    cols=["close"],
    plot=True,
)

dist_result = eda.distribution_report(
    col="return",
    plot=True,
)

season_result = eda.seasonality_report(
    col="return",
    period=1440,
    plot=True,
)

ft_result = eda.feature_target_report(
    features=["volume", "spread", "feature_1"],
    horizons=[1, 5, 10],
    method="spearman",
    plot=True,
)

fr_result = eda.feature_relation_report(
    features=["volume", "spread", "feature_1"],
    plot=True,
)
```

Report methods return `EDAResult`, a `dict` subclass that displays tables cleanly in notebooks while keeping normal dictionary access:

```python
target_result.keys()
target_result["summary"]
```

## Public API Overview

### Main Class Methods

| Method | Purpose | Main Input | Main Output |
|---|---|---|---|
| `EDA.forward_return` | Build aligned forward returns | price column, horizon | `pd.Series` |
| `EDA.target_selection` | Compare target horizons | price column, horizons, cost | `EDAResult` |
| `EDA.data_diagnostics` | Data quality, stationarity, ACF/PACF, rolling moments | selected columns | `EDAResult` |
| `EDA.distribution_report` | Distribution, tails, normality, ARCH | one column | `EDAResult` |
| `EDA.seasonality_report` | Seasonality, Hurst, CUSUM, breaks, STL | one column | `EDAResult` |
| `EDA.feature_target_report` | Feature-target IC and nonlinear diagnostics | features, target or price | `EDAResult` |
| `EDA.feature_relation_report` | Correlation, VIF, clustering, PCA, t-SNE | feature columns | `EDAResult` |

### Standalone Functions

| Function | Purpose | Output |
|---|---|---|
| `forward_return` | Build aligned forward returns | `pd.Series` |
| `target_selection` | Compare forward-return horizons | `EDAResult` |
| `rolling_target_probability` | Rolling probability that target exceeds cost | `EDAResult` |
| `missing_pct` | Missing and infinite value percentage | `pd.Series` |
| `series_summary` | Summary moments and quantiles | `pd.DataFrame` |
| `rolling_mean`, `rolling_std`, `rolling_median`, `rolling_mode`, `rolling_skewness`, `rolling_kurtosis` | Trailing rolling statistics | `pd.Series` |
| `adf_test`, `kpss_test`, `zivot_andrews_test`, `stationarity_tests`, `stationarity_summary` | Stationarity tests | `pd.DataFrame` |
| `acf_pacf` | ACF/PACF table and plot | `EDAResult` |
| `data_diagnostics` | Aggregate data diagnostics | `EDAResult` |
| `qq_plot`, `density_plot` | Distribution plots and tables | `EDAResult` |
| `hill_estimator`, `evt_gpd_fit` | Tail index and GPD tail fit | `pd.DataFrame`, `EDAResult` |
| `normality_tests`, `arch_lm_test`, `class_balance` | Distribution tests and class counts | `pd.DataFrame` |
| `distribution_report` | Aggregate distribution diagnostics | `EDAResult` |
| `periodogram`, `lomb_scargle_periodogram`, `stl_decomposition`, `hurst_exponent`, `cusum_test`, `structural_breaks`, `seasonality_report` | Seasonality and regime diagnostics | `EDAResult` or `pd.DataFrame` |
| `feature_target_correlation`, `ic_summary`, `cumulative_ic`, `cross_sectional_ic`, `rolling_ic`, `rolling_ic_stats` | IC diagnostics | `pd.DataFrame` or `EDAResult` |
| `feature_quantile_stats`, `granger_causality`, `mutual_information`, `rolling_mutual_information`, `distance_correlation`, `rolling_distance_correlation`, `conditional_ic`, `rolling_conditional_ic` | Feature-target diagnostics | `pd.DataFrame`, `pd.Series`, or `float` |
| `feature_target_report` | Aggregate feature-target report | `EDAResult` |
| `correlation_matrix`, `heatmap_correlation_matrix`, `vif`, `cluster_features`, `pca_analysis`, `tsne_projection`, `feature_relation_report` | Feature-feature diagnostics | `pd.DataFrame` or `EDAResult` |
| `realized_volatility`, `bipower_variation`, `rolling_sharpe`, `drawdown_diagnostics`, `tail_dependence`, `upside_downside_volatility`, `hit_rate`, `turnover_cost_diagnostics`, `ljung_box_tests` | Trading and risk diagnostics | `pd.Series`, `pd.DataFrame`, or `EDAResult` |
| `missingness_by_time_bucket`, `calendar_seasonality` | Calendar diagnostics | `pd.DataFrame` |
| `set_plot_style`, `style_axis` | Plot style helpers | `None`, `Axes` |
| `EDAResult` | Notebook-friendly result container | `dict` subclass |

## Detailed Function Documentation

## `EDA(df, time_col=None, price_col="close", target_col=None, cold_palette=True, copy=True)`

### Purpose

Notebook-friendly facade around the standalone functions.

### Inputs

- `df`: source `pd.DataFrame`.
- `time_col`: optional timestamp column. If provided, it is converted to datetime, set as index, and sorted.
- `price_col`: default price column for forward-return targets.
- `target_col`: optional default target column.
- `cold_palette`: if `True`, applies the module plotting style.
- `copy`: if `True`, keeps a defensive copy of the input frame.

### Main Calculations

The class does not duplicate analytical logic. Methods delegate to standalone functions.

### Returns

An `EDA` object with a cleaned and sorted `df` attribute.

### Notes

Use `EDA` for notebook workflows. Use standalone functions when you want pure functional calls.

### Example

```python
eda = EDA(df, time_col="datetime", price_col="close")
```

## `EDA.forward_return(horizon=1, price_col=None, log_return=True)`

### Purpose

Compute a forward return aligned at decision time `t`.

### Inputs

- `horizon`: positive number of bars ahead.
- `price_col`: optional price column. Defaults to `EDA.price_col`.
- `log_return`: if `True`, use log returns; otherwise use simple returns.

### Main Calculations

<p align="center">
  <img src="https://latex.codecogs.com/svg.image?r_{t,h}=\log(P_{t+h})-\log(P_t)" />
</p>

or:

<p align="center">
  <img src="https://latex.codecogs.com/svg.image?r_{t,h}=\frac{P_{t+h}}{P_t}-1" />
</p>

### Returns

`pd.Series` with the same index as `EDA.df`. The last `horizon` rows are `NaN`.

### Notes

For log returns, prices must be positive. The result is aligned to feature time `t`.

### Example

```python
y = eda.forward_return(horizon=5)
```

## `EDA.target_selection(horizons, price_col=None, cost=0.0005, cost_is_multiplier=False, log_return=True, rolling_window=None, min_periods=None, plot=True)`

### Purpose

Compare candidate forward-return targets across horizons.

### Inputs

- `horizons`: sequence of positive horizons.
- `price_col`: optional price column.
- `cost`: cost threshold in return units.
- `cost_is_multiplier`: convert multiplier-style costs like `1.002` to log units.
- `log_return`: use log or simple returns.
- `rolling_window`: optional trailing rolling window.
- `min_periods`: minimum rolling observations.
- `plot`: include rolling probability figure.

### Main Calculations

For each horizon, the method computes absolute return statistics and cost-aware metrics such as:

<p align="center">
  <img src="https://latex.codecogs.com/svg.image?\Pr(|r_{t,h}|>c)" />
</p>

### Returns

`EDAResult` with keys:

- `summary`;
- `targets`;
- `rolling_probability`;
- `figure`.

### Notes

Forward returns are aligned at time `t`. Rolling windows are trailing. Final unavailable target rows stay `NaN`.

### Example

```python
res = eda.target_selection(horizons=[1, 5, 10], cost=0.0005, plot=False)
res["summary"]
```

## `EDA.data_diagnostics(cols=None, rolling_windows=(10080, 43200, 86400), quantiles=(...), lags=40, step=1, plot=True, verbose=False)`

### Purpose

Run first-pass data quality, rolling, autocorrelation, and stationarity diagnostics.

### Inputs

- `cols`: selected columns. If `None`, all columns are considered.
- `rolling_windows`: trailing window sizes.
- `quantiles`: quantiles for summary table.
- `lags`: ACF/PACF lags.
- `step`: subsampling step for stationarity and ACF/PACF.
- `plot`: include figures.
- `verbose`: warn when column diagnostics fail.

### Main Calculations

Computes missingness, moments, quantiles, ADF/KPSS/Zivot-Andrews tests, ACF/PACF, and rolling moments.

### Returns

`EDAResult` with keys:

- `summary`;
- `stationarity`;
- `acf_pacf`;
- `rolling`;
- `figures`;
- `warnings`.

### Notes

Use `step=h` for overlapping `h`-bar returns when non-overlapping test samples are desired.

### Example

```python
res = eda.data_diagnostics(cols=["return"], lags=24, step=5, plot=False)
res["stationarity"]
```

## `EDA.distribution_report(col, q=0.95, tail="abs", arch_lags=10, step=1, plot=True)`

### Purpose

Analyze distribution shape, tails, normality, and ARCH effects.

### Inputs

- `col`: column name.
- `q`: tail threshold quantile.
- `tail`: `"abs"`, `"right"`, or `"left"`.
- `arch_lags`: lags for Engle ARCH LM test.
- `step`: subsampling step for tests.
- `plot`: include density, QQ, and GPD figures.

### Main Calculations

Combines summary stats, density, QQ plots, Hill tail index, GPD fit, normality tests, ARCH LM test, and optional class balance.

### Returns

`EDAResult` with keys:

- `summary`;
- `density`;
- `qq`;
- `hill`;
- `evt`;
- `normality`;
- `arch_lm`;
- `class_balance`.

### Notes

Use `step=h` for overlapping forward returns.

### Example

```python
res = eda.distribution_report(col="return", q=0.99, step=5, plot=False)
res["normality"]
```

## `EDA.seasonality_report(col, period=None, fs=1.0, plot=True, verbose=False)`

### Purpose

Run seasonality, persistence, and structural-change diagnostics.

### Inputs

- `col`: numeric column.
- `period`: STL seasonal period in rows.
- `fs`: sampling frequency for periodogram.
- `plot`: include figures.
- `verbose`: warn when optional diagnostics fail.

### Main Calculations

Computes periodogram, Lomb-Scargle periodogram, Hurst exponent, CUSUM test, structural breaks, and optional STL decomposition.

### Returns

`EDAResult` with keys:

- `periodogram`;
- `lomb_scargle`;
- `hurst_rs`;
- `cusum`;
- `structural_breaks`;
- `stl`.

### Notes

A `DatetimeIndex` improves period inference and Lomb-Scargle interpretation.

### Example

```python
res = eda.seasonality_report(col="return", period=1440, plot=False)
```

## `EDA.feature_target_report(features, target=None, price_col=None, horizons=(1,), log_return=True, method="spearman", rolling_window=None, min_periods=None, rolling_step=1, quantiles=5, cost=None, hac_lags="auto", plot=True, run_granger=False, granger_maxlag=5, run_nonlinear=True, verbose=False)`

### Purpose

Run feature-vs-target diagnostics over one or several horizons.

### Inputs

- `features`: feature column names observed at time `t`.
- `target`: optional target column or series already aligned at `t`.
- `price_col`: price column used to build forward returns if `target` is not supplied.
- `horizons`: forward-return horizons.
- `method`: `"pearson"`, `"spearman"`, or `"kendall"` for IC summary.
- `rolling_window`: optional trailing rolling IC window.
- `min_periods`: minimum observations in rolling windows.
- `rolling_step`: evaluate expensive rolling summaries every `rolling_step` rows.
- `quantiles`: number of feature buckets.
- `cost`: optional cost threshold for quantile stats.
- `hac_lags`: HAC/Newey-West lags.
- `plot`: include figures.
- `run_granger`: include Granger causality tests.
- `granger_maxlag`: maximum Granger lag.
- `run_nonlinear`: include mutual information and distance correlation.
- `verbose`: print/warn on optional failures.

### Main Calculations

Computes IC, HAC t-statistics, cumulative IC, rolling IC, feature quantile stats, mutual information, distance correlation, and optional Granger causality.

<p align="center">
  <img src="https://latex.codecogs.com/svg.image?IC=\operatorname{corr}(x_t,r_{t,h})" />
</p>

### Returns

`EDAResult` with keys:

- `ic`;
- `cumulative_ic`;
- `rolling_ic`;
- `rolling_ic_stats`;
- `quantiles`;
- `mutual_information`;
- `distance_correlation`;
- `granger`;
- `figures`;
- `warnings`.

### Notes

If the method builds forward returns internally, feature rows at `t` are aligned to returns from `t` to `t+h`. HAC lags are at least `h - 1` for overlapping returns.

### Example

```python
res = eda.feature_target_report(
    features=["volume", "spread"],
    horizons=[1, 5, 10],
    method="spearman",
    rolling_window=500,
    cost=0.0005,
    plot=False,
)
res["ic"]
```

## `EDA.feature_relation_report(features, method="pearson", scale=True, max_rows=10000, plot=True, run_tsne=True)`

### Purpose

Analyze relationships between features.

### Inputs

- `features`: feature column names.
- `method`: correlation method.
- `scale`: standardize features for PCA and t-SNE.
- `max_rows`: optional row subsampling.
- `plot`: include heatmap, dendrogram, and PCA figure.
- `run_tsne`: include feature-level t-SNE projection.

### Main Calculations

Computes correlation matrix, VIF, hierarchical feature clustering, PCA, and optional t-SNE.

### Returns

`EDAResult` with keys:

- `correlation`;
- `vif`;
- `clustering`;
- `pca`;
- `tsne`;
- optional `tsne_warning`.

### Notes

Rows with missing or infinite values are dropped for matrix diagnostics. Scaling is recommended for PCA and t-SNE.

### Example

```python
res = eda.feature_relation_report(["volume", "spread", "feature_1"], plot=False)
res["vif"]
```

## `EDAResult(data=None, title="EDA result", max_rows=20)`

### Purpose

Dictionary result container with clean notebook HTML display.

### Inputs

- `data`: optional dictionary.
- `title`: display title.
- `max_rows`: number of rows shown in HTML preview.

### Main Calculations

No statistical calculation. It stores result tables, figures, warnings, and nested result objects.

### Returns

An `EDAResult`, which behaves like a normal `dict`.

### Notes

Use normal dictionary access for full tables.

### Example

```python
res = eda.target_selection([1, 5], plot=False)
res["summary"]
```

## `forward_return(price, horizon=1, log_return=True, name=None)`

### Purpose

Standalone version of forward-return construction.

### Inputs

- `price`: price `pd.Series`.
- `horizon`: positive forecast horizon.
- `log_return`: log or simple return.
- `name`: optional output name.

### Main Calculations

Uses `price.shift(-horizon)` to align the future return at current time `t`.

### Returns

`pd.Series` with the same index as `price`.

### Notes

The last `horizon` rows are `NaN`. Log returns require positive prices.

### Example

```python
y = forward_return(df["close"], horizon=5)
```

## `target_selection(close, horizons, cost=0.0005, cost_is_multiplier=False, log_return=True, rolling_window=None, min_periods=None, plot=True)`

### Purpose

Compare forward-return targets across horizons.

### Inputs

- `close`: price series.
- `horizons`: positive horizons.
- `cost`: threshold in return units.
- `cost_is_multiplier`: convert multiplier-style cost to log threshold.
- `log_return`: log or simple returns.
- `rolling_window`: optional trailing window.
- `min_periods`: minimum rolling observations.
- `plot`: include rolling probability plot.

### Main Calculations

Computes mean absolute return, median absolute return, cost exceedance probability, mean excess, skew, kurtosis, and optional rolling standard deviation.

### Returns

`EDAResult` with `summary`, `targets`, `rolling_probability`, and `figure`.

### Notes

Targets are aligned at time `t`; unavailable final rows are `NaN`.

### Example

```python
res = target_selection(df["close"], [1, 5, 10], cost=0.0005, plot=False)
```

## `rolling_target_probability(close, horizons, cost=0.0005, cost_is_multiplier=False, window=10080, min_periods=None, log_return=True, plot=True)`

### Purpose

Compute trailing probability that absolute forward returns exceed cost.

### Inputs

- `close`: price series.
- `horizons`: positive horizons.
- `cost`: threshold in return units.
- `window`: trailing window length.
- `min_periods`: minimum observations.
- `log_return`: log or simple returns.
- `plot`: include figure.

### Main Calculations

<p align="center">
  <img src="https://latex.codecogs.com/svg.image?\Pr(|r_{t,h}|>c)" />
</p>

### Returns

Same `EDAResult` structure as `target_selection`.

### Notes

The rolling calculation is trailing.

### Example

```python
res = rolling_target_probability(df["close"], [1, 5], window=500, plot=False)
```

## `missing_pct(data, cols=None)`

### Purpose

Compute missing and infinite value percentage.

### Inputs

- `data`: `pd.Series` or `pd.DataFrame`.
- `cols`: optional selected columns.

### Main Calculations

Converts `inf` and `-inf` to `NaN`, then computes missing percentage.

### Returns

`pd.Series` with percentages in `[0, 100]`.

### Notes

Useful as the first data quality check.

### Example

```python
missing_pct(df, cols=["close", "volume"])
```

## `series_summary(data, cols=None, quantiles=(...))`

### Purpose

Summarize basic distribution and data quality properties.

### Inputs

- `data`: `pd.Series` or `pd.DataFrame`.
- `cols`: optional selected columns.
- `quantiles`: quantiles to include.

### Main Calculations

Computes count, missing percentage, mean, standard deviation, median, mode, skew, kurtosis, min, max, and quantiles.

### Returns

`pd.DataFrame`, one row per column.

### Notes

Non-numeric values are coerced to numeric where possible.

### Example

```python
summary = series_summary(df, cols=["return"])
```

## `rolling_mean(series, window, min_periods=None)`

### Purpose

Compute trailing rolling mean.

### Inputs

- `series`: numeric `pd.Series`.
- `window`: trailing window length.
- `min_periods`: minimum observations.

### Main Calculations

Uses pandas trailing `rolling(...).mean()`.

### Returns

`pd.Series` aligned to the input index.

### Notes

No future values are used.

### Example

```python
rm = rolling_mean(df["return"], window=100)
```

## `rolling_std(series, window, min_periods=None)`

### Purpose

Compute trailing rolling standard deviation.

### Inputs

- `series`: numeric `pd.Series`.
- `window`: trailing window length.
- `min_periods`: minimum observations.

### Main Calculations

Uses pandas trailing `rolling(...).std()`.

### Returns

`pd.Series`.

### Notes

Index is preserved.

### Example

```python
rs = rolling_std(df["return"], window=100)
```

## `rolling_median(series, window, min_periods=None)`

### Purpose

Compute trailing rolling median.

### Inputs

- `series`: numeric `pd.Series`.
- `window`: trailing window length.
- `min_periods`: minimum observations.

### Main Calculations

Uses pandas trailing `rolling(...).median()`.

### Returns

`pd.Series`.

### Notes

Useful for robust local level diagnostics.

### Example

```python
med = rolling_median(df["return"], window=100)
```

## `rolling_mode(series, window, min_periods=None)`

### Purpose

Compute trailing rolling mode.

### Inputs

- `series`: numeric or discrete `pd.Series`.
- `window`: trailing window length.
- `min_periods`: minimum observations.

### Main Calculations

Computes the most frequent value inside each trailing window.

### Returns

`pd.Series`.

### Notes

If several values share the highest frequency, pandas' first sorted mode is used.

### Example

```python
mode = rolling_mode(signal_class, window=50)
```

## `rolling_skewness(series, window, min_periods=None)`

### Purpose

Compute trailing rolling skewness.

### Inputs

- `series`: numeric `pd.Series`.
- `window`: trailing window length.
- `min_periods`: minimum observations.

### Main Calculations

Uses pandas trailing `rolling(...).skew()`.

### Returns

`pd.Series`.

### Notes

Use for local asymmetry diagnostics.

### Example

```python
sk = rolling_skewness(df["return"], window=500)
```

## `rolling_kurtosis(series, window, min_periods=None)`

### Purpose

Compute trailing rolling excess kurtosis.

### Inputs

- `series`: numeric `pd.Series`.
- `window`: trailing window length.
- `min_periods`: minimum observations.

### Main Calculations

Uses pandas trailing `rolling(...).kurt()`.

### Returns

`pd.Series`.

### Notes

Use for local tail-shape diagnostics.

### Example

```python
rk = rolling_kurtosis(df["return"], window=500)
```

## `adf_test(series, maxlag=10, regression="c")`

### Purpose

Run the Augmented Dickey-Fuller unit-root test.

### Inputs

- `series`: numeric `pd.Series`.
- `maxlag`: maximum ADF lag. `None` enables AIC autolag.
- `regression`: deterministic terms passed to statsmodels.

### Main Calculations

Tests whether the series has a unit root.

### Returns

`pd.DataFrame` with statistic, p-value, lags, observations, and warning.

### Notes

Too-short or constant samples return warning rows.

### Example

```python
adf_test(df["return"])
```

## `kpss_test(series, nlags=10, regression="c")`

### Purpose

Run the KPSS stationarity test.

### Inputs

- `series`: numeric `pd.Series`.
- `nlags`: KPSS lags.
- `regression`: `"c"` for level stationarity or `"ct"` for trend stationarity.

### Main Calculations

Tests whether the series is stationary around a level or trend.

### Returns

`pd.DataFrame`.

### Notes

KPSS has the opposite null hypothesis from ADF.

### Example

```python
kpss_test(df["return"], regression="c")
```

## `zivot_andrews_test(series, maxlag=10, regression="c")`

### Purpose

Run the Zivot-Andrews unit-root test with one endogenous structural break.

### Inputs

- `series`: numeric `pd.Series`.
- `maxlag`: maximum lag.
- `regression`: deterministic terms.

### Main Calculations

Tests for a unit root while allowing one structural break under the alternative.

### Returns

`pd.DataFrame` with statistic, p-value, lag, break index/time, critical value, and warning.

### Notes

Requires a longer sample than ADF/KPSS.

### Example

```python
zivot_andrews_test(df["close"])
```

## `stationarity_tests(series, step=1, max_test_size=100000, adf_lag=10, kpss_lag=10, za_lag=10)`

### Purpose

Run ADF, KPSS, and Zivot-Andrews together.

### Inputs

- `series`: numeric `pd.Series`.
- `step`: use every `step`-th observation after cleaning.
- `max_test_size`: keep the last `max_test_size` rows.
- `adf_lag`: ADF lag.
- `kpss_lag`: KPSS lag.
- `za_lag`: Zivot-Andrews lag.

### Main Calculations

Runs all three stationarity tests on a cleaned and optionally stepped sample.

### Returns

`pd.DataFrame` with one row per test.

### Notes

Use `step=h` for overlapping `h`-bar returns.

### Example

```python
stationarity_tests(df["fwd_return_5"], step=5)
```

## `stationarity_summary(data, cols=None, **kwargs)`

### Purpose

Run `stationarity_tests` for multiple columns.

### Inputs

- `data`: `pd.Series` or `pd.DataFrame`.
- `cols`: optional selected columns.
- `**kwargs`: passed to `stationarity_tests`.

### Main Calculations

Loops over selected columns and combines stationarity test results.

### Returns

`pd.DataFrame` with a leading `column` field.

### Notes

Pass `step=h` through `kwargs` for overlapping returns.

### Example

```python
stationarity_summary(df, cols=["return", "fwd_return_5"], step=5)
```

## `acf_pacf(series, lags=40, alpha=0.05, step=1, plot=True)`

### Purpose

Compute ACF and PACF values.

### Inputs

- `series`: numeric `pd.Series`.
- `lags`: number of lags.
- `alpha`: confidence interval level. If `None`, CI columns are omitted.
- `step`: subsampling step.
- `plot`: include bar plots.

### Main Calculations

Computes autocorrelation and partial autocorrelation on `series.dropna()[::step]`.

### Returns

`EDAResult` with:

- `table`;
- `figure`.

### Notes

With `step=5`, lag 1 means 5 original bars. Use `step=h` for overlapping `h`-bar returns.

### Example

```python
res = acf_pacf(df["fwd_return_5"], lags=30, step=5, plot=False)
```

## `data_diagnostics(data, cols=None, rolling_windows=(10080, 43200, 86400), quantiles=(...), lags=40, step=1, plot=True, verbose=False)`

### Purpose

Standalone aggregate data diagnostics.

### Inputs

Same as `EDA.data_diagnostics`.

### Main Calculations

Combines `series_summary`, `stationarity_summary`, `acf_pacf`, and rolling moments.

### Returns

`EDAResult` with `summary`, `stationarity`, `acf_pacf`, `rolling`, `figures`, and `warnings`.

### Notes

Rolling outputs preserve the original index.

### Example

```python
res = data_diagnostics(df, cols=["return"], rolling_windows=[500], step=5, plot=False)
```

## `qq_plot(series, dist="both", max_points=50000, plot=True)`

### Purpose

Create QQ diagnostics against normal and/or Student-t distributions.

### Inputs

- `series`: numeric `pd.Series`.
- `dist`: `"normal"`, `"student_t"`, or `"both"`.
- `max_points`: maximum observations used.
- `plot`: include QQ plot.

### Main Calculations

Fits normal or Student-t parameters and creates QQ data/plots.

### Returns

`EDAResult` with:

- `fit`;
- `figure`.

### Notes

Large samples are trimmed to the last `max_points`.

### Example

```python
qq = qq_plot(df["return"], dist="both", plot=False)
```

## `density_plot(series, normal_overlay=True, points=600, plot=True)`

### Purpose

Estimate empirical density.

### Inputs

- `series`: numeric `pd.Series`.
- `normal_overlay`: add fitted normal density.
- `points`: grid size.
- `plot`: include figure.

### Main Calculations

Uses Gaussian KDE on a grid between the 0.1% and 99.9% quantiles.

### Returns

`EDAResult` with:

- `density`;
- `warning`;
- `figure`.

### Notes

KDE can fail on degenerate samples; the warning key records the error.

### Example

```python
density = density_plot(df["return"], plot=False)
```

## `hill_estimator(series, q=0.95, tail="abs")`

### Purpose

Estimate power-law tail index alpha using the Hill estimator.

### Inputs

- `series`: numeric `pd.Series`.
- `q`: threshold quantile.
- `tail`: `"abs"`, `"right"`, or `"left"`.

### Main Calculations

<p align="center">
  <img src="https://latex.codecogs.com/svg.image?\hat{\alpha}^{-1}=\frac{1}{k}\sum_{i=1}^{k}\log\left(\frac{X_{(i)}}{X_{(k+1)}}\right)" />
</p>

### Returns

`pd.DataFrame` with `tail`, `n`, `q`, `threshold`, `tail_n`, and `alpha`.

### Notes

Smaller alpha means heavier tails. The estimate is sensitive to `q`.

### Example

```python
hill_estimator(df["return"], q=0.99, tail="abs")
```

## `evt_gpd_fit(series, q=0.95, tail="abs", plot=False)`

### Purpose

Fit a Generalized Pareto Distribution to threshold exceedances.

### Inputs

- `series`: numeric `pd.Series`.
- `q`: threshold quantile.
- `tail`: `"abs"`, `"right"`, or `"left"`.
- `plot`: include histogram and GPD PDF overlay.

### Main Calculations

Fits GPD parameters `xi` and `beta` to exceedances above the threshold.

### Returns

`EDAResult` with:

- `fit`;
- `excess`;
- `figure`.

### Notes

If too few exceedances exist, parameters are `NaN` and `warning` explains why.

### Example

```python
evt = evt_gpd_fit(df["return"], q=0.99, tail="left", plot=False)
```

## `normality_tests(series, step=1, max_test_size=100000, shapiro_size=5000, random_state=42)`

### Purpose

Run normality diagnostics.

### Inputs

- `series`: numeric `pd.Series`.
- `step`: subsampling step.
- `max_test_size`: maximum observations after stepping.
- `shapiro_size`: maximum Shapiro-Wilk sample size.
- `random_state`: reproducible subsampling seed.

### Main Calculations

Runs Jarque-Bera, Shapiro-Wilk, Anderson-Darling, and Kolmogorov-Smirnov diagnostics.

### Returns

`pd.DataFrame` with `test`, `n_test`, statistic, p-value, and warnings.

### Notes

Shapiro-Wilk is subsampled for large samples. KS uses estimated mean/std and is a practical diagnostic, not a Lilliefors-corrected test.

### Example

```python
normality_tests(df["fwd_return_10"], step=10)
```

## `arch_lm_test(series, nlags=10, step=1, max_test_size=100000)`

### Purpose

Run Engle ARCH LM test for volatility clustering.

### Inputs

- `series`: return-like `pd.Series`.
- `nlags`: number of ARCH lags.
- `step`: subsampling step.
- `max_test_size`: maximum observations after stepping.

### Main Calculations

Tests whether squared residuals have ARCH effects.

### Returns

`pd.DataFrame` with LM statistic, LM p-value, F statistic, F p-value, sample size, lags, and warning.

### Notes

Use `step=h` for overlapping forward returns.

### Example

```python
arch_lm_test(df["fwd_return_5"], nlags=20, step=5)
```

## `class_balance(target, normalize=True, max_classes=50)`

### Purpose

Count target classes.

### Inputs

- `target`: categorical or discrete `pd.Series`.
- `normalize`: include frequency column.
- `max_classes`: warn if unique class count is high.

### Main Calculations

Computes value counts and optional relative frequencies.

### Returns

`pd.DataFrame` with `class`, `count`, and optionally `frequency`.

### Notes

Useful for directional or bucketed targets.

### Example

```python
class_balance(np.sign(df["return"]))
```

## `distribution_report(series, q=0.95, tail="abs", arch_lags=10, step=1, plot=True)`

### Purpose

Aggregate distribution, tail, normality, and ARCH diagnostics.

### Inputs

Same as `EDA.distribution_report`, but pass a series directly.

### Main Calculations

Calls `series_summary`, `density_plot`, `qq_plot`, `hill_estimator`, `evt_gpd_fit`, `normality_tests`, `arch_lm_test`, and possibly `class_balance`.

### Returns

`EDAResult` with `summary`, `density`, `qq`, `hill`, `evt`, `normality`, `arch_lm`, and `class_balance`.

### Notes

Use `step=h` for overlapping returns.

### Example

```python
res = distribution_report(df["return"], q=0.99, plot=False)
```

## `periodogram(series, fs=1.0, detrend="constant", plot=True)`

### Purpose

Estimate spectral density for regularly sampled data.

### Inputs

- `series`: numeric `pd.Series`.
- `fs`: sampling frequency.
- `detrend`: scipy periodogram detrend mode.
- `plot`: include figure.

### Main Calculations

Uses `scipy.signal.periodogram`.

### Returns

`EDAResult` with `table` and `figure`.

### Notes

For one-minute bars and frequencies in cycles per day, use `fs=1440`.

### Example

```python
pg = periodogram(df["return"], fs=1440, plot=False)
```

## `lomb_scargle_periodogram(series, min_frequency=None, max_frequency=None, n_freq=1000, plot=True)`

### Purpose

Compute Lomb-Scargle periodogram for irregular timestamps or missing observations.

### Inputs

- `series`: numeric `pd.Series`.
- `min_frequency`: optional lower frequency.
- `max_frequency`: optional upper frequency.
- `n_freq`: number of frequency grid points.
- `plot`: include figure.

### Main Calculations

Uses timestamp distances for `DatetimeIndex`, otherwise row numbers.

### Returns

`EDAResult` with `table` and `figure`.

### Notes

Frequencies are cycles per second for `DatetimeIndex` and cycles per row otherwise.

### Example

```python
ls = lomb_scargle_periodogram(irregular_returns, plot=False)
```

## `stl_decomposition(series, period=None, robust=True, plot=True)`

### Purpose

Decompose a series into observed, trend, seasonal, and residual components.

### Inputs

- `series`: numeric `pd.Series`.
- `period`: seasonal period in rows.
- `robust`: use robust STL.
- `plot`: include component plots.

### Main Calculations

Uses `statsmodels.tsa.seasonal.STL`.

### Returns

`EDAResult` with:

- `components`;
- `period`;
- `figure`.

### Notes

If `period` is not supplied, the function tries to infer a daily period from `DatetimeIndex`.

### Example

```python
stl = stl_decomposition(df["return"], period=1440, plot=False)
```

## `hurst_exponent(series, method="rs", min_window=16, max_window=None, n_windows=20)`

### Purpose

Estimate the Hurst exponent.

### Inputs

- `series`: numeric `pd.Series`.
- `method`: `"rs"` or `"dfa"`.
- `min_window`: minimum window.
- `max_window`: maximum window.
- `n_windows`: number of window sizes.

### Main Calculations

Fits log-log scaling of R/S or DFA statistics.

### Returns

`pd.DataFrame` with method, Hurst exponent, intercept, and number of windows.

### Notes

`H > 0.5` suggests persistence. `H < 0.5` suggests anti-persistence.

### Example

```python
hurst_exponent(df["return"], method="rs")
```

## `cusum_test(series)`

### Purpose

Run CUSUM test for instability around the sample mean.

### Inputs

- `series`: numeric `pd.Series`.

### Main Calculations

Uses `statsmodels.stats.diagnostic.breaks_cusumolsresid`.

### Returns

`pd.DataFrame` with statistic, p-value, critical values, and warning.

### Notes

This is a structural instability diagnostic, not a full regime model.

### Example

```python
cusum_test(df["return"])
```

## `structural_breaks(series, n_bkps=None, penalty=None, model="rbf", min_size=20, max_points=20000, verbose=False)`

### Purpose

Detect multiple structural breaks.

### Inputs

- `series`: numeric `pd.Series`.
- `n_bkps`: fixed number of breaks for binary segmentation.
- `penalty`: PELT penalty.
- `model`: ruptures model, default `"rbf"`.
- `min_size`: minimum segment size.
- `max_points`: downsample long series.
- `verbose`: warn when detection fails.

### Main Calculations

Uses `ruptures` if available. If not, returns a warning.

### Returns

`EDAResult` with:

- `breaks`;
- `sample_step`;
- `warning`.

### Notes

This is a practical Bai-Perron-style alternative, not a strict econometric Bai-Perron implementation.

### Example

```python
breaks = structural_breaks(df["return"], penalty=None)
```

## `seasonality_report(series, period=None, fs=1.0, plot=True, verbose=False)`

### Purpose

Aggregate seasonality and regime diagnostics.

### Inputs

- `series`: numeric `pd.Series`.
- `period`: STL period.
- `fs`: sampling frequency for periodogram.
- `plot`: include figures.
- `verbose`: warn on optional failures.

### Main Calculations

Calls `periodogram`, `lomb_scargle_periodogram`, `hurst_exponent`, `cusum_test`, `structural_breaks`, and `stl_decomposition`.

### Returns

`EDAResult` with `periodogram`, `lomb_scargle`, `hurst_rs`, `cusum`, `structural_breaks`, and `stl`.

### Notes

STL may return a warning object if period inference fails.

### Example

```python
res = seasonality_report(df["return"], period=1440, plot=False)
```

## `feature_target_correlation(features, target, method="spearman")`

### Purpose

Compute correlation between each feature and an aligned target.

### Inputs

- `features`: `pd.Series` or `pd.DataFrame`.
- `target`: aligned target `pd.Series`.
- `method`: `"pearson"`, `"spearman"`, or `"kendall"`.

### Main Calculations

Aligns by index, drops invalid rows, and computes correlation per feature.

### Returns

`pd.DataFrame` with `feature`, `correlation`, `method`, and `n`.

### Notes

The function does not shift inputs. Build forward targets first.

### Example

```python
y = forward_return(df["close"], 5)
feature_target_correlation(df[["volume", "spread"]], y)
```

## `ic_summary(features, target, method="spearman", hac_lags="auto", min_hac_lags=0)`

### Purpose

Compute IC summary with HAC/Newey-West t-statistics.

### Inputs

- `features`: feature series or DataFrame.
- `target`: aligned target.
- `method`: `"pearson"`, `"spearman"`, or `"kendall"`.
- `hac_lags`: Newey-West lags or `"auto"`.
- `min_hac_lags`: lower bound for HAC lags.

### Main Calculations

Computes IC and a HAC t-statistic from standardized IC contributions.

<p align="center">
  <img src="https://latex.codecogs.com/svg.image?t=\frac{\bar{IC}}{SE_{HAC}(\bar{IC})}" />
</p>

### Returns

`pd.DataFrame` with `feature`, `n`, `ic`, `std_ic`, `hac_se`, `t_stat_hac`, `pvalue_hac`, `hac_lags`, and `method`.

### Notes

For overlapping `h`-bar returns, set `min_hac_lags=h-1`.

### Example

```python
y = forward_return(df["close"], 10)
ic_summary(df[["volume", "spread"]], y, min_hac_lags=9)
```

## `cumulative_ic(features, target, method="spearman", cumulative="sum", plot=True)`

### Purpose

Build cumulative IC contribution diagnostics.

### Inputs

- `features`: feature series or DataFrame.
- `target`: aligned target.
- `method`: `"pearson"` or `"spearman"`.
- `cumulative`: `"sum"` or `"mean"`.
- `plot`: include figure.

### Main Calculations

Uses standardized feature-target products and then cumulative sum or expanding mean.

### Returns

`EDAResult` with `table` and `figure`.

### Notes

Spearman mode rank-transforms features and target before standardization.

### Example

```python
res = cumulative_ic(df[["x1", "x2"]], y, plot=False)
```

## `cross_sectional_ic(features, target, time_level=0, method="spearman")`

### Purpose

Compute cross-sectional IC for MultiIndex data.

### Inputs

- `features`: `pd.DataFrame` with MultiIndex, such as `(timestamp, symbol)`.
- `target`: aligned `pd.Series` with same MultiIndex.
- `time_level`: index level used for time grouping.
- `method`: correlation method.

### Main Calculations

Computes one IC per timestamp and feature, then summarizes across timestamps.

### Returns

`EDAResult` with:

- `ic_by_time`;
- `summary`.

### Notes

Requires MultiIndex inputs.

### Example

```python
res = cross_sectional_ic(features_panel, target_panel, time_level="datetime")
```

## `rolling_ic(features, target, window, min_periods=None, method="spearman")`

### Purpose

Compute trailing rolling IC per feature.

### Inputs

- `features`: feature series or DataFrame.
- `target`: aligned target.
- `window`: trailing window.
- `min_periods`: minimum observations.
- `method`: `"pearson"` or `"spearman"`.

### Main Calculations

Computes rolling correlation. Spearman rank-transforms globally before rolling correlation.

### Returns

`pd.DataFrame` indexed by time.

### Notes

For Kendall or rolling HAC summaries, use `rolling_ic_stats`.

### Example

```python
ric = rolling_ic(df[["x1", "x2"]], y, window=500)
```

## `rolling_ic_stats(features, target, window, min_periods=None, method="spearman", hac_lags="auto", min_hac_lags=0, step=1)`

### Purpose

Compute trailing rolling IC, IC dispersion, and HAC t-statistics.

### Inputs

- `features`: feature series or DataFrame.
- `target`: aligned target.
- `window`: trailing window.
- `min_periods`: minimum observations.
- `method`: IC method.
- `hac_lags`: HAC lag setting.
- `min_hac_lags`: minimum HAC lags.
- `step`: evaluate every `step` rows.

### Main Calculations

Calls `ic_summary` inside each trailing window.

### Returns

Long `pd.DataFrame` with timestamp, feature, IC, std, HAC SE, t-stat, p-value, and lags.

### Notes

Use `min_hac_lags=h-1` for overlapping forward returns.

### Example

```python
stats = rolling_ic_stats(df[["x1", "x2"]], y, window=1000, min_hac_lags=9, step=50)
```

## `feature_quantile_stats(features, target, quantiles=5, cost=None)`

### Purpose

Analyze target behavior by feature quantile bucket.

### Inputs

- `features`: feature series or DataFrame.
- `target`: aligned target.
- `quantiles`: number of buckets.
- `cost`: optional return threshold.

### Main Calculations

Uses `pd.qcut` per feature and computes target stats inside each bucket.

### Returns

`pd.DataFrame` with feature, bucket, count, mean, median, std, hit rate, and optional cost-aware columns.

### Notes

Useful for monotonicity checks.

### Example

```python
qstats = feature_quantile_stats(df[["x1", "x2"]], y, quantiles=5, cost=0.0005)
```

## `granger_causality(feature, target, maxlag=5, test="ssr_ftest", verbose=False)`

### Purpose

Test whether lagged feature values help predict target.

### Inputs

- `feature`: feature `pd.Series`.
- `target`: aligned target `pd.Series`.
- `maxlag`: maximum lag.
- `test`: statsmodels test name.
- `verbose`: statsmodels verbosity.

### Main Calculations

Runs statsmodels Granger causality tests with input order `[target, feature]`.

### Returns

`pd.DataFrame` with lag, statistic, p-value, and test name.

### Notes

Granger causality is predictive causality, not economic causality.

### Example

```python
granger_causality(df["volume"], y, maxlag=5)
```

## `mutual_information(features, target, discrete_target=None, n_neighbors=3, random_state=42)`

### Purpose

Estimate nonlinear dependence between features and target.

### Inputs

- `features`: feature series or DataFrame.
- `target`: aligned target.
- `discrete_target`: if `None`, inferred from low-cardinality integer-like targets.
- `n_neighbors`: sklearn MI neighbor count.
- `random_state`: seed.

### Main Calculations

Uses `mutual_info_regression` or `mutual_info_classif`.

### Returns

`pd.DataFrame` with `feature`, `mutual_information`, and `discrete_target`.

### Notes

MI values are not always directly comparable across very different target scalings.

### Example

```python
mi = mutual_information(df[["x1", "x2"]], y)
```

## `rolling_mutual_information(features, target, window, min_periods=None, step=1, n_neighbors=3, random_state=42)`

### Purpose

Compute trailing rolling mutual information.

### Inputs

- `features`: feature series or DataFrame.
- `target`: aligned target.
- `window`: trailing window.
- `min_periods`: minimum observations.
- `step`: evaluation step.
- `n_neighbors`: MI neighbor count.
- `random_state`: seed.

### Main Calculations

Calls `mutual_information` inside trailing windows.

### Returns

Long `pd.DataFrame` with timestamp, feature, and mutual information.

### Notes

Can be expensive on long high-frequency data.

### Example

```python
rmi = rolling_mutual_information(df[["x1", "x2"]], y, window=1000, step=100)
```

## `distance_correlation(x, y, max_n=3000, random_state=42)`

### Purpose

Measure nonlinear dependence between two aligned series.

### Inputs

- `x`: first series.
- `y`: second series.
- `max_n`: maximum rows for exact O(n²) estimator.
- `random_state`: seed for subsampling.

### Main Calculations

Computes distance covariance and distance variances.

### Returns

`float` in `[0, 1]` when defined, otherwise `NaN`.

### Notes

Subsamples with warning when data is larger than `max_n`.

### Example

```python
dc = distance_correlation(df["x1"], y)
```

## `rolling_distance_correlation(feature, target, window, min_periods=None, step=1, max_n=1000)`

### Purpose

Compute trailing rolling distance correlation.

### Inputs

- `feature`: feature series.
- `target`: aligned target.
- `window`: trailing window.
- `min_periods`: minimum observations.
- `step`: evaluation step.
- `max_n`: maximum rows inside each estimator call.

### Main Calculations

Calls `distance_correlation` inside trailing windows.

### Returns

`pd.Series` indexed by window end timestamp.

### Notes

Can be computationally expensive.

### Example

```python
rdc = rolling_distance_correlation(df["x1"], y, window=500, step=50)
```

## `conditional_ic(features, target, condition_feature, quantile=0.5, side="above", method="spearman", hac_lags="auto")`

### Purpose

Compute IC under a condition defined by another feature.

### Inputs

- `features`: candidate features.
- `target`: aligned target.
- `condition_feature`: regime-defining feature.
- `quantile`: threshold quantile.
- `side`: `"above"` or `"below"`.
- `method`: IC method.
- `hac_lags`: HAC lags.

### Main Calculations

Filters rows where the condition feature is above or below its quantile, then calls `ic_summary`.

### Returns

`pd.DataFrame` with IC summary and condition metadata.

### Notes

The threshold is estimated on the aligned sample.

### Example

```python
conditional_ic(df[["x1", "x2"]], y, df["volume"], quantile=0.8, side="above")
```

## `rolling_conditional_ic(features, target, condition_feature, window, min_periods=None, quantile=0.5, side="above", method="spearman", step=1)`

### Purpose

Compute trailing rolling conditional IC.

### Inputs

- `features`: candidate features.
- `target`: aligned target.
- `condition_feature`: regime feature.
- `window`: trailing window.
- `min_periods`: minimum observations.
- `quantile`: rolling threshold quantile.
- `side`: `"above"` or `"below"`.
- `method`: IC method.
- `step`: evaluation step.

### Main Calculations

Computes condition threshold inside each trailing window and then calls `ic_summary`.

### Returns

Long `pd.DataFrame` with timestamp and IC summary columns.

### Notes

Rows with fewer than 5 conditional observations are skipped.

### Example

```python
rolling_conditional_ic(df[["x1"]], y, df["volume"], window=1000, step=100)
```

## `feature_target_report(data, features, target=None, price=None, horizons=(1,), log_return=True, method="spearman", rolling_window=None, min_periods=None, rolling_step=1, quantiles=5, cost=None, hac_lags="auto", plot=True, run_granger=False, granger_maxlag=5, run_nonlinear=True, verbose=False)`

### Purpose

Standalone aggregate feature-target report.

### Inputs

- `data`: source DataFrame.
- `features`: feature column names.
- `target`: optional target column or series.
- `price`: price column or series used to build forward targets.
- Other parameters match `EDA.feature_target_report`.

### Main Calculations

Builds or accepts aligned targets and runs IC, rolling IC, quantile, nonlinear, and optional Granger diagnostics.

### Returns

`EDAResult` with `ic`, `cumulative_ic`, `rolling_ic`, `rolling_ic_stats`, `quantiles`, `mutual_information`, `distance_correlation`, `granger`, `figures`, and `warnings`.

### Notes

If `target` is omitted, `price` is required.

### Example

```python
res = feature_target_report(
    df,
    features=["volume", "spread"],
    price="close",
    horizons=[1, 5],
    plot=False,
)
```

## `correlation_matrix(data, features=None, method="pearson")`

### Purpose

Compute feature correlation matrix.

### Inputs

- `data`: feature DataFrame.
- `features`: optional selected columns.
- `method`: `"pearson"`, `"spearman"`, or `"kendall"`.

### Main Calculations

Drops invalid rows and computes pandas correlation.

### Returns

Square `pd.DataFrame`.

### Notes

Correlation is linear for Pearson and rank-based for Spearman/Kendall.

### Example

```python
corr = correlation_matrix(df, ["x1", "x2", "x3"])
```

## `heatmap_correlation_matrix(data, features=None, method="pearson", plot=True)`

### Purpose

Return correlation matrix with optional heatmap.

### Inputs

Same as `correlation_matrix`, plus `plot`.

### Main Calculations

Computes correlation and optionally renders a blue heatmap.

### Returns

`EDAResult` with `correlation` and `figure`.

### Notes

Use `plot=False` for pure table output.

### Example

```python
res = heatmap_correlation_matrix(df, ["x1", "x2"], plot=False)
```

## `vif(data, features=None)`

### Purpose

Compute variance inflation factor by feature.

### Inputs

- `data`: feature DataFrame.
- `features`: optional selected columns.

### Main Calculations

Standardizes finite rows and computes VIF for each feature.

<p align="center">
  <img src="https://latex.codecogs.com/svg.image?VIF_j=\frac{1}{1-R_j^2}" />
</p>

### Returns

`pd.DataFrame` with `feature`, `vif`, and `vif_gt_10`.

The share of features with VIF above 10 is stored in:

```python
result.attrs["share_vif_gt_10"]
```

### Notes

Constant columns are dropped. Infinite VIF means exact or near-exact collinearity.

### Example

```python
v = vif(df, ["x1", "x2", "x3"])
```

## `cluster_features(data, features=None, method="pearson", linkage_method="average", use_abs=True, plot=True)`

### Purpose

Cluster features by correlation distance.

### Inputs

- `data`: feature DataFrame.
- `features`: optional selected columns.
- `method`: correlation method.
- `linkage_method`: scipy hierarchical linkage method.
- `use_abs`: use `1 - abs(correlation)`.
- `plot`: include dendrogram.

### Main Calculations

Builds a distance matrix from correlations and applies hierarchical clustering.

### Returns

`EDAResult` with `correlation`, `distance`, `linkage`, `order`, `figure`, and optional `warning`.

### Notes

Needs at least two features for clustering.

### Example

```python
res = cluster_features(df, ["x1", "x2", "x3"], plot=False)
```

## `pca_analysis(data, features=None, n_components=None, scale=True, max_rows=None, random_state=42, plot=True)`

### Purpose

Run PCA on feature data.

### Inputs

- `data`: feature DataFrame.
- `features`: optional selected columns.
- `n_components`: number of components.
- `scale`: standardize features before PCA.
- `max_rows`: optional row subsampling.
- `random_state`: seed.
- `plot`: include cumulative explained variance plot.

### Main Calculations

Fits sklearn PCA and returns explained variance, loadings, transformed scores, and model.

### Returns

`EDAResult` with `explained_variance`, `loadings`, `transformed`, `model`, and `figure`.

### Notes

Scaling is enabled by default because PCA is sensitive to feature units.

### Example

```python
pca = pca_analysis(df, ["x1", "x2", "x3"], scale=True, plot=False)
```

## `tsne_projection(data, features=None, scale=True, by_features=False, perplexity=30.0, max_rows=10000, random_state=42)`

### Purpose

Compute two-dimensional t-SNE projection.

### Inputs

- `data`: feature DataFrame.
- `features`: optional selected columns.
- `scale`: standardize features.
- `by_features`: embed features instead of observations.
- `perplexity`: t-SNE perplexity.
- `max_rows`: optional row subsampling.
- `random_state`: seed.

### Main Calculations

Uses sklearn `TSNE`.

### Returns

`pd.DataFrame` with `tsne_1` and `tsne_2`.

### Notes

t-SNE is exploratory and can be expensive.

### Example

```python
emb = tsne_projection(df, ["x1", "x2", "x3"], by_features=True)
```

## `feature_relation_report(data, features, method="pearson", scale=True, max_rows=10000, plot=True, run_tsne=True)`

### Purpose

Aggregate feature-feature diagnostics.

### Inputs

Same as `EDA.feature_relation_report`, but pass `data` explicitly.

### Main Calculations

Calls heatmap correlation, VIF, clustering, PCA, and optional t-SNE.

### Returns

`EDAResult` with `correlation`, `vif`, `clustering`, `pca`, `tsne`, and optional `tsne_warning`.

### Notes

Use `run_tsne=False` for faster reports.

### Example

```python
res = feature_relation_report(df, ["x1", "x2", "x3"], plot=False, run_tsne=False)
```

## `realized_volatility(returns, window, min_periods=None, annualization=None)`

### Purpose

Compute trailing realized volatility.

### Inputs

- `returns`: return series.
- `window`: trailing window.
- `min_periods`: minimum observations.
- `annualization`: optional annualization factor.

### Main Calculations

<p align="center">
  <img src="https://latex.codecogs.com/svg.image?RV_t=\sqrt{\sum_{i=t-w+1}^{t}r_i^2}" />
</p>

### Returns

`pd.Series`.

### Notes

If `annualization` is supplied, output is multiplied by its square root.

### Example

```python
rv = realized_volatility(df["return"], window=1440)
```

## `bipower_variation(returns, window, min_periods=None)`

### Purpose

Compute trailing bipower variation.

### Inputs

- `returns`: return series.
- `window`: trailing window.
- `min_periods`: minimum observations.

### Main Calculations

Uses lagged absolute returns:

<p align="center">
  <img src="https://latex.codecogs.com/svg.image?BV_t=\frac{\pi}{2}\sum_{i=t-w+1}^{t}|r_i||r_{i-1}|" />
</p>

### Returns

`pd.Series`.

### Notes

Often used as a jump-robust volatility diagnostic.

### Example

```python
bv = bipower_variation(df["return"], window=1440)
```

## `rolling_sharpe(returns, window, min_periods=None, periods_per_year=None)`

### Purpose

Compute trailing Sharpe-like mean/std diagnostic.

### Inputs

- `returns`: return series.
- `window`: trailing window.
- `min_periods`: minimum observations.
- `periods_per_year`: optional annualization factor.

### Main Calculations

Computes rolling mean divided by rolling standard deviation.

### Returns

`pd.Series`.

### Notes

This is a diagnostic, not a full performance attribution.

### Example

```python
rs = rolling_sharpe(df["return"], window=1440)
```

## `drawdown_diagnostics(series, input_type="returns")`

### Purpose

Compute equity and drawdown diagnostics.

### Inputs

- `series`: returns, price, or equity curve.
- `input_type`: `"returns"`, `"price"`, or `"equity"`.

### Main Calculations

Builds normalized equity and drawdown curve.

### Returns

`EDAResult` with `equity`, `drawdown`, and `summary`.

### Notes

For returns input, equity is computed as cumulative product of `1 + return`.

### Example

```python
dd = drawdown_diagnostics(df["return"], input_type="returns")
```

## `tail_dependence(x, y, q=0.95, tail="upper")`

### Purpose

Estimate empirical tail dependence.

### Inputs

- `x`: first series.
- `y`: second series.
- `q`: tail quantile.
- `tail`: `"upper"` or `"lower"`.

### Main Calculations

Estimates conditional probability that `y` is in its tail given that `x` is in its tail.

### Returns

`pd.DataFrame` with `tail`, `q`, `n_tail_x`, and `tail_dependence`.

### Notes

This is empirical and sample-size sensitive.

### Example

```python
tail_dependence(df["btc_return"], df["eth_return"], q=0.95)
```

## `upside_downside_volatility(returns, threshold=0.0)`

### Purpose

Split volatility into upside and downside parts.

### Inputs

- `returns`: return series.
- `threshold`: split point.

### Main Calculations

Computes standard deviation above and below the threshold.

### Returns

`pd.DataFrame` with upside/downside volatility, counts, and downside share.

### Notes

Useful for asymmetric risk checks.

### Example

```python
upside_downside_volatility(df["return"])
```

## `hit_rate(prediction, target, threshold=0.0)`

### Purpose

Compute directional hit rate.

### Inputs

- `prediction`: signal or prediction series.
- `target`: aligned target series.
- `threshold`: prediction threshold.

### Main Calculations

Compares signs of prediction and target.

### Returns

`pd.DataFrame` with `n`, `coverage`, and `hit_rate`.

### Notes

Rows are aligned by index and invalid rows are dropped.

### Example

```python
hit_rate(df["signal"], y)
```

## `turnover_cost_diagnostics(positions, returns=None, cost=0.0005)`

### Purpose

Compute turnover and optional cost-aware strategy returns.

### Inputs

- `positions`: position or signal series.
- `returns`: optional return series.
- `cost`: per-unit turnover cost.

### Main Calculations

Turnover is absolute position change. If returns are provided, position at `t-1` is applied to return at `t`.

### Returns

`EDAResult` with:

- `turnover`;
- `summary`;
- optional `strategy_returns`.

### Notes

The `t-1` position shift avoids look-ahead bias.

### Example

```python
res = turnover_cost_diagnostics(positions, returns=df["return"], cost=0.0005)
```

## `ljung_box_tests(returns, lags=(10, 20, 50), squared=True)`

### Purpose

Run Ljung-Box autocorrelation tests.

### Inputs

- `returns`: return series.
- `lags`: lags to test.
- `squared`: also test squared returns.

### Main Calculations

Uses `statsmodels.stats.diagnostic.acorr_ljungbox`.

### Returns

`pd.DataFrame` with rows for returns and optionally squared returns.

### Notes

Squared returns are useful for volatility clustering diagnostics.

### Example

```python
ljung_box_tests(df["return"], lags=[10, 20, 50])
```

## `missingness_by_time_bucket(series, bucket="hour")`

### Purpose

Compute missingness by calendar bucket.

### Inputs

- `series`: `pd.Series` with `DatetimeIndex`.
- `bucket`: `"hour"`, `"dayofweek"`, or `"month"`.

### Main Calculations

Groups missing indicators by calendar bucket.

### Returns

`pd.DataFrame` with bucket and `missing_pct`.

### Notes

Requires a `DatetimeIndex`.

### Example

```python
missingness_by_time_bucket(df["close"], bucket="hour")
```

## `calendar_seasonality(series, bucket="hour")`

### Purpose

Compute calendar bucket seasonality.

### Inputs

- `series`: numeric `pd.Series` with `DatetimeIndex`.
- `bucket`: `"hour"`, `"dayofweek"`, or `"month"`.

### Main Calculations

Groups values by calendar bucket and computes count, mean, median, std, and skew.

### Returns

`pd.DataFrame`.

### Notes

Requires a `DatetimeIndex`.

### Example

```python
calendar_seasonality(df["return"], bucket="dayofweek")
```

## `set_plot_style()`

### Purpose

Apply module-wide matplotlib defaults.

### Inputs

No inputs.

### Main Calculations

Sets lightweight plotting parameters.

### Returns

`None`.

### Notes

Called automatically by `EDA(..., cold_palette=True)`.

### Example

```python
set_plot_style()
```

## `style_axis(ax, xlabel="", ylabel="", title=None, grid=False, legend=True)`

### Purpose

Apply shared minimalist axis styling.

### Inputs

- `ax`: matplotlib `Axes`.
- `xlabel`: x-axis label.
- `ylabel`: y-axis label.
- `title`: optional title.
- `grid`: show light grid.
- `legend`: show legend if handles exist.

### Main Calculations

Styles spines, grid, labels, and legend.

### Returns

The same matplotlib `Axes`.

### Notes

Used internally by plotting functions.

### Example

```python
fig, ax = plt.subplots()
ax.plot(df.index, df["return"])
style_axis(ax, ylabel="Return")
```

## Statistical Tests

## ADF Test

Purpose: stationarity / unit-root test.

Null hypothesis `H0`: the series has a unit root.

Alternative hypothesis `H1`: the series is stationary.

A small p-value means evidence against a unit root.

<p align="center">
  <img src="https://latex.codecogs.com/svg.image?\Delta%20y_t=\alpha+\beta%20t+\gamma%20y_{t-1}+\sum_{i=1}^{p}\delta_i\Delta%20y_{t-i}+\varepsilon_t" />
</p>

Unreliable when the sample is too short, constant, strongly regime-changing, or badly misspecified.

## KPSS Test

Purpose: stationarity test with the opposite null from ADF.

`H0`: the series is stationary around a level or trend.

`H1`: the series is non-stationary.

A small p-value means evidence against stationarity.

Unreliable with short samples, structural breaks, or bad lag choice.

## Zivot-Andrews Test

Purpose: unit-root test allowing one endogenous structural break.

`H0`: unit root without structural break.

`H1`: trend-stationary process with one structural break.

A small p-value means evidence against the unit-root null.

Unreliable on short samples or when there are many breaks.

## Jarque-Bera Test

Purpose: test whether skewness and kurtosis match normality.

`H0`: skewness and kurtosis match a normal distribution.

`H1`: distribution is not normal.

<p align="center">
  <img src="https://latex.codecogs.com/svg.image?JB=\frac{n}{6}\left(S^2+\frac{(K-3)^2}{4}\right)" />
</p>

A small p-value suggests non-normality. On large samples, even tiny deviations can be significant.

## Shapiro-Wilk Test

Purpose: normality test.

`H0`: data comes from a normal distribution.

`H1`: data does not come from a normal distribution.

The module subsamples large samples for Shapiro-Wilk and records that in the warning column.

## Anderson-Darling Test

Purpose: distribution fit diagnostic.

`H0`: data follows the selected distribution.

`H1`: data does not follow it.

The module uses the normal distribution and reports the statistic and 5% critical value.

## Kolmogorov-Smirnov Test

Purpose: compare sample distribution to a reference distribution.

`H0`: data follows the reference normal distribution.

`H1`: data does not follow it.

The module estimates mean and standard deviation from the sample, so this is a practical diagnostic rather than a strict Lilliefors-corrected test.

## Engle ARCH LM Test

Purpose: test volatility clustering.

`H0`: no ARCH effects.

`H1`: ARCH effects exist.

A small p-value suggests conditional heteroskedasticity.

Unreliable when returns overlap heavily unless you use `step=h`.

## Ljung-Box Test

Purpose: test autocorrelation up to selected lags.

`H0`: no autocorrelation up to the selected lag.

`H1`: autocorrelation exists.

The module can run it on returns and squared returns. Squared-return autocorrelation is useful for volatility clustering.

## Granger Causality Test

Purpose: test whether lagged feature values help predict the target.

`H0`: the feature does not Granger-cause the target.

`H1`: lagged feature values improve prediction of the target.

A small p-value means the feature has predictive content in this test setup.

Granger causality is not true economic causality.

## CUSUM Test

Purpose: test parameter instability around the sample mean.

`H0`: stable parameters.

`H1`: instability or structural change.

A small p-value suggests instability.

## Metrics and Formulas

## Absolute Return

<p align="center">
  <img src="https://latex.codecogs.com/svg.image?|r_{t,h}|" />
</p>

Used to compare return magnitudes across horizons.

## Probability That Return Exceeds Cost

<p align="center">
  <img src="https://latex.codecogs.com/svg.image?\Pr(|r_{t,h}|>c)" />
</p>

Used in `target_selection` and `rolling_target_probability`.

## Mean Excess Over Cost

<p align="center">
  <img src="https://latex.codecogs.com/svg.image?\mathbb{E}\left[\max(|r_{t,h}|-c,0)\right]" />
</p>

In the code, `mean_excess` is computed as the unconditional average of clipped excess over cost.

## Information Coefficient

<p align="center">
  <img src="https://latex.codecogs.com/svg.image?IC=\operatorname{corr}(x_t,r_{t,h})" />
</p>

Pearson IC measures linear correlation. Spearman IC measures rank correlation and is often more robust for noisy financial features.

## HAC / Newey-West T-Statistic

<p align="center">
  <img src="https://latex.codecogs.com/svg.image?t=\frac{\bar{IC}}{SE_{HAC}(\bar{IC})}" />
</p>

HAC standard errors are useful when observations are autocorrelated or when forward returns overlap.

## Rolling Correlation / Rolling IC

<p align="center">
  <img src="https://latex.codecogs.com/svg.image?IC_t^{(w)}=\operatorname{corr}(x_{t-w+1:t},r_{t-w+1:t,h})" />
</p>

The module uses trailing windows only.

## Hill Tail Index

<p align="center">
  <img src="https://latex.codecogs.com/svg.image?\hat{\alpha}^{-1}=\frac{1}{k}\sum_{i=1}^{k}\log\left(\frac{X_{(i)}}{X_{(k+1)}}\right)" />
</p>

Smaller alpha means heavier tails.

## VIF

<p align="center">
  <img src="https://latex.codecogs.com/svg.image?VIF_j=\frac{1}{1-R_j^2}" />
</p>

High VIF means the feature is strongly linearly dependent on other features.

## PCA Explained Variance

<p align="center">
  <img src="https://latex.codecogs.com/svg.image?\text{ExplainedVarianceRatio}_j=\frac{\lambda_j}{\sum_k\lambda_k}" />
</p>

Used to understand how much variation each principal component explains.

## Realized Volatility

<p align="center">
  <img src="https://latex.codecogs.com/svg.image?RV_t=\sqrt{\sum_{i=t-w+1}^{t}r_i^2}" />
</p>

Computed with trailing windows.

## Bipower Variation

<p align="center">
  <img src="https://latex.codecogs.com/svg.image?BV_t=\frac{\pi}{2}\sum_{i=t-w+1}^{t}|r_i||r_{i-1}|" />
</p>

Used as a jump-robust volatility diagnostic.

## Plotting Style

The module uses a minimal plotting style designed for notebooks:

- cold blue palette;
- no unnecessary titles;
- no unnecessary x-axis label such as `"Date"`;
- light or disabled grid;
- reduced chart borders;
- readable figure sizes.

Palette:

```python
["#6EC6FF", "#2E86DE", "#003B5C"]
```

Shared helpers:

```python
set_plot_style()
style_axis(ax)
```

## Result Objects

Most report functions return `EDAResult`, a `dict` subclass.

Example:

```python
result = eda.target_selection(horizons=[1, 5, 10], plot=False)

result.keys()
result["summary"]
```

Main result keys:

| Function / Method | Keys |
|---|---|
| `target_selection` | `summary`, `targets`, `rolling_probability`, `figure` |
| `data_diagnostics` | `summary`, `stationarity`, `acf_pacf`, `rolling`, `figures`, `warnings` |
| `distribution_report` | `summary`, `density`, `qq`, `hill`, `evt`, `normality`, `arch_lm`, `class_balance` |
| `seasonality_report` | `periodogram`, `lomb_scargle`, `hurst_rs`, `cusum`, `structural_breaks`, `stl` |
| `feature_target_report` | `ic`, `cumulative_ic`, `rolling_ic`, `rolling_ic_stats`, `quantiles`, `mutual_information`, `distance_correlation`, `granger`, `figures`, `warnings` |
| `feature_relation_report` | `correlation`, `vif`, `clustering`, `pca`, `tsne`, optional `tsne_warning` |
| `acf_pacf` | `table`, `figure` |
| `density_plot` | `density`, `warning`, `figure` |
| `qq_plot` | `fit`, `figure` |
| `evt_gpd_fit` | `fit`, `excess`, `figure` |
| `periodogram` | `table`, `figure` |
| `lomb_scargle_periodogram` | `table`, `figure` |
| `stl_decomposition` | `components`, `period`, `figure` |
| `structural_breaks` | `breaks`, `sample_step`, `warning` |
| `cumulative_ic` | `table`, `figure` |
| `cross_sectional_ic` | `ic_by_time`, `summary` |
| `heatmap_correlation_matrix` | `correlation`, `figure` |
| `cluster_features` | `correlation`, `distance`, `linkage`, `order`, `figure`, optional `warning` |
| `pca_analysis` | `explained_variance`, `loadings`, `transformed`, `model`, `figure` |
| `drawdown_diagnostics` | `equity`, `drawdown`, `summary` |
| `turnover_cost_diagnostics` | `turnover`, `summary`, optional `strategy_returns` |

Because `EDAResult` is a dictionary, all nested results remain accessible:

```python
dist = eda.distribution_report("return", plot=False)

dist["hill"]
dist["evt"]["fit"]
dist["density"]["density"]
```

## Practical Workflow

1. Check data quality.

```python
diag = eda.data_diagnostics(cols=["close", "return"], plot=False)
```

2. Build and compare forward-return targets.

```python
targets = eda.target_selection([1, 5, 10, 20], cost=0.0005, plot=False)
```

3. Check distribution and tails.

```python
dist = eda.distribution_report("return", q=0.99, plot=False)
```

4. Check seasonality and structural breaks.

```python
season = eda.seasonality_report("return", period=1440, plot=False)
```

5. Test feature-target links.

```python
ft = eda.feature_target_report(
    features=["volume", "spread", "feature_1"],
    horizons=[1, 5, 10],
    method="spearman",
    plot=False,
)
```

6. Check feature-feature redundancy.

```python
fr = eda.feature_relation_report(
    features=["volume", "spread", "feature_1"],
    plot=False,
)
```

7. Use the results to decide whether a hypothesis is worth modeling.


