"""Compact EDA toolkit for crypto time series and quant hypothesis checks.

The module is designed for notebook usage and can be imported either as a
library of standalone functions or through the convenience :class:`EDA` facade.
Report functions return :class:`EDAResult`, a ``dict`` subclass that renders
tables cleanly in Jupyter while preserving normal dictionary access.

Core Assumptions
----------------
* Forward targets are aligned at decision time ``t``:
  ``forward_return[t] = log(price[t + h] / price[t])``.  Features observed at
  ``t`` should be joined to that target at the same index.
* Rolling statistics use pandas' trailing windows, i.e. past/current
  observations only.  The helpers do not intentionally use future feature
  values.
* Overlapping ``h``-bar returns need special care in tests.  Use ``step=h`` in
  ACF/PACF, stationarity, normality and ARCH diagnostics when you want a
  non-overlapping subsample.  Feature-target HAC/Newey-West standard errors use
  at least ``h - 1`` lags when forward targets are created from horizons.
* NaN and infinite values are either dropped with explicit validation or kept as
  NaN where preserving target alignment matters.

Public API Index
----------------
Target construction and horizon choice
    ``forward_return``, ``target_selection``, ``rolling_target_probability``.

Data quality and stationarity
    ``missing_pct``, ``series_summary``, ``data_diagnostics``,
    ``acf_pacf``, ``stationarity_tests``, ``stationarity_summary``,
    ``adf_test``, ``kpss_test``, ``zivot_andrews_test``, rolling moment helpers.

Distribution and tail diagnostics
    ``distribution_report``, ``qq_plot``, ``density_plot``,
    ``normality_tests``, ``arch_lm_test``, ``hill_estimator`` for Hill tail
    index alpha, ``evt_gpd_fit`` for GPD tail fitting, ``class_balance``.

Seasonality, regimes and structural breaks
    ``seasonality_report``, ``periodogram``, ``lomb_scargle_periodogram``,
    ``stl_decomposition``, ``hurst_exponent``, ``cusum_test``,
    ``structural_breaks``, ``missingness_by_time_bucket``,
    ``calendar_seasonality``.

Feature-target diagnostics
    ``feature_target_report``, ``feature_target_correlation``, ``ic_summary``,
    ``cumulative_ic``, ``rolling_ic``, ``rolling_ic_stats``,
    ``feature_quantile_stats``, ``granger_causality``, ``mutual_information``,
    ``rolling_mutual_information``, ``distance_correlation``,
    ``rolling_distance_correlation``, ``conditional_ic``,
    ``rolling_conditional_ic``.

Feature-feature diagnostics
    ``feature_relation_report``, ``correlation_matrix``,
    ``heatmap_correlation_matrix``, ``vif`` for VIF tables,
    ``cluster_features``, ``pca_analysis``, ``tsne_projection``.

Trading and risk utilities
    ``realized_volatility``, ``bipower_variation``, ``rolling_sharpe``,
    ``drawdown_diagnostics``, ``tail_dependence``,
    ``upside_downside_volatility``, ``hit_rate``,
    ``turnover_cost_diagnostics``, ``ljung_box_tests``.

Plotting and notebook helpers
    ``set_plot_style``, ``style_axis``, ``EDAResult``.  Figures use a minimalist
    cold-blue style matching the notebooks in this project.

Typical Usage
-------------
>>> import pandas as pd
>>> from eda import EDA
>>> df = pd.DataFrame({"datetime": pd.date_range("2024-01-01", periods=100),
...                    "close": 100 + pd.Series(range(100)).astype(float)})
>>> eda = EDA(df, time_col="datetime", price_col="close")
>>> result = eda.target_selection(horizons=[1, 5, 10], cost=0.0005, plot=False)
>>> result["summary"].head()
"""

from __future__ import annotations

import html
import math
import warnings
from collections.abc import Iterable, Sequence
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy import signal, stats
from scipy.cluster.hierarchy import dendrogram, leaves_list, linkage
from scipy.spatial.distance import squareform
from scipy.stats import genpareto
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.diagnostic import acorr_ljungbox, breaks_cusumolsresid, het_arch
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import (
    acf as sm_acf,
    adfuller,
    grangercausalitytests,
    kpss,
    pacf as sm_pacf,
    zivot_andrews,
)


COLD_PALETTE = ["#6EC6FF", "#2E86DE", "#003B5C"]
__all__ = [
    "COLD_PALETTE",
    "EDA",
    "EDAResult",
    "acf_pacf",
    "adf_test",
    "arch_lm_test",
    "bipower_variation",
    "calendar_seasonality",
    "class_balance",
    "cluster_features",
    "conditional_ic",
    "correlation_matrix",
    "cross_sectional_ic",
    "cusum_test",
    "cumulative_ic",
    "data_diagnostics",
    "density_plot",
    "distance_correlation",
    "distribution_report",
    "drawdown_diagnostics",
    "evt_gpd_fit",
    "feature_quantile_stats",
    "feature_relation_report",
    "feature_target_correlation",
    "feature_target_report",
    "forward_return",
    "granger_causality",
    "heatmap_correlation_matrix",
    "hill_estimator",
    "hit_rate",
    "hurst_exponent",
    "ic_summary",
    "kpss_test",
    "ljung_box_tests",
    "lomb_scargle_periodogram",
    "missing_pct",
    "missingness_by_time_bucket",
    "mutual_information",
    "normality_tests",
    "pca_analysis",
    "periodogram",
    "qq_plot",
    "realized_volatility",
    "rolling_conditional_ic",
    "rolling_distance_correlation",
    "rolling_ic",
    "rolling_ic_stats",
    "rolling_kurtosis",
    "rolling_mean",
    "rolling_median",
    "rolling_mode",
    "rolling_mutual_information",
    "rolling_sharpe",
    "rolling_skewness",
    "rolling_std",
    "rolling_target_probability",
    "seasonality_report",
    "series_summary",
    "set_plot_style",
    "stationarity_summary",
    "stationarity_tests",
    "structural_breaks",
    "style_axis",
    "tail_dependence",
    "target_selection",
    "tsne_projection",
    "turnover_cost_diagnostics",
    "upside_downside_volatility",
    "zivot_andrews_test",
]


def set_plot_style() -> None:
    """Set lightweight matplotlib defaults used by this module.

    The function is intentionally small and non-invasive.  It does not switch
    global styles such as seaborn; it only tunes defaults that make notebook
    figures readable.
    """

    plt.rcParams.update(
        {
            "axes.prop_cycle": plt.cycler(color=COLD_PALETTE),
            "axes.grid": False,
            "figure.figsize": (8, 4.5),
            "font.size": 10,
            "legend.frameon": False,
        }
    )


def style_axis(
    ax: Axes,
    *,
    xlabel: str | None = "",
    ylabel: str | None = "",
    title: str | None = None,
    grid: bool = False,
    legend: bool = True,
) -> Axes:
    """Apply the shared minimalist axis style.

    Parameters
    ----------
    ax:
        Matplotlib axis to style.
    xlabel, ylabel, title:
        Optional labels.  The default x-label is blank to avoid redundant
        ``Date`` labels when the index is already temporal.
    grid:
        If ``True``, draw a light y-axis grid.
    legend:
        If ``True`` and labelled artists are present, show a frameless legend.

    Returns
    -------
    matplotlib.axes.Axes
        The same axis, for chaining.
    """

    ax.set_xlabel("" if xlabel is None else xlabel)
    ax.set_ylabel("" if ylabel is None else ylabel)
    if title is not None:
        ax.set_title(title)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(grid, axis="y", alpha=0.18, linewidth=0.8)
    if legend:
        handles, labels = ax.get_legend_handles_labels()
        if labels:
            ax.legend(frameon=False)
    return ax


class EDAResult(dict):
    """Dictionary result with a clean HTML representation for notebooks.

    Analytical functions still return normal mapping semantics: use
    ``result["summary"]``, ``result.keys()`` or ``dict(result)`` exactly as with
    a regular dict.  In Jupyter/IPython, the last expression in a cell is
    rendered as titled sections with DataFrames/Series shown as HTML tables
    instead of a raw nested-dict repr.

    Parameters
    ----------
    data:
        Mapping with result objects such as DataFrames, Series, nested dicts,
        warnings and matplotlib figures.
    title:
        Optional title shown above the rendered result.
    max_rows:
        Maximum rows rendered per table preview.  Full objects remain available
        through regular dictionary access.
    """

    def __init__(self, data: dict[str, Any] | None = None, *, title: str = "EDA result", max_rows: int = 20) -> None:
        super().__init__(data or {})
        self.title = title
        self.max_rows = max_rows

    def _preview_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        if len(df) <= self.max_rows:
            return df
        head_n = max(1, self.max_rows // 2)
        tail_n = max(1, self.max_rows - head_n)
        ellipsis = pd.DataFrame({c: ["..."] for c in df.columns}, index=["..."])
        return pd.concat([df.head(head_n), ellipsis, df.tail(tail_n)])

    def _repr_html_value(self, key: Any, value: Any, level: int = 3) -> str:
        label = html.escape(str(key))
        tag = f"h{min(level, 5)}"
        if isinstance(value, pd.DataFrame):
            table = self._preview_frame(value).to_html(classes="eda-table", border=0, notebook=True)
            note = f'<div class="eda-note">{len(value):,} rows x {len(value.columns):,} columns</div>'
            return f"<section><{tag}>{label}</{tag}>{note}{table}</section>"
        if isinstance(value, pd.Series):
            return self._repr_html_value(key, value.to_frame(), level=level)
        if isinstance(value, dict):
            if not value:
                return f"<section><{tag}>{label}</{tag}><div class='eda-note'>empty</div></section>"
            inner = "".join(self._repr_html_value(k, v, level + 1) for k, v in value.items())
            return f"<section><{tag}>{label}</{tag}>{inner}</section>"
        if isinstance(value, (list, tuple)) and value and all(isinstance(v, Figure) for v in value):
            return f"<section><{tag}>{label}</{tag}><div class='eda-note'>{len(value)} matplotlib figure(s)</div></section>"
        if isinstance(value, Figure):
            return f"<section><{tag}>{label}</{tag}><div class='eda-note'>matplotlib Figure</div></section>"
        if isinstance(value, np.ndarray):
            return f"<section><{tag}>{label}</{tag}><div class='eda-note'>ndarray shape={value.shape}, dtype={value.dtype}</div></section>"
        if value is None or (isinstance(value, list) and not value):
            return f"<section><{tag}>{label}</{tag}><div class='eda-note'>None</div></section>"
        if isinstance(value, (list, tuple)):
            items = "".join(f"<li>{html.escape(str(v))}</li>" for v in value[: self.max_rows])
            more = f"<li>... {len(value) - self.max_rows} more</li>" if len(value) > self.max_rows else ""
            return f"<section><{tag}>{label}</{tag}><ul>{items}{more}</ul></section>"
        return f"<section><{tag}>{label}</{tag}><code>{html.escape(str(value))}</code></section>"

    def _repr_html_(self) -> str:
        css = """
        <style>
        .eda-result {font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;}
        .eda-result h2 {font-size: 18px; margin: 0 0 10px 0;}
        .eda-result h3, .eda-result h4, .eda-result h5 {font-size: 14px; margin: 16px 0 6px 0;}
        .eda-result .eda-note {color: #52606d; font-size: 12px; margin: 0 0 4px 0;}
        .eda-result table.eda-table {border-collapse: collapse; font-size: 12px;}
        .eda-result table.eda-table th, .eda-result table.eda-table td {
            border-bottom: 1px solid #e6eef5; padding: 4px 8px; text-align: right;
        }
        .eda-result table.eda-table th {background: #f5f9fc; color: #003B5C; font-weight: 600;}
        </style>
        """
        body = "".join(self._repr_html_value(k, v) for k, v in self.items())
        return f'{css}<div class="eda-result"><h2>{html.escape(self.title)}</h2>{body}</div>'


def _result(title: str, data: dict[str, Any]) -> EDAResult:
    return EDAResult(data, title=title)


def _as_series(x: pd.Series | pd.DataFrame | Sequence[float], name: str | None = None) -> pd.Series:
    if isinstance(x, pd.DataFrame):
        if x.shape[1] != 1:
            raise ValueError("Expected a Series or one-column DataFrame.")
        x = x.iloc[:, 0]
    elif not isinstance(x, pd.Series):
        x = pd.Series(x, name=name)
    if name is not None and x.name is None:
        x = x.rename(name)
    return x


def _numeric_series(
    x: pd.Series | pd.DataFrame | Sequence[float],
    *,
    name: str | None = None,
    dropna: bool = True,
    min_obs: int = 1,
    warn: bool = True,
) -> pd.Series:
    s = pd.to_numeric(_as_series(x, name=name), errors="coerce").replace([np.inf, -np.inf], np.nan)
    bad = int(s.isna().sum())
    if warn and bad:
        warnings.warn(f"{s.name or 'series'}: dropped/kept {bad} NaN or infinite values.", RuntimeWarning)
    if dropna:
        s = s.dropna()
    if len(s) < min_obs:
        raise ValueError(f"{s.name or 'series'} needs at least {min_obs} finite observations; got {len(s)}.")
    return s


def _as_frame(data: pd.Series | pd.DataFrame, cols: Sequence[str] | None = None) -> pd.DataFrame:
    if isinstance(data, pd.Series):
        df = data.to_frame()
    elif isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        raise TypeError("Expected pandas Series or DataFrame.")
    if cols is not None:
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        df = df.loc[:, list(cols)]
    return df


def _numeric_frame(
    data: pd.Series | pd.DataFrame,
    cols: Sequence[str] | None = None,
    *,
    dropna: bool = True,
    min_obs: int = 1,
) -> pd.DataFrame:
    df = _as_frame(data, cols).apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if dropna:
        df = df.dropna()
    if len(df) < min_obs:
        raise ValueError(f"Need at least {min_obs} finite rows; got {len(df)}.")
    return df


def _validate_horizons(horizons: Iterable[int]) -> list[int]:
    hs = sorted({int(h) for h in horizons})
    if not hs or any(h <= 0 for h in hs):
        raise ValueError("horizons must contain positive integers.")
    return hs


def _validate_window(window: int, min_periods: int | None = None) -> tuple[int, int]:
    window = int(window)
    if window < 2:
        raise ValueError("rolling window must be at least 2.")
    if min_periods is None:
        min_periods = max(2, window // 3)
    min_periods = int(min_periods)
    if min_periods < 1 or min_periods > window:
        raise ValueError("min_periods must be between 1 and window.")
    return window, min_periods


def _maybe_sample_for_tests(
    s: pd.Series,
    *,
    step: int = 1,
    max_test_size: int | None = 100_000,
    random_state: int | None = None,
) -> pd.Series:
    if step < 1:
        raise ValueError("step must be >= 1.")
    x = s.iloc[::step] if step > 1 else s
    if max_test_size is not None and len(x) > max_test_size:
        if random_state is None:
            x = x.iloc[-max_test_size:]
        else:
            x = x.sample(max_test_size, random_state=random_state).sort_index()
    return x


def _result_row(test: str, statistic: float = np.nan, pvalue: float = np.nan, warning: str = "") -> dict[str, Any]:
    return {"test": test, "statistic": statistic, "pvalue": pvalue, "warning": warning}


def _test_series_or_warning(series: pd.Series, test: str, min_obs: int) -> tuple[pd.Series | None, pd.DataFrame | None]:
    x = pd.to_numeric(_as_series(series), errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(x) < min_obs:
        return None, pd.DataFrame([{**_result_row(test, warning=f"{test} needs at least {min_obs} finite observations; got {len(x)}."), "nobs": len(x)}])
    if x.nunique(dropna=True) < 2:
        return None, pd.DataFrame([{**_result_row(test, warning="Test is undefined for a constant series."), "nobs": len(x)}])
    return x, None


def _cost_threshold(cost: float, cost_is_multiplier: bool = False) -> float:
    if cost < 0:
        raise ValueError("cost must be non-negative.")
    if cost_is_multiplier:
        if cost <= 0:
            raise ValueError("cost multiplier must be positive.")
        return float(np.log(cost))
    if cost > 0.05:
        warnings.warn(
            "cost looks large for a return threshold. If you meant a price multiplier "
            "such as 1.002, pass cost_is_multiplier=True.",
            RuntimeWarning,
        )
    return float(cost)


def _newey_west_se_mean(x: pd.Series, lags: int | str | None = "auto", *, min_lags: int = 0) -> tuple[float, int]:
    v = _numeric_series(x, dropna=True, min_obs=3, warn=False).to_numpy(dtype=float)
    n = len(v)
    if lags in (None, "auto"):
        lag = int(round(4 * (n / 100) ** (2 / 9)))
    else:
        lag = int(lags)
    lag = max(int(min_lags), lag)
    lag = max(0, min(lag, n - 2))
    u = v - v.mean()
    gamma0 = float(np.dot(u, u) / n)
    lrv = gamma0
    for k in range(1, lag + 1):
        gamma = float(np.dot(u[k:], u[:-k]) / n)
        lrv += 2 * (1 - k / (lag + 1)) * gamma
    return float(np.sqrt(max(lrv, 0) / n)), lag


def _zscore(s: pd.Series) -> pd.Series:
    std = s.std(ddof=0)
    return (s - s.mean()) / std if np.isfinite(std) and std > 0 else s * np.nan


def _align_xy(
    features: pd.Series | pd.DataFrame,
    target: pd.Series,
    *,
    min_obs: int = 3,
) -> pd.DataFrame:
    x = _numeric_frame(features, dropna=False)
    y = _numeric_series(target, name="target", dropna=False, warn=False).rename("__target__")
    df = pd.concat([x, y], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    if len(df) < min_obs:
        raise ValueError(f"Need at least {min_obs} aligned finite observations; got {len(df)}.")
    return df


def _prepare_feature_matrix(
    data: pd.Series | pd.DataFrame,
    features: Sequence[str] | None = None,
    *,
    scale: bool = True,
    max_rows: int | None = None,
    random_state: int = 42,
) -> tuple[pd.DataFrame, np.ndarray]:
    df = _numeric_frame(data, features, dropna=True, min_obs=3)
    if max_rows is not None and len(df) > max_rows:
        df = df.sample(max_rows, random_state=random_state).sort_index()
    x = df.to_numpy(dtype=float)
    if scale:
        x = StandardScaler().fit_transform(x)
    return df, x


def forward_return(
    price: pd.Series,
    horizon: int = 1,
    *,
    log_return: bool = True,
    name: str | None = None,
) -> pd.Series:
    """Compute a forward return aligned at the current timestamp.

    This is the canonical target-construction helper used by the module.  The
    output is indexed at decision time ``t`` rather than realization time
    ``t + horizon``.

    Parameters
    ----------
    price:
        Price series indexed by time or row number.
    horizon:
        Positive number of bars ahead.
    log_return:
        If ``True``, return ``log(price.shift(-h) / price)``.  Otherwise return
        simple percentage return ``price.shift(-h) / price - 1``.
    name:
        Optional output series name.  Defaults to ``f"fwd_ret_{horizon}"``.

    Returns
    -------
    pandas.Series
        Forward return with the same index as ``price``.  The last ``horizon``
        rows are ``NaN`` because the future price is unavailable.

    Notes
    -----
    This is the central anti-look-ahead convention used by the module: a feature
    observed at index ``t`` should be joined to this target at the same ``t``.

    Examples
    --------
    >>> target = forward_return(df["close"], horizon=5)
    >>> aligned = pd.concat([df[["volume"]], target], axis=1).dropna()
    """

    h = _validate_horizons([horizon])[0]
    p = _numeric_series(price, name="price", dropna=False)
    if (p <= 0).any() and log_return:
        raise ValueError("log forward returns require strictly positive prices.")
    out = np.log(p.shift(-h) / p) if log_return else p.shift(-h) / p - 1
    return out.rename(name or f"fwd_ret_{h}")


def target_selection(
    close: pd.Series,
    horizons: Sequence[int],
    *,
    cost: float = 0.0005,
    cost_is_multiplier: bool = False,
    log_return: bool = True,
    rolling_window: int | None = None,
    min_periods: int | None = None,
    plot: bool = True,
) -> EDAResult:
    """Compare forward-return targets over multiple horizons.

    The function computes cost-aware target statistics for each horizon and can
    optionally produce a trailing rolling probability of exceeding transaction
    cost.  It is intended for selecting prediction horizons before modelling.

    Parameters
    ----------
    close:
        Price series.  For log returns all prices must be positive.
    horizons:
        Positive forecast horizons in bars.
    cost:
        Threshold in return units.  Example: ``0.0005`` means 5 bps.  To use the
        old notebook style ``1.002``, set ``cost_is_multiplier=True``.
    cost_is_multiplier:
        If ``True``, convert ``cost`` to ``log(cost)`` before comparisons.
    log_return:
        Use log or simple forward returns.
    rolling_window, min_periods:
        Optional trailing rolling window for the ``rolling_std`` table column and
        the probability plot.
    plot:
        If ``True``, include a rolling ``P(|r| > cost)`` figure.

    Returns
    -------
    EDAResult
        ``{"summary": DataFrame, "targets": DataFrame, "rolling_probability":
        DataFrame | None, "figure": Figure | None}``.

    Notes
    -----
    Targets are computed with :func:`forward_return`, so the target at index
    ``t`` uses the future price ``t + h`` but remains aligned to ``t``.  The
    final ``h`` rows are ``NaN`` and are not counted as cost failures in rolling
    probabilities.  Rolling windows are trailing only.

    Examples
    --------
    >>> res = target_selection(df["close"], horizons=[1, 5, 10], cost=0.0005, plot=False)
    >>> res["summary"][["horizon", "median_abs", "prob_abs_gt_cost"]]
    """

    hs = _validate_horizons(horizons)
    threshold = _cost_threshold(cost, cost_is_multiplier)
    if rolling_window is not None:
        rolling_window, min_periods = _validate_window(rolling_window, min_periods)

    targets: dict[str, pd.Series] = {}
    rolling_prob: dict[str, pd.Series] = {}
    rows = []
    for h in hs:
        r = forward_return(close, h, log_return=log_return)
        x = _numeric_series(r, dropna=True, min_obs=max(5, h + 1), warn=False)
        abs_x = x.abs()
        std = x.std()
        rolling_std = x.rolling(rolling_window, min_periods=min_periods).std().mean() if rolling_window else np.nan
        rows.append(
            {
                "horizon": h,
                "n": len(x),
                "mean_abs": abs_x.mean(),
                "median_abs": abs_x.median(),
                "prob_abs_gt_cost": (abs_x > threshold).mean(),
                "mean_abs_gt_cost": abs_x[abs_x > threshold].mean(),
                "mean_excess": (abs_x - threshold).clip(lower=0).mean(),
                "abs_mean_over_std": abs_x.mean() / std if std and np.isfinite(std) else np.nan,
                "skew": x.skew(),
                "kurtosis": x.kurtosis(),
                "rolling_std": rolling_std,
            }
        )
        targets[f"h{h}"] = r
        if rolling_window:
            event = r.abs().gt(threshold).astype(float).where(r.notna())
            rolling_prob[f"h{h}"] = event.rolling(rolling_window, min_periods=min_periods).mean().where(r.notna())

    summary = pd.DataFrame(rows).sort_values("horizon").reset_index(drop=True)
    rolling_df = pd.DataFrame(rolling_prob) if rolling_prob else None
    fig = None
    if plot and rolling_df is not None:
        fig, ax = plt.subplots(figsize=(9, 4.5))
        for i, col in enumerate(rolling_df):
            ax.plot(rolling_df.index, rolling_df[col], lw=1.8, color=COLD_PALETTE[i % len(COLD_PALETTE)], label=col)
        style_axis(ax, ylabel="Probability", grid=False)
        fig.tight_layout()
    return _result("Target selection", {"summary": summary, "targets": pd.DataFrame(targets), "rolling_probability": rolling_df, "figure": fig})


def rolling_target_probability(
    close: pd.Series,
    horizons: Sequence[int],
    *,
    cost: float = 0.0005,
    cost_is_multiplier: bool = False,
    window: int = 10_080,
    min_periods: int | None = None,
    log_return: bool = True,
    plot: bool = True,
) -> EDAResult:
    """Compute trailing ``P(|forward_return| > cost)`` for each horizon.

    Parameters are the same as :func:`target_selection`, with an explicit
    ``window`` in bars.  The rolling calculation is trailing and therefore does
    not use future events.

    Returns
    -------
    EDAResult
        Same keys as :func:`target_selection`.
    """

    window, min_periods = _validate_window(window, min_periods)
    return target_selection(
        close,
        horizons,
        cost=cost,
        cost_is_multiplier=cost_is_multiplier,
        log_return=log_return,
        rolling_window=window,
        min_periods=min_periods,
        plot=plot,
    )


def missing_pct(data: pd.Series | pd.DataFrame, cols: Sequence[str] | None = None) -> pd.Series:
    """Return percentage of missing or infinite values by column.

    Parameters
    ----------
    data:
        Series or DataFrame to inspect.
    cols:
        Optional DataFrame columns.

    Returns
    -------
    pandas.Series
        Missing percentage in ``[0, 100]``.

    Examples
    --------
    >>> missing_pct(df, cols=["close", "volume"])
    """

    df = _as_frame(data, cols).replace([np.inf, -np.inf], np.nan)
    return df.isna().mean().mul(100).rename("missing_pct")


def series_summary(
    data: pd.Series | pd.DataFrame,
    cols: Sequence[str] | None = None,
    *,
    quantiles: Sequence[float] = (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99),
) -> pd.DataFrame:
    """Summarize basic distribution and data-quality properties.

    Returns one row per column with counts, missingness, central moments,
    quantiles, mode, min and max.

    Parameters
    ----------
    data : pandas.Series or pandas.DataFrame
        Input data.
    cols : sequence of str, optional
        Selected DataFrame columns.
    quantiles : sequence of float
        Quantiles to include.

    Returns
    -------
    pandas.DataFrame
        One summary row per selected column.
    """

    df0 = _as_frame(data, cols)
    df = df0.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    rows = []
    for col in df:
        x = df[col].dropna()
        mode = x.mode().iloc[0] if not x.mode().empty else np.nan
        row = {
            "column": col,
            "n": int(len(x)),
            "missing_pct": df[col].isna().mean() * 100,
            "mean": x.mean(),
            "std": x.std(),
            "median": x.median(),
            "mode": mode,
            "skew": x.skew(),
            "kurtosis": x.kurtosis(),
            "min": x.min(),
            "max": x.max(),
        }
        row.update({f"q{q:g}": x.quantile(q) for q in quantiles})
        rows.append(row)
    return pd.DataFrame(rows)


def rolling_mean(series: pd.Series, window: int, min_periods: int | None = None) -> pd.Series:
    """Compute trailing rolling mean.

    Parameters
    ----------
    series : pandas.Series
        Numeric series.  NaN/inf values are kept as NaN.
    window : int
        Trailing window length in rows.
    min_periods : int, optional
        Minimum observations required in each window.

    Returns
    -------
    pandas.Series
        Rolling mean with the original index preserved.
    """

    window, min_periods = _validate_window(window, min_periods)
    return _numeric_series(series, dropna=False).rolling(window, min_periods=min_periods).mean()


def rolling_std(series: pd.Series, window: int, min_periods: int | None = None) -> pd.Series:
    """Compute trailing rolling standard deviation preserving the index.

    Parameters are the same as :func:`rolling_mean`.  The calculation uses only
    current and past observations in each pandas rolling window.
    """

    window, min_periods = _validate_window(window, min_periods)
    return _numeric_series(series, dropna=False).rolling(window, min_periods=min_periods).std()


def rolling_median(series: pd.Series, window: int, min_periods: int | None = None) -> pd.Series:
    """Compute trailing rolling median preserving the original index.

    Parameters are the same as :func:`rolling_mean`; output is a
    ``pandas.Series`` aligned to the input.

    Notes
    -----
    The window is trailing, so the statistic at time ``t`` only uses
    observations up to ``t``.
    """

    window, min_periods = _validate_window(window, min_periods)
    return _numeric_series(series, dropna=False).rolling(window, min_periods=min_periods).median()


def rolling_mode(series: pd.Series, window: int, min_periods: int | None = None) -> pd.Series:
    """Trailing rolling mode preserving the original index.

    If several values share the highest frequency, pandas' first sorted mode is
    returned.

    Parameters
    ----------
    series : pandas.Series
        Numeric or discrete series.
    window, min_periods : int
        Trailing rolling window controls.

    Returns
    -------
    pandas.Series
        Rolling mode aligned to the input index.
    """

    window, min_periods = _validate_window(window, min_periods)

    def mode_one(x: pd.Series) -> float:
        m = pd.Series(x).dropna().mode()
        return float(m.iloc[0]) if not m.empty else np.nan

    return _numeric_series(series, dropna=False).rolling(window, min_periods=min_periods).apply(mode_one, raw=False)


def rolling_skewness(series: pd.Series, window: int, min_periods: int | None = None) -> pd.Series:
    """Compute trailing rolling skewness preserving the original index.

    Use this for local asymmetry diagnostics.  Windows are trailing and therefore
    do not use future observations.

    Returns
    -------
    pandas.Series
        Rolling skewness aligned to the input index.
    """

    window, min_periods = _validate_window(window, min_periods)
    return _numeric_series(series, dropna=False).rolling(window, min_periods=min_periods).skew()


def rolling_kurtosis(series: pd.Series, window: int, min_periods: int | None = None) -> pd.Series:
    """Compute trailing rolling excess kurtosis preserving the original index.

    Use this for local tail-shape diagnostics.  Windows are trailing and NaN/inf
    inputs remain missing in the aligned output.
    """

    window, min_periods = _validate_window(window, min_periods)
    return _numeric_series(series, dropna=False).rolling(window, min_periods=min_periods).kurt()


def adf_test(series: pd.Series, *, maxlag: int | None = 10, regression: str = "c") -> pd.DataFrame:
    """Augmented Dickey-Fuller stationarity test.

    Parameters
    ----------
    series : pandas.Series
        Numeric series.  NaN/inf values are dropped.
    maxlag : int or None, default 10
        Maximum lag passed to statsmodels.  ``None`` enables AIC autolag.
    regression : str, default "c"
        Deterministic terms passed to statsmodels.

    Returns
    -------
    pandas.DataFrame
        One row with statistic, p-value, used lags, observations and warning.

    Notes
    -----
    Returns a one-row DataFrame with statistic, p-value, lags and warning text
    if the test cannot be computed.
    """

    x, warning = _test_series_or_warning(series, "ADF", 8)
    if warning is not None:
        warning["lags"] = np.nan
        return warning
    try:
        lag = min(maxlag, max(0, len(x) // 2 - 2)) if maxlag is not None else None
        stat, pvalue, usedlag, nobs, *_ = adfuller(x, maxlag=lag, autolag=None if lag is not None else "AIC", regression=regression)
        return pd.DataFrame([{**_result_row("ADF", stat, pvalue), "lags": usedlag, "nobs": nobs}])
    except Exception as exc:
        return pd.DataFrame([{**_result_row("ADF", warning=str(exc)), "lags": np.nan, "nobs": len(x)}])


def kpss_test(series: pd.Series, *, nlags: int | str = 10, regression: str = "c") -> pd.DataFrame:
    """KPSS stationarity test.

    Parameters
    ----------
    series : pandas.Series
        Numeric series.  NaN/inf values are dropped.
    nlags : int or str, default 10
        Lags passed to statsmodels KPSS.
    regression : {"c", "ct"}, default "c"
        Level or trend stationarity specification.

    Returns
    -------
    pandas.DataFrame
        One row with statistic, p-value, lags, observations and warning.

    Notes
    -----
    ``regression="c"`` tests level stationarity; ``"ct"`` tests trend
    stationarity.
    """

    x, warning = _test_series_or_warning(series, "KPSS", 8)
    if warning is not None:
        warning["lags"] = np.nan
        return warning
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            stat, pvalue, usedlags, _ = kpss(x, regression=regression, nlags=nlags)
        warning = "; ".join(str(w.message) for w in caught)
        return pd.DataFrame([{**_result_row("KPSS", stat, pvalue, warning), "lags": usedlags, "nobs": len(x)}])
    except Exception as exc:
        return pd.DataFrame([{**_result_row("KPSS", warning=str(exc)), "lags": np.nan, "nobs": len(x)}])


def zivot_andrews_test(series: pd.Series, *, maxlag: int | None = 10, regression: str = "c") -> pd.DataFrame:
    """Run Zivot-Andrews unit-root test with one endogenous structural break.

    Parameters
    ----------
    series : pandas.Series
        Numeric series.  NaN/inf values are dropped.
    maxlag : int or None, default 10
        Maximum lag passed to statsmodels.
    regression : str, default "c"
        Deterministic terms passed to statsmodels.

    Returns
    -------
    pandas.DataFrame
        One row with statistic, p-value, lag, break index/time and warning.

    Notes
    -----
    Requires a longer non-constant sample than ADF/KPSS.  Infeasible tests return
    warning rows instead of silently failing.
    """

    x, warning = _test_series_or_warning(series, "Zivot-Andrews", 30)
    if warning is not None:
        warning["lags"] = np.nan
        return warning
    try:
        lag = min(maxlag, max(0, len(x) // 3 - 2)) if maxlag is not None else None
        result = zivot_andrews(x, maxlag=lag, regression=regression)
        stat, pvalue, crit, baselag, break_idx = result[:5]
        break_time = x.index[int(break_idx)] if int(break_idx) < len(x) else np.nan
        return pd.DataFrame(
            [
                {
                    **_result_row("Zivot-Andrews", stat, pvalue),
                    "lags": baselag,
                    "break_index": int(break_idx),
                    "break_time": break_time,
                    "crit_5pct": crit.get("5%", np.nan) if isinstance(crit, dict) else np.nan,
                    "nobs": len(x),
                }
            ]
        )
    except Exception as exc:
        return pd.DataFrame([{**_result_row("Zivot-Andrews", warning=str(exc)), "lags": np.nan, "nobs": len(x)}])


def stationarity_tests(
    series: pd.Series,
    *,
    step: int = 1,
    max_test_size: int | None = 100_000,
    adf_lag: int | None = 10,
    kpss_lag: int | str = 10,
    za_lag: int | None = 10,
) -> pd.DataFrame:
    """Run ADF, KPSS and Zivot-Andrews on a cleaned, optionally sampled series.

    The function returns one table containing all three tests.  If a test cannot
    be run because the sample is too short or constant, the corresponding row
    contains ``NaN`` statistics and a human-readable ``warning``.

    Parameters
    ----------
    step:
        Use every ``step``-th observation before tests.  This is useful for
        heavily overlapping forward-return horizons.
    max_test_size:
        Keep the last ``max_test_size`` observations after stepping.  ``None``
        disables trimming.

    Returns
    -------
    pandas.DataFrame
        Rows for ADF, KPSS and Zivot-Andrews with statistics, p-values, lags,
        observations and warnings.

    Notes
    -----
    Use ``step=h`` for overlapping ``h``-bar returns.  ADF/KPSS can still return
    warnings/results on samples where Zivot-Andrews is too short.

    Examples
    --------
    >>> stationarity_tests(df["return"], step=5)
    """

    x = pd.to_numeric(_as_series(series), errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    x = _maybe_sample_for_tests(x, step=step, max_test_size=max_test_size) if len(x) else x
    return pd.concat(
        [
            adf_test(x, maxlag=adf_lag),
            kpss_test(x, nlags=kpss_lag),
            zivot_andrews_test(x, maxlag=za_lag),
        ],
        ignore_index=True,
    )


def stationarity_summary(data: pd.Series | pd.DataFrame, cols: Sequence[str] | None = None, **kwargs: Any) -> pd.DataFrame:
    """Run :func:`stationarity_tests` for each selected column.

    Parameters
    ----------
    data : pandas.Series or pandas.DataFrame
        Input series or frame.
    cols : sequence of str, optional
        Selected DataFrame columns.
    **kwargs
        Passed to :func:`stationarity_tests`, including ``step`` for
        overlapping returns.

    Returns
    -------
    pandas.DataFrame
        Combined stationarity table with a leading ``column`` field.
    """

    frames = []
    for col, s in _as_frame(data, cols).items():
        t = stationarity_tests(s, **kwargs)
        t.insert(0, "column", col)
        frames.append(t)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def acf_pacf(
    series: pd.Series,
    *,
    lags: int = 40,
    alpha: float | None = 0.05,
    step: int = 1,
    plot: bool = True,
) -> EDAResult:
    """Compute ACF and PACF values, with optional minimalist plots.

    Parameters
    ----------
    series : pandas.Series
        Numeric time series, typically returns or forward returns.
    lags : int, default 40
        Number of autocorrelation lags.
    alpha : float or None, default 0.05
        Confidence interval level passed to statsmodels.  If ``None``, interval
        columns are omitted.
    step : int, default 1
        Subsampling step after dropping NaN/inf values.  Use ``step=h`` for
        overlapping ``h``-bar returns.
    plot : bool, default True
        Include ACF/PACF bar plots.

    Use ``step=h`` for overlapping ``h``-bar forward returns when you want ACF
    and PACF on a non-overlapping subsample, equivalent to ``series.dropna()[::h]``.

    Returns
    -------
    EDAResult
        ``{"table": DataFrame, "figure": Figure | None}``.

    Notes
    -----
    The returned lag numbers are in units of the stepped sample.  With
    ``step=5``, lag 1 corresponds to 5 original bars.

    Examples
    --------
    >>> acf_pacf(five_min_returns, lags=24, step=5, plot=False)["table"].head()
    """

    if step < 1:
        raise ValueError("step must be >= 1.")
    x = _numeric_series(series, dropna=True, min_obs=1, warn=False)
    x = x.iloc[::step] if step > 1 else x
    if len(x) < lags + 3:
        raise ValueError(f"ACF/PACF needs at least {lags + 3} finite observations after step={step}; got {len(x)}.")
    acf_vals, acf_ci = sm_acf(x, nlags=lags, alpha=alpha, fft=True)
    pacf_vals, pacf_ci = sm_pacf(x, nlags=lags, alpha=alpha, method="ywm")
    table = pd.DataFrame({"lag": range(lags + 1), "acf": acf_vals, "pacf": pacf_vals, "step": step, "n_used": len(x)})
    if alpha is not None:
        table[["acf_ci_low", "acf_ci_high"]] = acf_ci
        table[["pacf_ci_low", "pacf_ci_high"]] = pacf_ci
    fig = None
    if plot:
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
        axes[0].bar(table["lag"], table["acf"], color=COLD_PALETTE[1], width=0.8)
        axes[1].bar(table["lag"], table["pacf"], color=COLD_PALETTE[2], width=0.8)
        style_axis(axes[0], xlabel="Lag", ylabel="ACF", legend=False)
        style_axis(axes[1], xlabel="Lag", ylabel="PACF", legend=False)
        fig.tight_layout()
    return _result("ACF/PACF", {"table": table, "figure": fig})


def data_diagnostics(
    data: pd.Series | pd.DataFrame,
    cols: Sequence[str] | None = None,
    *,
    rolling_windows: Sequence[int] = (10_080, 43_200, 86_400),
    quantiles: Sequence[float] = (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99),
    lags: int = 40,
    step: int = 1,
    plot: bool = True,
    verbose: bool = False,
) -> EDAResult:
    """Aggregate data-quality, moment, stationarity, ACF/PACF and rolling checks.

    This is the main first-pass diagnostics function for one or more time-series
    columns.  It combines static summary tables with optional rolling plots and
    preserves the original index for rolling outputs.

    Parameters
    ----------
    data, cols:
        Series/DataFrame and optional selected columns.
    rolling_windows:
        Trailing windows in rows for rolling mean, median, std, skew and
        kurtosis.
    quantiles:
        Quantiles included in the summary table.
    lags:
        Number of ACF/PACF lags.
    step:
        Sampling step for stationarity tests and ACF/PACF.  Use ``step=h`` for
        overlapping ``h``-bar returns.
    plot:
        Include figures for ACF/PACF and rolling diagnostics.
    verbose:
        If ``True``, warn when a column-level diagnostic fails.

    Returns
    -------
    EDAResult
        Keys: ``summary``, ``stationarity``, ``acf_pacf``, ``rolling``,
        ``figures`` and ``warnings``.

    Notes
    -----
    Rolling statistics are trailing.  Tests drop NaN/inf values.  Use
    ``step=h`` when columns are overlapping ``h``-bar returns and you want
    stationarity/ACF/PACF diagnostics on a non-overlapping subsample.

    Examples
    --------
    >>> res = data_diagnostics(df, cols=["return"], lags=24, step=5, plot=False)
    >>> res["summary"]
    """

    df = _as_frame(data, cols)
    summary = series_summary(df, quantiles=quantiles)
    stationarity = stationarity_summary(df, step=step)
    acf_out: dict[str, pd.DataFrame] = {}
    rolling_out: dict[str, pd.DataFrame] = {}
    figures: list[Figure] = []
    diagnostic_warnings: list[str] = []

    for col in df:
        try:
            ac = acf_pacf(df[col], lags=lags, step=step, plot=plot)
            acf_out[col] = ac["table"]
            if ac["figure"] is not None:
                figures.append(ac["figure"])
        except Exception as exc:
            diagnostic_warnings.append(f"ACF/PACF failed for {col}: {exc}")
            if verbose:
                warnings.warn(f"ACF/PACF failed for {col}: {exc}", RuntimeWarning)
        roll_cols = {}
        for w in rolling_windows:
            try:
                w, mp = _validate_window(w)
                x = _numeric_series(df[col], dropna=False, warn=False)
                r = x.rolling(w, min_periods=mp)
                roll_cols[f"mean_{w}"] = r.mean()
                roll_cols[f"median_{w}"] = r.median()
                roll_cols[f"std_{w}"] = r.std()
                roll_cols[f"skew_{w}"] = r.skew()
                roll_cols[f"kurtosis_{w}"] = r.kurt()
            except Exception as exc:
                diagnostic_warnings.append(f"Rolling diagnostics failed for {col}, window={w}: {exc}")
                if verbose:
                    warnings.warn(f"Rolling diagnostics failed for {col}, window={w}: {exc}", RuntimeWarning)
        rolling_out[col] = pd.DataFrame(roll_cols, index=df.index)
        if plot and roll_cols:
            fig, ax = plt.subplots(figsize=(9, 4.2))
            for i, name in enumerate([c for c in roll_cols if c.startswith("std_")][:3]):
                ax.plot(rolling_out[col].index, rolling_out[col][name], color=COLD_PALETTE[i], lw=1.5, label=name)
            style_axis(ax, ylabel=f"{col} rolling std")
            fig.tight_layout()
            figures.append(fig)
    return _result("Data diagnostics", {"summary": summary, "stationarity": stationarity, "acf_pacf": acf_out, "rolling": rolling_out, "figures": figures, "warnings": diagnostic_warnings})


def qq_plot(series: pd.Series, *, dist: str = "both", max_points: int = 50_000, plot: bool = True) -> EDAResult:
    """Create QQ diagnostics against normal and/or Student-t distributions.

    Parameters
    ----------
    dist:
        ``"normal"``, ``"student_t"`` or ``"both"``.
    max_points:
        Use the last ``max_points`` observations to keep plots responsive.

    Returns
    -------
    EDAResult
        ``{"fit": DataFrame, "figure": Figure | None}``.

    Notes
    -----
    NaN/inf values are dropped.  Large samples are trimmed to the last
    ``max_points`` observations for responsiveness.

    Examples
    --------
    >>> qq_plot(df["return"], dist="both", plot=False)["fit"]
    """

    x = _maybe_sample_for_tests(_numeric_series(series, dropna=True, min_obs=8, warn=False), max_test_size=max_points)
    dists = ["normal", "student_t"] if dist == "both" else [dist]
    if any(d not in {"normal", "student_t"} for d in dists):
        raise ValueError("dist must be 'normal', 'student_t' or 'both'.")
    rows = []
    fig = None
    axes: Sequence[Axes] = []
    if plot:
        fig, axes_arr = plt.subplots(1, len(dists), figsize=(5 * len(dists), 4.5))
        axes = np.atleast_1d(axes_arr)
    for i, d in enumerate(dists):
        if d == "normal":
            params = (x.mean(), x.std())
            if plot:
                stats.probplot(x, dist=stats.norm, sparams=params, plot=axes[i])
            rows.append({"distribution": "normal", "param_1": params[0], "param_2": params[1]})
        else:
            df, loc, scale = stats.t.fit(x)
            if plot:
                stats.probplot(x, dist=stats.t, sparams=(df, loc, scale), plot=axes[i])
            rows.append({"distribution": "student_t", "df": df, "loc": loc, "scale": scale})
        if plot:
            style_axis(axes[i], legend=False)
            axes[i].set_title("")
    if fig is not None:
        fig.tight_layout()
    return _result("QQ plot", {"fit": pd.DataFrame(rows), "figure": fig})


def density_plot(series: pd.Series, *, normal_overlay: bool = True, points: int = 600, plot: bool = True) -> EDAResult:
    """Estimate empirical density with an optional normal overlay.

    Parameters
    ----------
    series : pandas.Series
        Numeric series.  NaN/inf values are dropped.
    normal_overlay : bool, default True
        Add a fitted normal PDF to the output table and plot.
    points : int, default 600
        Grid size between the 0.1% and 99.9% sample quantiles.
    plot : bool, default True
        Include a minimalist density figure.

    Returns
    -------
    EDAResult
        Keys: ``density`` table, ``warning`` and optional ``figure``.

    Examples
    --------
    >>> density_plot(df["return"], plot=False)["density"].head()
    """

    x = _numeric_series(series, dropna=True, min_obs=5, warn=False)
    grid = np.linspace(x.quantile(0.001), x.quantile(0.999), points)
    try:
        density = stats.gaussian_kde(x)(grid)
        warning = ""
    except Exception as exc:
        density = np.full_like(grid, np.nan, dtype=float)
        warning = str(exc)
    table = pd.DataFrame({"x": grid, "density": density})
    if normal_overlay:
        table["normal_density"] = stats.norm.pdf(grid, x.mean(), x.std())
    fig = None
    if plot:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(table["x"], table["density"], color=COLD_PALETTE[0], lw=2.2, label="Empirical")
        if normal_overlay:
            ax.plot(table["x"], table["normal_density"], color=COLD_PALETTE[2], lw=1.8, ls="--", label="Normal")
        style_axis(ax)
        fig.tight_layout()
    return _result("Density plot", {"density": table, "warning": warning, "figure": fig})


def hill_estimator(series: pd.Series, *, q: float = 0.95, tail: str = "abs") -> pd.DataFrame:
    """Estimate power-law tail index alpha with the Hill estimator.

    This is the module's tail-index function; if you expected a function named
    ``hill_tail_index`` or ``tail_index``, use ``hill_estimator``.

    Parameters
    ----------
    series : pandas.Series
        Numeric series, typically returns.  NaN and infinite values are dropped.
    q:
        Tail threshold quantile, e.g. ``0.95``.
    tail:
        ``"abs"``, ``"right"`` or ``"left"``.  Left tail is estimated on
        ``-x``.

    Returns
    -------
    pandas.DataFrame
        One row with threshold, tail size and ``alpha``.  Higher alpha means
        thinner tails.

    Notes
    -----
    Hill estimates are sensitive to the threshold quantile and should be read as
    exploratory tail diagnostics, not as a stable structural parameter.

    Examples
    --------
    >>> hill_estimator(df["return"], q=0.99, tail="abs")
    """

    if not 0 < q < 1:
        raise ValueError("q must be in (0, 1).")
    x = _numeric_series(series, dropna=True, min_obs=10, warn=False)
    if tail == "abs":
        z = x.abs()
    elif tail == "right":
        z = x
    elif tail == "left":
        z = -x
    else:
        raise ValueError("tail must be 'abs', 'right' or 'left'.")
    z = z[z > 0].sort_values()
    u = z.quantile(q)
    tail_values = z[z > u]
    alpha = np.nan
    if len(tail_values) >= 2 and u > 0:
        alpha = 1 / np.mean(np.log(tail_values / u))
    return pd.DataFrame([{"tail": tail, "n": len(z), "q": q, "threshold": u, "tail_n": len(tail_values), "alpha": alpha}])


def evt_gpd_fit(series: pd.Series, *, q: float = 0.95, tail: str = "abs", plot: bool = False) -> EDAResult:
    """Fit a Generalized Pareto distribution to threshold exceedances.

    This is the module's GPD tail-fit function; if you expected a function
    named ``gpd_tail_fit``, use ``evt_gpd_fit``.

    Parameters
    ----------
    series : pandas.Series
        Numeric series, typically returns.
    q : float, default 0.95
        Threshold quantile used to define exceedances.
    tail : {"abs", "right", "left"}, default "abs"
        Tail to fit.  Left tail is fit on ``-series``.
    plot : bool, default False
        Include a histogram/PDF overlay figure when enough exceedances exist.

    Returns
    -------
    EDAResult
        Keys: ``fit`` with Hill/GPD parameters, ``excess`` with threshold
        exceedances and optional ``figure``.

    Notes
    -----
    Returns fit parameters ``xi`` and ``beta`` plus the exceedance table.  If
    the tail is too small, parameters are ``NaN`` and a warning string explains
    why.

    Examples
    --------
    >>> evt_gpd_fit(df["return"], q=0.99, tail="left", plot=False)["fit"]
    """

    h = hill_estimator(series, q=q, tail=tail)
    x = _numeric_series(series, dropna=True, min_obs=10, warn=False)
    z = x.abs() if tail == "abs" else x if tail == "right" else -x
    u = float(h["threshold"].iloc[0])
    excess = (z[z > u] - u).dropna()
    warning = ""
    xi = beta = np.nan
    if len(excess) >= 5:
        try:
            xi, _, beta = genpareto.fit(excess, floc=0)
        except Exception as exc:
            warning = str(exc)
    else:
        warning = "Too few exceedances for GPD fit."
    fit = h.assign(gpd_xi=xi, gpd_beta=beta, warning=warning)
    fig = None
    if plot and len(excess) >= 5:
        fig, ax = plt.subplots(figsize=(7, 4.3))
        xs = np.linspace(0, excess.quantile(0.995), 300)
        ax.hist(excess, bins=50, density=True, color=COLD_PALETTE[0], alpha=0.35, label="Excess")
        ax.plot(xs, genpareto.pdf(xs, xi, loc=0, scale=beta), color=COLD_PALETTE[2], lw=2, label="GPD")
        style_axis(ax)
        fig.tight_layout()
    return _result("EVT/GPD fit", {"fit": fit, "excess": excess.rename("excess"), "figure": fig})


def normality_tests(
    series: pd.Series,
    *,
    step: int = 1,
    max_test_size: int | None = 100_000,
    shapiro_size: int = 5_000,
    random_state: int = 42,
) -> pd.DataFrame:
    """Run JB, Shapiro-Wilk, Anderson-Darling and KS normality diagnostics.

    Parameters
    ----------
    series : pandas.Series
        Numeric series.  NaN and infinite values are dropped.
    step : int, default 1
        Optional subsampling step after cleaning.  Use ``step=h`` for
        overlapping ``h``-bar returns.
    max_test_size : int or None, default 100000
        Maximum number of observations used after stepping.  ``None`` disables
        trimming.
    shapiro_size : int, default 5000
        Maximum sample size used by Shapiro-Wilk.
    random_state : int, default 42
        Seed for Shapiro subsampling.

    Returns
    -------
    pandas.DataFrame
        One row per test with ``test``, ``n_test``, statistic, p-value and
        ``warning`` columns.

    For Shapiro-Wilk, SciPy p-values are not reliable for very large samples.
    If the cleaned sample is larger than ``shapiro_size``, this function uses a
    reproducible random subsample and records that fact in the warning column.
    The KS test uses estimated mean/std, so it is a practical diagnostic rather
    than a strict Lilliefors-corrected test.

    Notes
    -----
    Constant or too-short samples return warning rows.  Use ``step=h`` for
    overlapping ``h``-bar returns when independence assumptions are a concern.

    Examples
    --------
    >>> normality_tests(df["return"], step=5)
    """

    x = pd.to_numeric(_as_series(series), errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(x) < 8 or x.nunique(dropna=True) < 2:
        warning = (
            f"Normality tests need at least 8 finite observations; got {len(x)}."
            if len(x) < 8
            else "Normality tests are undefined for a constant series."
        )
        out = pd.DataFrame([_result_row(t, warning=warning) for t in ["Jarque-Bera", "Shapiro-Wilk", "Anderson-Darling", "Kolmogorov-Smirnov"]])
        out.insert(1, "n_test", len(x))
        return out
    x = _maybe_sample_for_tests(x, step=step, max_test_size=max_test_size)
    if len(x) < 8 or x.nunique(dropna=True) < 2:
        warning = (
            f"Normality tests need at least 8 finite observations after step={step}; got {len(x)}."
            if len(x) < 8
            else f"Normality tests are undefined for a constant series after step={step}."
        )
        out = pd.DataFrame([_result_row(t, warning=warning) for t in ["Jarque-Bera", "Shapiro-Wilk", "Anderson-Darling", "Kolmogorov-Smirnov"]])
        out.insert(1, "n_test", len(x))
        return out
    rows: list[dict[str, Any]] = []
    try:
        stat, pvalue = stats.jarque_bera(x)
        rows.append(_result_row("Jarque-Bera", stat, pvalue))
    except Exception as exc:
        rows.append(_result_row("Jarque-Bera", warning=str(exc)))
    try:
        xs = x.sample(shapiro_size, random_state=random_state) if len(x) > shapiro_size else x
        stat, pvalue = stats.shapiro(xs)
        warning = f"Subsampled {shapiro_size} of {len(x)} observations." if len(x) > shapiro_size else ""
        rows.append(_result_row("Shapiro-Wilk", stat, pvalue, warning))
    except Exception as exc:
        rows.append(_result_row("Shapiro-Wilk", warning=str(exc)))
    try:
        ad = stats.anderson(x, dist="norm")
        crit_5 = ad.critical_values[list(ad.significance_level).index(5.0)] if 5.0 in ad.significance_level else np.nan
        rows.append({**_result_row("Anderson-Darling", ad.statistic, np.nan), "crit_5pct": crit_5})
    except Exception as exc:
        rows.append(_result_row("Anderson-Darling", warning=str(exc)))
    try:
        mu, sigma = x.mean(), x.std()
        if not np.isfinite(sigma) or sigma <= 0:
            raise ValueError("sample standard deviation is zero or non-finite")
        stat, pvalue = stats.kstest(x, lambda z: stats.norm.cdf(z, loc=mu, scale=sigma))
        rows.append(_result_row("Kolmogorov-Smirnov", stat, pvalue, "Mean/std estimated from sample."))
    except Exception as exc:
        rows.append(_result_row("Kolmogorov-Smirnov", warning=str(exc)))
    out = pd.DataFrame(rows)
    out.insert(1, "n_test", len(x))
    return out


def arch_lm_test(series: pd.Series, *, nlags: int = 10, step: int = 1, max_test_size: int | None = 100_000) -> pd.DataFrame:
    """Run Engle's ARCH LM test for volatility clustering.

    Parameters
    ----------
    series : pandas.Series
        Numeric return-like series.
    nlags : int, default 10
        Number of ARCH lags.
    step : int, default 1
        Optional subsampling step after cleaning.  Use ``step=h`` for
        overlapping ``h``-bar returns.
    max_test_size : int or None, default 100000
        Maximum observations used after stepping.

    Returns
    -------
    pandas.DataFrame
        One-row table with LM/F statistics, p-values, sample size, lags and a
        ``warning`` string when the test is not feasible.

    Notes
    -----
    Constant or too-short samples return a warning row instead of raising.

    Examples
    --------
    >>> arch_lm_test(df["return"], nlags=20, step=5)
    """

    if nlags < 1:
        raise ValueError("nlags must be positive.")
    x = pd.to_numeric(_as_series(series), errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(x) < nlags + 5 or x.nunique(dropna=True) < 2:
        warning = (
            f"ARCH LM needs at least {nlags + 5} finite observations; got {len(x)}."
            if len(x) < nlags + 5
            else "ARCH LM is undefined for a constant series."
        )
        return pd.DataFrame([{"n_test": len(x), "lags": nlags, "LM_stat": np.nan, "LM_pvalue": np.nan, "F_stat": np.nan, "F_pvalue": np.nan, "warning": warning}])
    x = _maybe_sample_for_tests(x, step=step, max_test_size=max_test_size)
    if len(x) < nlags + 5 or x.nunique(dropna=True) < 2:
        warning = (
            f"ARCH LM needs at least {nlags + 5} finite observations after step={step}; got {len(x)}."
            if len(x) < nlags + 5
            else f"ARCH LM is undefined for a constant series after step={step}."
        )
        return pd.DataFrame([{"n_test": len(x), "lags": nlags, "LM_stat": np.nan, "LM_pvalue": np.nan, "F_stat": np.nan, "F_pvalue": np.nan, "warning": warning}])
    try:
        lm_stat, lm_pval, f_stat, f_pval = het_arch(x, nlags=nlags)
        row = {"n_test": len(x), "lags": nlags, "LM_stat": lm_stat, "LM_pvalue": lm_pval, "F_stat": f_stat, "F_pvalue": f_pval, "warning": ""}
    except Exception as exc:
        row = {"n_test": len(x), "lags": nlags, "LM_stat": np.nan, "LM_pvalue": np.nan, "F_stat": np.nan, "F_pvalue": np.nan, "warning": str(exc)}
    return pd.DataFrame([row])


def class_balance(target: pd.Series, *, normalize: bool = True, max_classes: int = 50) -> pd.DataFrame:
    """Return class counts and optional frequencies for discrete targets.

    Parameters
    ----------
    target : pandas.Series
        Categorical or discrete target.
    normalize : bool, default True
        Include relative frequencies.
    max_classes : int, default 50
        Warn if more unique classes are present.

    Returns
    -------
    pandas.DataFrame
        Columns: ``class``, ``count`` and optionally ``frequency``.
    """

    y = _as_series(target, name="target").replace([np.inf, -np.inf], np.nan).dropna()
    if y.nunique() > max_classes:
        warnings.warn(f"Target has {y.nunique()} unique values; class balance may not be meaningful.", RuntimeWarning)
    counts = y.value_counts(dropna=False).sort_index()
    out = counts.rename_axis("class").reset_index(name="count")
    if normalize:
        out["frequency"] = out["count"] / out["count"].sum()
    return out


def distribution_report(
    series: pd.Series,
    *,
    q: float = 0.95,
    tail: str = "abs",
    arch_lags: int = 10,
    step: int = 1,
    plot: bool = True,
) -> EDAResult:
    """Aggregate distribution, tail, normality and ARCH diagnostics.

    Parameters
    ----------
    series : pandas.Series
        Numeric series, usually returns.
    q : float, default 0.95
        Tail threshold quantile for Hill and GPD diagnostics.
    tail : {"abs", "right", "left"}, default "abs"
        Tail used by extreme-value diagnostics.
    arch_lags : int, default 10
        Lags for Engle ARCH LM test.
    step : int, default 1
        Optional test subsampling step after cleaning.
    plot : bool, default True
        Include density, QQ and optional GPD figures.

    Returns
    -------
    EDAResult
        Keys: ``summary``, ``density``, ``qq``, ``hill``, ``evt``,
        ``normality``, ``arch_lm`` and optional ``class_balance``.

    Notes
    -----
    Use ``step=h`` when normality and ARCH tests are run on overlapping
    ``h``-bar returns and you want a non-overlapping test subsample.

    Examples
    --------
    >>> distribution_report(df["return"], q=0.99, step=10, plot=False)["normality"]
    """

    return _result("Distribution report", {
        "summary": series_summary(series),
        "density": density_plot(series, plot=plot),
        "qq": qq_plot(series, plot=plot),
        "hill": hill_estimator(series, q=q, tail=tail),
        "evt": evt_gpd_fit(series, q=q, tail=tail, plot=plot),
        "normality": normality_tests(series, step=step),
        "arch_lm": arch_lm_test(series, nlags=arch_lags, step=step),
        "class_balance": class_balance(series, normalize=True) if _as_series(series).nunique(dropna=True) <= 20 else None,
    })


def periodogram(series: pd.Series, *, fs: float = 1.0, detrend: str = "constant", plot: bool = True) -> EDAResult:
    """Estimate spectral density with scipy's periodogram.

    ``fs`` is samples per unit time.  For one-minute bars and frequency in
    cycles per day, use ``fs=1440``.

    Returns
    -------
    EDAResult
        Keys: ``table`` with frequency/power and optional ``figure``.

    Examples
    --------
    >>> periodogram(df["return"], fs=1440, plot=False)["table"].head()
    """

    x = _numeric_series(series, dropna=True, min_obs=8, warn=False)
    freq, power = signal.periodogram(x.to_numpy(dtype=float), fs=fs, detrend=detrend)
    table = pd.DataFrame({"frequency": freq, "power": power})
    fig = None
    if plot:
        fig, ax = plt.subplots(figsize=(8, 4.2))
        ax.plot(table["frequency"], table["power"], color=COLD_PALETTE[1], lw=1.5)
        style_axis(ax, xlabel="Frequency", ylabel="Power", legend=False)
        fig.tight_layout()
    return _result("Periodogram", {"table": table, "figure": fig})


def _index_to_float_time(index: pd.Index) -> np.ndarray:
    if isinstance(index, pd.DatetimeIndex):
        return (index - index[0]).total_seconds().to_numpy(dtype=float)
    return np.arange(len(index), dtype=float)


def lomb_scargle_periodogram(
    series: pd.Series,
    *,
    min_frequency: float | None = None,
    max_frequency: float | None = None,
    n_freq: int = 1_000,
    plot: bool = True,
) -> EDAResult:
    """Lomb-Scargle periodogram for irregular timestamps or missing observations.

    Frequencies are cycles per second for ``DatetimeIndex`` and cycles per row
    for non-datetime indexes.

    Returns
    -------
    EDAResult
        Keys: ``table`` with frequency/power and optional ``figure``.

    Examples
    --------
    >>> lomb_scargle_periodogram(irregular_returns, plot=False)["table"].head()
    """

    x = _numeric_series(series, dropna=True, min_obs=8, warn=False)
    t = _index_to_float_time(x.index)
    span = t.max() - t.min()
    if span <= 0:
        raise ValueError("Time span must be positive for Lomb-Scargle.")
    min_frequency = min_frequency or 1 / span
    max_frequency = max_frequency or len(x) / (2 * span)
    freqs = np.linspace(min_frequency, max_frequency, n_freq)
    power = signal.lombscargle(t, x.to_numpy(dtype=float) - x.mean(), 2 * np.pi * freqs, normalize=True)
    table = pd.DataFrame({"frequency": freqs, "power": power})
    fig = None
    if plot:
        fig, ax = plt.subplots(figsize=(8, 4.2))
        ax.plot(table["frequency"], table["power"], color=COLD_PALETTE[1], lw=1.5)
        style_axis(ax, xlabel="Frequency", ylabel="Power", legend=False)
        fig.tight_layout()
    return _result("Lomb-Scargle periodogram", {"table": table, "figure": fig})


def _infer_period_from_index(index: pd.Index) -> int | None:
    if not isinstance(index, pd.DatetimeIndex) or len(index) < 3:
        return None
    diffs = index.to_series().diff().dropna().dt.total_seconds()
    if diffs.empty or diffs.median() <= 0:
        return None
    daily = int(round(86_400 / diffs.median()))
    return daily if daily >= 2 else None


def stl_decomposition(series: pd.Series, *, period: int | None = None, robust: bool = True, plot: bool = True) -> EDAResult:
    """Decompose a series into observed, trend, seasonal and residual components.

    Parameters
    ----------
    series : pandas.Series
        Numeric time series.
    period : int, optional
        Seasonal period in rows.  Inferred from ``DatetimeIndex`` when possible.
    robust : bool, default True
        Use robust STL fitting.
    plot : bool, default True
        Include component plots.

    Returns
    -------
    EDAResult
        Keys: ``components``, ``period`` and optional ``figure``.
    """

    x = _numeric_series(series, dropna=True, min_obs=20, warn=False)
    period = period or _infer_period_from_index(x.index)
    if period is None or period < 2:
        raise ValueError("period is required unless it can be inferred from a DatetimeIndex.")
    res = STL(x, period=period, robust=robust).fit()
    components = pd.DataFrame({"observed": x, "trend": res.trend, "seasonal": res.seasonal, "resid": res.resid}, index=x.index)
    fig = None
    if plot:
        fig, axes = plt.subplots(4, 1, figsize=(9, 7), sharex=True)
        for ax, col in zip(axes, components.columns):
            ax.plot(components.index, components[col], color=COLD_PALETTE[1], lw=1)
            style_axis(ax, ylabel=col, legend=False)
        fig.tight_layout()
    return _result("STL decomposition", {"components": components, "period": period, "figure": fig})


def hurst_exponent(
    series: pd.Series,
    *,
    method: str = "rs",
    min_window: int = 16,
    max_window: int | None = None,
    n_windows: int = 20,
) -> pd.DataFrame:
    """Estimate the Hurst exponent using R/S or DFA scaling.

    Parameters
    ----------
    series : pandas.Series
        Numeric series.
    method : {"rs", "dfa"}, default "rs"
        Scaling method.
    min_window, max_window, n_windows : int
        Window grid controls.

    Returns
    -------
    pandas.DataFrame
        One row with method, estimated Hurst exponent and fit metadata.

    ``H > 0.5`` suggests persistence, ``H < 0.5`` anti-persistence, and
    ``H ~= 0.5`` random-walk-like scaling.
    """

    x = _numeric_series(series, dropna=True, min_obs=max(64, min_window * 4), warn=False).to_numpy(dtype=float)
    max_window = max_window or len(x) // 4
    windows = np.unique(np.logspace(np.log10(min_window), np.log10(max_window), n_windows).astype(int))
    vals = []
    for w in windows:
        if w < 4 or w >= len(x):
            continue
        chunks = len(x) // w
        arr = x[: chunks * w].reshape(chunks, w)
        if method == "rs":
            z = arr - arr.mean(axis=1, keepdims=True)
            y = z.cumsum(axis=1)
            r = y.max(axis=1) - y.min(axis=1)
            s = arr.std(axis=1, ddof=1)
            value = np.nanmean(r[s > 0] / s[s > 0])
        elif method == "dfa":
            y = (arr - arr.mean(axis=1, keepdims=True)).cumsum(axis=1)
            t = np.arange(w)
            rms = []
            for row in y:
                coef = np.polyfit(t, row, 1)
                rms.append(np.sqrt(np.mean((row - np.polyval(coef, t)) ** 2)))
            value = np.nanmean(rms)
        else:
            raise ValueError("method must be 'rs' or 'dfa'.")
        if np.isfinite(value) and value > 0:
            vals.append((w, value))
    if len(vals) < 3:
        raise ValueError("Not enough valid windows for Hurst estimate.")
    table = pd.DataFrame(vals, columns=["window", "scale_stat"])
    slope, intercept = np.polyfit(np.log(table["window"]), np.log(table["scale_stat"]), 1)
    return pd.DataFrame([{"method": method, "hurst": slope, "intercept": intercept, "n_windows": len(table)}])


def cusum_test(series: pd.Series) -> pd.DataFrame:
    """Run CUSUM test for parameter instability around the sample mean.

    Parameters
    ----------
    series : pandas.Series
        Numeric series.

    Returns
    -------
    pandas.DataFrame
        One-row table with statistic, p-value, critical values and warning.
    """

    x = _numeric_series(series, dropna=True, min_obs=10, warn=False)
    try:
        stat, pvalue, crit = breaks_cusumolsresid(x - x.mean(), ddof=0)
        row = {"statistic": stat, "pvalue": pvalue, "crit": crit, "warning": ""}
    except Exception as exc:
        row = {"statistic": np.nan, "pvalue": np.nan, "crit": np.nan, "warning": str(exc)}
    return pd.DataFrame([row])


def structural_breaks(
    series: pd.Series,
    *,
    n_bkps: int | None = None,
    penalty: float | None = None,
    model: str = "rbf",
    min_size: int = 20,
    max_points: int = 20_000,
    verbose: bool = False,
) -> EDAResult:
    """Detect multiple structural breaks with ``ruptures`` when available.

    This is a practical Bai-Perron-style alternative rather than a strict
    econometric Bai-Perron implementation.  If ``ruptures`` is unavailable, the
    function returns an empty break table and a warning.

    Parameters
    ----------
    n_bkps:
        If provided, use binary segmentation and return this many breaks.
    penalty:
        PELT penalty.  If neither ``n_bkps`` nor ``penalty`` is provided, a
        conservative log-size penalty is used.
    max_points:
        Long series are downsampled before break detection, then break positions
        are mapped back to the original index.
    """

    x = _numeric_series(series, dropna=True, min_obs=max(50, min_size * 3), warn=False)
    step = max(1, int(math.ceil(len(x) / max_points)))
    xs = x.iloc[::step]
    warning = ""
    try:
        import ruptures as rpt

        arr = xs.to_numpy(dtype=float).reshape(-1, 1)
        if n_bkps is not None:
            bkps = rpt.Binseg(model=model, min_size=min_size).fit(arr).predict(n_bkps=int(n_bkps))
        else:
            pen = penalty if penalty is not None else np.log(len(xs)) * np.nanvar(arr)
            bkps = rpt.Pelt(model=model, min_size=min_size).fit(arr).predict(pen=pen)
        positions = [min((b - 1) * step, len(x) - 1) for b in bkps if b < len(xs)]
        table = pd.DataFrame({"position": positions, "timestamp": [x.index[p] for p in positions]})
    except Exception as exc:
        warning = str(exc)
        table = pd.DataFrame(columns=["position", "timestamp"])
        if verbose:
            warnings.warn(f"structural_breaks failed: {exc}", RuntimeWarning)
    return _result("Structural breaks", {"breaks": table, "sample_step": step, "warning": warning})


def seasonality_report(
    series: pd.Series,
    *,
    period: int | None = None,
    fs: float = 1.0,
    plot: bool = True,
    verbose: bool = False,
) -> EDAResult:
    """Aggregate seasonality, persistence and structural-break diagnostics.

    Parameters
    ----------
    series : pandas.Series
        Numeric time series.  A ``DatetimeIndex`` improves period inference and
        Lomb-Scargle interpretation.
    period : int, optional
        Seasonal period in rows for STL.  If omitted, a daily period is inferred
        from a regular ``DatetimeIndex`` where possible.
    fs : float, default 1.0
        Sampling frequency for the regular periodogram.
    plot : bool, default True
        Include diagnostic figures where available.
    verbose : bool, default False
        Emit warnings when optional diagnostics fail.

    Returns
    -------
    EDAResult
        Keys: ``periodogram``, ``lomb_scargle``, ``hurst_rs``, ``cusum``,
        ``structural_breaks`` and ``stl``.

    Notes
    -----
    ``structural_breaks`` uses ``ruptures`` as a practical Bai-Perron-style
    alternative.  STL may return a warning entry instead of components if a
    period cannot be inferred or supplied.

    Examples
    --------
    >>> seasonality_report(df["return"], period=1440, plot=False)
    """

    out: dict[str, Any] = {
        "periodogram": periodogram(series, fs=fs, plot=plot),
        "lomb_scargle": lomb_scargle_periodogram(series, plot=plot),
        "hurst_rs": hurst_exponent(series, method="rs"),
        "cusum": cusum_test(series),
        "structural_breaks": structural_breaks(series, verbose=verbose),
    }
    try:
        out["stl"] = stl_decomposition(series, period=period, plot=plot)
    except Exception as exc:
        out["stl"] = {"components": None, "period": period, "figure": None, "warning": str(exc)}
        if verbose:
            warnings.warn(f"STL failed: {exc}", RuntimeWarning)
    return _result("Seasonality report", out)


def feature_target_correlation(
    features: pd.Series | pd.DataFrame,
    target: pd.Series,
    *,
    method: str = "spearman",
) -> pd.DataFrame:
    """Compute correlation between each feature and an aligned target.

    Parameters
    ----------
    features : pandas.Series or pandas.DataFrame
        Feature values observed at time ``t``.
    target : pandas.Series
        Target aligned to the same timestamp.
    method : {"pearson", "spearman", "kendall"}, default "spearman"
        Correlation method.

    Returns
    -------
    pandas.DataFrame
        Columns: ``feature``, ``correlation``, ``method`` and ``n``.

    Notes
    -----
    The function does not shift inputs.  For forward returns, create the target
    with :func:`forward_return` before calling.
    """

    df = _align_xy(features, target)
    y = df.pop("__target__")
    rows = [{"feature": col, "correlation": df[col].corr(y, method=method), "method": method, "n": len(df)} for col in df]
    return pd.DataFrame(rows)


def ic_summary(
    features: pd.Series | pd.DataFrame,
    target: pd.Series,
    *,
    method: str = "spearman",
    hac_lags: int | str | None = "auto",
    min_hac_lags: int = 0,
) -> pd.DataFrame:
    """Information coefficient summary with HAC/Newey-West t-statistics.

    Parameters
    ----------
    features:
        Feature series/DataFrame observed at decision time ``t``.
    target:
        Target already aligned at ``t``.  For forward returns, use
        :func:`forward_return`.
    method:
        ``"pearson"``, ``"spearman"`` or ``"kendall"`` for the reported IC.
        HAC t-statistics use standardized Pearson products; for Spearman, both
        sides are first rank-transformed.
    hac_lags:
        Newey-West lags.  ``"auto"`` uses a simple sample-size rule.
    min_hac_lags:
        Lower bound for Newey-West lags.  For overlapping ``h``-bar forward
        returns, use at least ``h - 1``.

    Returns
    -------
    pandas.DataFrame
        Columns: feature, n, ic, std_ic, hac_se, t_stat_hac, pvalue_hac,
        hac_lags and method.

    Notes
    -----
    ``target`` must already be aligned to the feature timestamp.  For
    overlapping forward returns, set ``min_hac_lags=h-1`` or use
    :func:`feature_target_report`, which does this automatically for internally
    created targets.

    Examples
    --------
    >>> y = forward_return(df["close"], horizon=5)
    >>> ic_summary(df[["volume", "turnover"]], y, min_hac_lags=4)
    """

    df = _align_xy(features, target)
    y = df["__target__"]
    xdf = df.drop(columns="__target__")
    rows = []
    for col in xdf:
        x, yy = xdf[col], y
        ic = x.corr(yy, method=method)
        if method == "spearman":
            xx, yy2 = x.rank(), yy.rank()
        elif method == "kendall":
            xx, yy2 = x.rank(), yy.rank()
        else:
            xx, yy2 = x, yy
        contrib = (_zscore(xx) * _zscore(yy2)).dropna()
        se, lags = _newey_west_se_mean(contrib, hac_lags, min_lags=min_hac_lags)
        mean_ic = contrib.mean()
        t_stat = mean_ic / se if se > 0 else np.nan
        pvalue = 2 * (1 - stats.norm.cdf(abs(t_stat))) if np.isfinite(t_stat) else np.nan
        rows.append(
            {
                "feature": col,
                "n": len(contrib),
                "ic": ic,
                "std_ic": contrib.std(ddof=1),
                "hac_se": se,
                "t_stat_hac": t_stat,
                "pvalue_hac": pvalue,
                "hac_lags": lags,
                "method": method,
            }
        )
    return pd.DataFrame(rows).sort_values("ic", key=lambda s: s.abs(), ascending=False).reset_index(drop=True)


def cumulative_ic(
    features: pd.Series | pd.DataFrame,
    target: pd.Series,
    *,
    method: str = "spearman",
    cumulative: str = "sum",
    plot: bool = True,
) -> EDAResult:
    """Cumulative IC contribution plot and table.

    Parameters
    ----------
    features:
        Feature series/DataFrame observed at time ``t``.
    target:
        Target aligned at time ``t``.
    method:
        ``"pearson"`` or ``"spearman"``.  Spearman uses rank-transformed
        features and target before standardization.
    cumulative:
        ``"sum"`` for cumulative standardized IC contribution or ``"mean"`` for
        expanding average contribution.
    plot:
        If ``True``, return a minimalist matplotlib figure.

    Returns
    -------
    dict
        ``{"table": DataFrame, "figure": Figure | None}`` where table columns
        are feature names.
    """

    if cumulative not in {"sum", "mean"}:
        raise ValueError("cumulative must be 'sum' or 'mean'.")
    df = _align_xy(features, target)
    y = df["__target__"]
    xdf = df.drop(columns="__target__")
    if method == "spearman":
        y = y.rank()
        xdf = xdf.rank()
    elif method != "pearson":
        raise ValueError("cumulative_ic supports method='pearson' or 'spearman'.")
    contrib = pd.DataFrame({_col: _zscore(xdf[_col]) * _zscore(y) for _col in xdf}, index=xdf.index)
    table = contrib.cumsum() if cumulative == "sum" else contrib.expanding().mean()
    fig = None
    if plot:
        fig, ax = plt.subplots(figsize=(9, 4.2))
        for i, col in enumerate(table.columns[:8]):
            ax.plot(table.index, table[col], color=COLD_PALETTE[i % len(COLD_PALETTE)], lw=1.4, label=col)
        style_axis(ax, ylabel=f"Cumulative IC ({cumulative})")
        fig.tight_layout()
    return _result("Cumulative IC", {"table": table, "figure": fig})


def cross_sectional_ic(
    features: pd.DataFrame,
    target: pd.Series,
    *,
    time_level: int | str = 0,
    method: str = "spearman",
) -> EDAResult:
    """Cross-sectional IC for MultiIndex data grouped by time level.

    ``features`` and ``target`` must share a MultiIndex such as
    ``(timestamp, symbol)``.  The function computes one IC per timestamp and
    feature, then summarizes mean, std and t-stat across timestamps.
    """

    if not isinstance(features.index, pd.MultiIndex) or not isinstance(target.index, pd.MultiIndex):
        raise ValueError("cross_sectional_ic requires MultiIndex features and target.")
    df = _align_xy(features, target)
    y_name = "__target__"
    by_time = []
    for t, g in df.groupby(level=time_level):
        if len(g) < 3:
            continue
        y = g[y_name]
        for col in features.columns:
            by_time.append({"time": t, "feature": col, "ic": g[col].corr(y, method=method), "n": len(g)})
    ic_ts = pd.DataFrame(by_time)
    if ic_ts.empty:
        return _result("Cross-sectional IC", {"ic_by_time": ic_ts, "summary": pd.DataFrame()})
    summary = (
        ic_ts.groupby("feature")["ic"]
        .agg(["count", "mean", "std"])
        .rename(columns={"count": "n_periods", "mean": "ic", "std": "std_ic"})
        .reset_index()
    )
    summary["t_stat"] = summary["ic"] / (summary["std_ic"] / np.sqrt(summary["n_periods"]))
    return _result("Cross-sectional IC", {"ic_by_time": ic_ts, "summary": summary.sort_values("ic", key=lambda s: s.abs(), ascending=False)})


def rolling_ic(
    features: pd.Series | pd.DataFrame,
    target: pd.Series,
    *,
    window: int,
    min_periods: int | None = None,
    method: str = "spearman",
) -> pd.DataFrame:
    """Trailing rolling IC for each feature.

    ``method`` can be ``"pearson"`` or ``"spearman"``.  For Spearman, both
    series are rank-transformed before rolling correlation.  This is stable and
    fast for long time series, but it is a rank-transformed rolling Pearson
    correlation rather than a full re-ranking inside every window.
    """

    window, min_periods = _validate_window(window, min_periods)
    df = _align_xy(features, target)
    y = df["__target__"]
    xdf = df.drop(columns="__target__")
    if method == "spearman":
        y = y.rank()
        xdf = xdf.rank()
    elif method != "pearson":
        raise ValueError("rolling_ic supports method='pearson' or 'spearman'. Use rolling_ic_stats for Kendall-style window summaries.")
    return pd.DataFrame({col: xdf[col].rolling(window, min_periods=min_periods).corr(y) for col in xdf}, index=xdf.index)


def rolling_ic_stats(
    features: pd.Series | pd.DataFrame,
    target: pd.Series,
    *,
    window: int,
    min_periods: int | None = None,
    method: str = "spearman",
    hac_lags: int | str | None = "auto",
    min_hac_lags: int = 0,
    step: int = 1,
) -> pd.DataFrame:
    """Trailing rolling IC, std(IC) and HAC/Newey-West t-statistics.

    Parameters
    ----------
    features, target:
        Aligned feature matrix and target series.
    window, min_periods:
        Trailing window size and minimum observations.
    method:
        Correlation method passed to :func:`ic_summary`.
    hac_lags:
        HAC lags inside each rolling window.  ``"auto"`` uses the sample-size
        rule inside each window.
    min_hac_lags:
        Lower bound for HAC lags.  Use ``h - 1`` for overlapping ``h``-bar
        forward returns.
    step:
        Evaluate every ``step`` rows to reduce cost on long minute-level data.

    Returns
    -------
    pandas.DataFrame
        Long table with timestamp, feature, n, ic, std_ic, hac_se,
        t_stat_hac and pvalue_hac.

    Notes
    -----
    Each row uses only observations inside the trailing window ending at
    ``timestamp``.  Set ``min_hac_lags=h-1`` for overlapping ``h``-bar returns.

    Examples
    --------
    >>> y = forward_return(df["close"], horizon=10)
    >>> rolling_ic_stats(df[["x1", "x2"]], y, window=500, min_hac_lags=9, step=50)
    """

    if step < 1:
        raise ValueError("step must be >= 1.")
    window, min_periods = _validate_window(window, min_periods)
    df = _align_xy(features, target, min_obs=min_periods)
    rows = []
    for end in range(min_periods, len(df) + 1, step):
        chunk = df.iloc[max(0, end - window) : end]
        if len(chunk) < min_periods:
            continue
        res = ic_summary(chunk.drop(columns="__target__"), chunk["__target__"], method=method, hac_lags=hac_lags, min_hac_lags=min_hac_lags)
        for _, r in res.iterrows():
            rows.append({"timestamp": df.index[end - 1], **r.to_dict()})
    return pd.DataFrame(rows)


def feature_quantile_stats(
    features: pd.Series | pd.DataFrame,
    target: pd.Series,
    *,
    quantiles: int = 5,
    cost: float | None = None,
) -> pd.DataFrame:
    """Target statistics by feature quantile bucket.

    This helps diagnose monotonicity and cost-aware signal usefulness.  Buckets
    are created with ``pd.qcut`` on each feature using only aligned rows.

    Returns
    -------
    pandas.DataFrame
        Long table with feature, bucket, count, target mean/median/std, hit rate
        and optional cost-aware columns.
    """

    if quantiles < 2:
        raise ValueError("quantiles must be >= 2.")
    df = _align_xy(features, target)
    y = df["__target__"]
    rows = []
    for col in df.drop(columns="__target__"):
        try:
            bins = pd.qcut(df[col], quantiles, labels=False, duplicates="drop")
        except ValueError as exc:
            warnings.warn(f"qcut failed for {col}: {exc}", RuntimeWarning)
            continue
        tmp = pd.DataFrame({"bucket": bins, "target": y}).dropna()
        g = tmp.groupby("bucket")["target"]
        out = g.agg(["count", "mean", "median", "std"]).reset_index()
        out.insert(0, "feature", col)
        out["hit_rate"] = g.apply(lambda s: (np.sign(s) > 0).mean()).to_numpy()
        if cost is not None:
            out["prob_abs_gt_cost"] = g.apply(lambda s: (s.abs() > cost).mean()).to_numpy()
            out["mean_excess"] = g.apply(lambda s: (s.abs() - cost).clip(lower=0).mean()).to_numpy()
        rows.append(out)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def granger_causality(
    feature: pd.Series,
    target: pd.Series,
    *,
    maxlag: int = 5,
    test: str = "ssr_ftest",
    verbose: bool = False,
) -> pd.DataFrame:
    """Test whether lagged feature values help predict target.

    The statsmodels input order is ``[target, feature]``; therefore the returned
    p-values answer "does feature Granger-cause target?".

    Returns
    -------
    pandas.DataFrame
        One row per lag with statistic, p-value and test name, or a warning row
        if the test cannot be computed.
    """

    if maxlag < 1:
        raise ValueError("maxlag must be positive.")
    df = _align_xy(_as_series(feature, name="feature"), _as_series(target, name="target"), min_obs=maxlag + 5)
    arr = df[["__target__", df.columns[0]]]
    try:
        res = grangercausalitytests(arr, maxlag=maxlag, verbose=verbose)
        rows = [{"lag": lag, "statistic": vals[0][test][0], "pvalue": vals[0][test][1], "test": test} for lag, vals in res.items()]
        return pd.DataFrame(rows)
    except Exception as exc:
        return pd.DataFrame([{"lag": np.nan, "statistic": np.nan, "pvalue": np.nan, "test": test, "warning": str(exc)}])


def mutual_information(
    features: pd.Series | pd.DataFrame,
    target: pd.Series,
    *,
    discrete_target: bool | None = None,
    n_neighbors: int = 3,
    random_state: int = 42,
) -> pd.DataFrame:
    """Estimate mutual information between features and target.

    Parameters
    ----------
    features : pandas.Series or pandas.DataFrame
        Feature values observed at time ``t``.
    target : pandas.Series
        Target aligned at the same index.  Use :func:`forward_return` for
        forward-return targets.
    discrete_target : bool, optional
        If ``None``, infer discrete targets from low-cardinality integer-like
        values.
    n_neighbors : int, default 3
        Neighbor count for sklearn's MI estimator.
    random_state : int, default 42
        Seed passed to sklearn.

    Returns
    -------
    pandas.DataFrame
        Columns: ``feature``, ``mutual_information`` and ``discrete_target``.

    Notes
    -----
    Rows are aligned by index and NaN/inf values are dropped.  MI is a nonlinear
    dependence diagnostic; values are not directly comparable across very
    different target scalings without care.

    Examples
    --------
    >>> mutual_information(df[["x1", "x2"]], forward_return(df["close"], 5))
    """

    df = _align_xy(features, target, min_obs=max(10, n_neighbors + 2))
    y = df.pop("__target__")
    if discrete_target is None:
        discrete_target = y.nunique() <= 20 and np.allclose(y, np.round(y))
    fn = mutual_info_classif if discrete_target else mutual_info_regression
    mi = fn(df, y, n_neighbors=n_neighbors, random_state=random_state)
    return pd.DataFrame({"feature": df.columns, "mutual_information": mi, "discrete_target": discrete_target}).sort_values(
        "mutual_information", ascending=False
    )


def rolling_mutual_information(
    features: pd.Series | pd.DataFrame,
    target: pd.Series,
    *,
    window: int,
    min_periods: int | None = None,
    step: int = 1,
    n_neighbors: int = 3,
    random_state: int = 42,
) -> pd.DataFrame:
    """Compute trailing rolling mutual information.

    Parameters
    ----------
    features, target : pandas.Series or pandas.DataFrame, pandas.Series
        Aligned feature matrix and target.
    window, min_periods : int
        Trailing window controls.
    step : int, default 1
        Evaluate every ``step`` rows to reduce cost.
    n_neighbors, random_state : int
        Passed to sklearn's mutual information estimator.

    Returns
    -------
    pandas.DataFrame
        Long table with timestamp, feature and mutual information.
    """

    window, min_periods = _validate_window(window, min_periods)
    df = _align_xy(features, target, min_obs=min_periods)
    y_name = "__target__"
    rows = []
    for end in range(min_periods, len(df) + 1, step):
        start = max(0, end - window)
        chunk = df.iloc[start:end]
        if len(chunk) < min_periods:
            continue
        mi = mutual_information(chunk.drop(columns=y_name), chunk[y_name], n_neighbors=n_neighbors, random_state=random_state)
        for _, r in mi.iterrows():
            rows.append({"timestamp": df.index[end - 1], "feature": r["feature"], "mutual_information": r["mutual_information"]})
    return pd.DataFrame(rows)


def distance_correlation(
    x: pd.Series,
    y: pd.Series,
    *,
    max_n: int = 3_000,
    random_state: int = 42,
) -> float:
    """Distance correlation for nonlinear dependence.

    Parameters
    ----------
    x, y : pandas.Series
        Numeric series aligned by index.  NaN/inf rows are dropped.
    max_n : int, default 3000
        Maximum rows used by the exact O(n^2) estimator.
    random_state : int, default 42
        Seed for subsampling when ``len(aligned) > max_n``.

    Returns
    -------
    float
        Distance correlation in ``[0, 1]`` when defined, otherwise ``NaN``.

    The exact estimator is O(n^2).  If aligned data has more than ``max_n``
    observations, a reproducible subsample is used and a warning is emitted.

    Notes
    -----
    This function measures nonlinear dependence but does not imply causality.
    For forward targets, pass an already aligned target from :func:`forward_return`.

    Examples
    --------
    >>> distance_correlation(df["x1"], forward_return(df["close"], 5))
    """

    df = _align_xy(_as_series(x, name="x"), _as_series(y, name="y"), min_obs=3)
    if len(df) > max_n:
        warnings.warn(f"Subsampled {max_n} of {len(df)} rows for distance correlation.", RuntimeWarning)
        df = df.sample(max_n, random_state=random_state).sort_index()
    a = np.abs(df.iloc[:, 0].to_numpy()[:, None] - df.iloc[:, 0].to_numpy()[None, :])
    b = np.abs(df["__target__"].to_numpy()[:, None] - df["__target__"].to_numpy()[None, :])
    A = a - a.mean(axis=0) - a.mean(axis=1)[:, None] + a.mean()
    B = b - b.mean(axis=0) - b.mean(axis=1)[:, None] + b.mean()
    dcov = np.sqrt(max(np.mean(A * B), 0))
    dvar_x = np.sqrt(np.mean(A * A))
    dvar_y = np.sqrt(np.mean(B * B))
    return float(dcov / np.sqrt(dvar_x * dvar_y)) if dvar_x > 0 and dvar_y > 0 else np.nan


def rolling_distance_correlation(
    feature: pd.Series,
    target: pd.Series,
    *,
    window: int,
    min_periods: int | None = None,
    step: int = 1,
    max_n: int = 1_000,
) -> pd.Series:
    """Compute trailing rolling distance correlation for one feature and target.

    Parameters
    ----------
    feature, target : pandas.Series
        Aligned series.
    window, min_periods : int
        Trailing window controls.
    step : int, default 1
        Evaluate every ``step`` rows.
    max_n : int, default 1000
        Maximum rows used inside each exact O(n^2) distance-correlation call.

    Returns
    -------
    pandas.Series
        Rolling distance correlation indexed by window end timestamp.
    """

    window, min_periods = _validate_window(window, min_periods)
    df = _align_xy(_as_series(feature, name="feature"), target, min_obs=min_periods)
    out = []
    idx = []
    for end in range(min_periods, len(df) + 1, step):
        start = max(0, end - window)
        chunk = df.iloc[start:end]
        out.append(distance_correlation(chunk.iloc[:, 0], chunk["__target__"], max_n=max_n))
        idx.append(df.index[end - 1])
    return pd.Series(out, index=idx, name="rolling_distance_correlation")


def conditional_ic(
    features: pd.Series | pd.DataFrame,
    target: pd.Series,
    condition_feature: pd.Series,
    *,
    quantile: float = 0.5,
    side: str = "above",
    method: str = "spearman",
    hac_lags: int | str | None = "auto",
) -> pd.DataFrame:
    """Compute IC conditional on another feature crossing a quantile.

    Parameters
    ----------
    features : pandas.Series or pandas.DataFrame
        Candidate feature(s) observed at time ``t``.
    target : pandas.Series
        Target aligned at time ``t``.
    condition_feature : pandas.Series
        Feature used to define the regime/condition.
    quantile : float, default 0.5
        Quantile threshold for the condition feature.
    side : {"above", "below"}, default "above"
        Whether to keep observations above or below the threshold.
    method : {"pearson", "spearman", "kendall"}, default "spearman"
        IC correlation method.
    hac_lags : int, "auto" or None, default "auto"
        Newey-West lags used by :func:`ic_summary`.

    Returns
    -------
    pandas.DataFrame
        IC summary table with additional condition metadata columns.

    Notes
    -----
    The condition threshold is estimated on the aligned sample passed to this
    function.  For rolling thresholds, use :func:`rolling_conditional_ic`.

    Examples
    --------
    >>> conditional_ic(df[["x1", "x2"]], y, df["vol"], quantile=0.8, side="above")
    """

    if not 0 < quantile < 1:
        raise ValueError("quantile must be in (0, 1).")
    side = side.lower()
    if side not in {"above", "below"}:
        raise ValueError("side must be 'above' or 'below'.")
    df = _align_xy(pd.concat([_as_frame(features), _as_series(condition_feature).rename("__condition__")], axis=1), target)
    threshold = df["__condition__"].quantile(quantile)
    mask = df["__condition__"].ge(threshold) if side == "above" else df["__condition__"].le(threshold)
    out = ic_summary(df.loc[mask].drop(columns=["__target__", "__condition__"]), df.loc[mask, "__target__"], method=method, hac_lags=hac_lags)
    out.insert(1, "condition_quantile", quantile)
    out.insert(2, "condition_side", side)
    out.insert(3, "condition_threshold", threshold)
    return out


def rolling_conditional_ic(
    features: pd.Series | pd.DataFrame,
    target: pd.Series,
    condition_feature: pd.Series,
    *,
    window: int,
    min_periods: int | None = None,
    quantile: float = 0.5,
    side: str = "above",
    method: str = "spearman",
    step: int = 1,
) -> pd.DataFrame:
    """Compute trailing rolling conditional IC.

    Parameters
    ----------
    features, target, condition_feature : pandas objects
        Feature matrix, aligned target and regime-defining feature.
    window, min_periods : int
        Trailing window controls.
    quantile : float, default 0.5
        Condition threshold estimated inside each window.
    side : {"above", "below"}, default "above"
        Keep observations above or below the rolling threshold.
    method : str, default "spearman"
        IC method passed to :func:`ic_summary`.
    step : int, default 1
        Evaluate every ``step`` rows.

    Returns
    -------
    pandas.DataFrame
        Long table with timestamp and IC summary columns.
    """

    window, min_periods = _validate_window(window, min_periods)
    df = _align_xy(pd.concat([_as_frame(features), _as_series(condition_feature).rename("__condition__")], axis=1), target, min_obs=min_periods)
    y_name = "__target__"
    rows = []
    for end in range(min_periods, len(df) + 1, step):
        chunk = df.iloc[max(0, end - window) : end]
        threshold = chunk["__condition__"].quantile(quantile)
        mask = chunk["__condition__"].ge(threshold) if side == "above" else chunk["__condition__"].le(threshold)
        if mask.sum() < 5:
            continue
        res = ic_summary(
            chunk.loc[mask].drop(columns=[y_name, "__condition__"]),
            chunk.loc[mask, y_name],
            method=method,
            hac_lags="auto",
        )
        for _, r in res.iterrows():
            rows.append({"timestamp": df.index[end - 1], **r.to_dict()})
    return pd.DataFrame(rows)


def feature_target_report(
    data: pd.DataFrame,
    features: Sequence[str],
    *,
    target: str | pd.Series | None = None,
    price: str | pd.Series | None = None,
    horizons: Sequence[int] = (1,),
    log_return: bool = True,
    method: str = "spearman",
    rolling_window: int | None = None,
    min_periods: int | None = None,
    rolling_step: int = 1,
    quantiles: int = 5,
    cost: float | None = None,
    hac_lags: int | str | None = "auto",
    plot: bool = True,
    run_granger: bool = False,
    granger_maxlag: int = 5,
    run_nonlinear: bool = True,
    verbose: bool = False,
) -> EDAResult:
    """Run feature-vs-target diagnostics over one or several horizons.

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame containing feature columns and, optionally, price/target
        columns.
    features : sequence of str
        Feature columns observed at time ``t``.
    target : str or pandas.Series, optional
        Already aligned target.  If provided, only the first value in
        ``horizons`` is used as a reporting label.
    price : str or pandas.Series, optional
        Price series used to create forward-return targets when ``target`` is
        omitted.
    horizons : sequence of int, default (1,)
        Forward-return horizons in bars.
    log_return : bool, default True
        Use log returns when creating targets from price.
    method : {"pearson", "spearman", "kendall"}, default "spearman"
        IC method for summary tables.  Rolling IC curves support Pearson and
        Spearman.
    rolling_window, min_periods : int, optional
        Trailing rolling window controls for rolling IC diagnostics.
    rolling_step : int, default 1
        Evaluate expensive rolling HAC summaries every ``rolling_step`` rows.
    quantiles : int, default 5
        Number of feature quantile buckets.
    cost : float, optional
        Return-unit threshold used in quantile target statistics.
    hac_lags : int, "auto" or None, default "auto"
        Newey-West lags.  For internally created horizon ``h`` targets, the
        effective lag is at least ``h - 1``.
    plot : bool, default True
        Include cumulative and rolling IC figures.
    run_granger : bool, default False
        Include Granger causality diagnostics.
    granger_maxlag : int, default 5
        Maximum lag for Granger tests.
    run_nonlinear : bool, default True
        Include mutual information and distance correlation diagnostics.
    verbose : bool, default False
        Emit warnings when optional diagnostics fail.

    Returns
    -------
    EDAResult
        Keys: ``ic``, ``cumulative_ic``, ``rolling_ic``,
        ``rolling_ic_stats``, ``quantiles``, ``mutual_information``,
        ``distance_correlation``, ``granger``, ``figures`` and ``warnings``.

    Notes
    -----
    If ``target`` is ``None``, ``price`` must be provided and forward returns are
    computed via :func:`forward_return`.  Feature rows at ``t`` are aligned to
    ``price[t+h] / price[t]`` at the same index ``t``; no future feature values
    are used.  Overlapping forward returns are handled in HAC t-statistics by
    enforcing at least ``h - 1`` Newey-West lags.

    Examples
    --------
    >>> feature_target_report(
    ...     df,
    ...     features=["volume", "turnover"],
    ...     price="close",
    ...     horizons=[1, 5, 10],
    ...     rolling_window=500,
    ...     plot=False,
    ... )
    """

    df = _as_frame(data)
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
    hs = _validate_horizons(horizons)
    x = df.loc[:, list(features)]

    if target is None:
        if price is None:
            raise ValueError("Provide either target or price for forward-return target creation.")
        price_s = df[price] if isinstance(price, str) else _as_series(price, name="price")
        targets = {h: forward_return(price_s, h, log_return=log_return) for h in hs}
    else:
        y = df[target] if isinstance(target, str) else _as_series(target, name="target")
        targets = {hs[0]: y}

    summaries = []
    rolling = {}
    rolling_stats = {}
    cumulative = {}
    quantile_tables = []
    mi_tables = []
    dcor_rows = []
    granger_tables = []
    report_warnings: list[str] = []
    figures: list[Figure] = []
    for h, y in targets.items():
        if verbose:
            print(f"feature_target_report: horizon={h}, n_target={y.notna().sum()}")
        min_hac_lags = max(int(h) - 1, 0)
        ic = ic_summary(x, y, method=method, hac_lags=hac_lags, min_hac_lags=min_hac_lags)
        ic.insert(0, "horizon", h)
        summaries.append(ic)
        if method in {"pearson", "spearman"}:
            cum = cumulative_ic(x, y, method=method, plot=plot)
            cumulative[h] = cum["table"]
            if cum["figure"] is not None:
                figures.append(cum["figure"])
        else:
            msg = f"cumulative_ic skipped for horizon={h}: method={method!r} is not supported."
            report_warnings.append(msg)
            if verbose:
                warnings.warn(msg, RuntimeWarning)
        qtab = feature_quantile_stats(x, y, quantiles=quantiles, cost=cost)
        if not qtab.empty:
            qtab.insert(0, "horizon", h)
            quantile_tables.append(qtab)
        if rolling_window is not None:
            rolling_stats[h] = rolling_ic_stats(
                x,
                y,
                window=rolling_window,
                min_periods=min_periods,
                method=method,
                hac_lags=hac_lags,
                min_hac_lags=min_hac_lags,
                step=rolling_step,
            )
            if method in {"pearson", "spearman"}:
                ric = rolling_ic(x, y, window=rolling_window, min_periods=min_periods, method=method)
                rolling[h] = ric
            else:
                ric = None
                msg = f"rolling_ic skipped for horizon={h}: method={method!r} is not supported by rolling correlation."
                report_warnings.append(msg)
                if verbose:
                    warnings.warn(msg, RuntimeWarning)
            if plot and ric is not None:
                fig, ax = plt.subplots(figsize=(9, 4.2))
                for i, col in enumerate(ric.columns[:6]):
                    ax.plot(ric.index, ric[col], color=COLD_PALETTE[i % len(COLD_PALETTE)], lw=1.4, label=col)
                style_axis(ax, ylabel="Rolling IC")
                fig.tight_layout()
                figures.append(fig)
        if run_nonlinear:
            try:
                mi = mutual_information(x, y)
                mi.insert(0, "horizon", h)
                mi_tables.append(mi)
            except Exception as exc:
                report_warnings.append(f"Mutual information failed for h={h}: {exc}")
                if verbose:
                    warnings.warn(f"Mutual information failed for h={h}: {exc}", RuntimeWarning)
            for col in features:
                try:
                    dcor_rows.append({"horizon": h, "feature": col, "distance_correlation": distance_correlation(x[col], y)})
                except Exception as exc:
                    report_warnings.append(f"Distance correlation failed for {col}, h={h}: {exc}")
                    if verbose:
                        warnings.warn(f"Distance correlation failed for {col}, h={h}: {exc}", RuntimeWarning)
        if run_granger:
            for col in features:
                g = granger_causality(x[col], y, maxlag=granger_maxlag, verbose=False)
                g.insert(0, "horizon", h)
                g.insert(1, "feature", col)
                granger_tables.append(g)

    return _result("Feature-target report", {
        "ic": pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame(),
        "cumulative_ic": cumulative,
        "rolling_ic": rolling,
        "rolling_ic_stats": rolling_stats,
        "quantiles": pd.concat(quantile_tables, ignore_index=True) if quantile_tables else pd.DataFrame(),
        "mutual_information": pd.concat(mi_tables, ignore_index=True) if mi_tables else pd.DataFrame(),
        "distance_correlation": pd.DataFrame(dcor_rows),
        "granger": pd.concat(granger_tables, ignore_index=True) if granger_tables else pd.DataFrame(),
        "figures": figures,
        "warnings": report_warnings,
    })


def correlation_matrix(data: pd.DataFrame, features: Sequence[str] | None = None, *, method: str = "pearson") -> pd.DataFrame:
    """Compute feature correlation matrix after numeric cleaning.

    Parameters
    ----------
    data : pandas.DataFrame
        Feature matrix.
    features : sequence of str, optional
        Columns to include.
    method : {"pearson", "spearman", "kendall"}, default "pearson"
        Correlation method.

    Returns
    -------
    pandas.DataFrame
        Square correlation matrix.
    """

    return _numeric_frame(data, features, dropna=True, min_obs=3).corr(method=method)


def heatmap_correlation_matrix(
    data: pd.DataFrame,
    features: Sequence[str] | None = None,
    *,
    method: str = "pearson",
    plot: bool = True,
) -> EDAResult:
    """Return correlation matrix plus an optional heatmap.

    Parameters are the same as :func:`correlation_matrix`, with ``plot``
    controlling the figure.

    Returns
    -------
    EDAResult
        Keys: ``correlation`` and optional ``figure``.
    """

    corr = correlation_matrix(data, features, method=method)
    fig = None
    if plot:
        fig, ax = plt.subplots(figsize=(max(5, 0.45 * len(corr)), max(4, 0.4 * len(corr))))
        im = ax.imshow(corr, vmin=-1, vmax=1, cmap="Blues")
        ax.set_xticks(range(len(corr)), corr.columns, rotation=90)
        ax.set_yticks(range(len(corr)), corr.index)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        style_axis(ax, legend=False)
        fig.tight_layout()
    return _result("Correlation matrix", {"correlation": corr, "figure": fig})


def vif(data: pd.DataFrame, features: Sequence[str] | None = None) -> pd.DataFrame:
    """Variance inflation factor by feature.

    This is the module's VIF-table helper; if you expected a function named
    ``vif_table``, use ``vif``.

    Parameters
    ----------
    data : pandas.DataFrame
        Feature matrix.
    features : sequence of str, optional
        Columns to include.  If omitted, all columns are used.

    Returns
    -------
    pandas.DataFrame
        Columns: ``feature``, ``vif`` and ``vif_gt_10``.  The share of features
        with VIF above 10 is stored in ``result.attrs["share_vif_gt_10"]``.

    Constant or all-missing columns are dropped.  Infinite VIF indicates exact
    or near-exact collinearity.

    Notes
    -----
    VIF is computed after standardizing finite rows.  It is a linear
    multicollinearity diagnostic and can be unstable when features are nearly
    perfectly collinear.

    Examples
    --------
    >>> vif(df, ["x1", "x2", "x3"])
    """

    df = _numeric_frame(data, features, dropna=True, min_obs=3)
    df = df.loc[:, df.std() > 0]
    if df.shape[1] < 2:
        return pd.DataFrame({"feature": df.columns, "vif": np.nan})
    x = StandardScaler().fit_transform(df)
    rows = []
    for i, col in enumerate(df.columns):
        try:
            val = variance_inflation_factor(x, i)
        except Exception:
            val = np.inf
        rows.append({"feature": col, "vif": val})
    out = pd.DataFrame(rows).sort_values("vif", ascending=False)
    out["vif_gt_10"] = out["vif"] > 10
    out.attrs["share_vif_gt_10"] = float(out["vif_gt_10"].mean()) if len(out) else np.nan
    return out


def cluster_features(
    data: pd.DataFrame,
    features: Sequence[str] | None = None,
    *,
    method: str = "pearson",
    linkage_method: str = "average",
    use_abs: bool = True,
    plot: bool = True,
) -> EDAResult:
    """Cluster features by correlation distance.

    Parameters
    ----------
    data, features : pandas.DataFrame, sequence of str
        Feature matrix and optional selected columns.
    method : str, default "pearson"
        Correlation method.
    linkage_method : str, default "average"
        Hierarchical linkage method.
    use_abs : bool, default True
        Cluster by ``1 - abs(corr)`` instead of signed distance.
    plot : bool, default True
        Include dendrogram figure.

    Returns
    -------
    EDAResult
        Keys: ``correlation``, ``distance``, ``linkage``, ``order``, optional
        ``figure`` and optional ``warning``.
    """

    corr = correlation_matrix(data, features, method=method)
    if len(corr) < 2:
        dist = pd.DataFrame(np.zeros(corr.shape), index=corr.index, columns=corr.columns)
        return _result("Feature clustering", {"correlation": corr, "distance": dist, "linkage": None, "order": corr.index.tolist(), "figure": None, "warning": "Need at least two features to cluster."})
    dist = (1 - corr.abs() if use_abs else (1 - corr) / 2).copy()
    dist_values = dist.to_numpy(copy=True)
    np.fill_diagonal(dist_values, 0)
    dist = pd.DataFrame(dist_values, index=dist.index, columns=dist.columns)
    z = linkage(squareform(dist.clip(lower=0).to_numpy()), method=linkage_method)
    order = corr.index[leaves_list(z)].tolist()
    fig = None
    if plot:
        fig, ax = plt.subplots(figsize=(max(6, 0.35 * len(order)), 4.5))
        dendrogram(z, labels=corr.index.tolist(), ax=ax, leaf_rotation=90, color_threshold=None)
        style_axis(ax, ylabel="Distance", legend=False)
        fig.tight_layout()
    return _result("Feature clustering", {"correlation": corr, "distance": dist, "linkage": z, "order": order, "figure": fig})


def pca_analysis(
    data: pd.DataFrame,
    features: Sequence[str] | None = None,
    *,
    n_components: int | None = None,
    scale: bool = True,
    max_rows: int | None = None,
    random_state: int = 42,
    plot: bool = True,
) -> EDAResult:
    """Run PCA with optional standardization.

    Parameters
    ----------
    data : pandas.DataFrame
        Feature matrix.
    features : sequence of str, optional
        Columns to include.
    n_components : int, optional
        Number of principal components.  Defaults to ``min(n_rows, n_features)``.
    scale : bool, default True
        Standardize features before PCA.
    max_rows : int, optional
        Optional reproducible row subsampling.
    random_state : int, default 42
        Seed for row sampling and PCA.
    plot : bool, default True
        Include cumulative explained-variance plot.

    Returns
    -------
    EDAResult
        Keys: ``explained_variance``, ``loadings``, ``transformed``, ``model``
        and optional ``figure``.

    Returns explained variance, loadings and transformed component scores.

    Notes
    -----
    Rows with NaN/inf values are dropped.  Scaling is enabled by default because
    PCA is sensitive to feature units.

    Examples
    --------
    >>> pca_analysis(df, ["x1", "x2", "x3"], scale=True, plot=False)["loadings"]
    """

    df, x = _prepare_feature_matrix(data, features, scale=scale, max_rows=max_rows, random_state=random_state)
    n_components = n_components or min(df.shape)
    model = PCA(n_components=n_components, random_state=random_state)
    scores = model.fit_transform(x)
    comp_cols = [f"PC{i + 1}" for i in range(scores.shape[1])]
    explained = pd.DataFrame(
        {
            "component": comp_cols,
            "explained_variance_ratio": model.explained_variance_ratio_,
            "cumulative": np.cumsum(model.explained_variance_ratio_),
        }
    )
    loadings = pd.DataFrame(model.components_.T, index=df.columns, columns=comp_cols)
    transformed = pd.DataFrame(scores, index=df.index, columns=comp_cols)
    fig = None
    if plot:
        fig, ax = plt.subplots(figsize=(7, 4.2))
        ax.plot(explained["component"], explained["cumulative"], marker="o", color=COLD_PALETTE[1], lw=2)
        style_axis(ax, xlabel="Component", ylabel="Cumulative explained variance", legend=False)
        fig.tight_layout()
    return _result("PCA analysis", {"explained_variance": explained, "loadings": loadings, "transformed": transformed, "model": model, "figure": fig})


def tsne_projection(
    data: pd.DataFrame,
    features: Sequence[str] | None = None,
    *,
    scale: bool = True,
    by_features: bool = False,
    perplexity: float = 30.0,
    max_rows: int | None = 10_000,
    random_state: int = 42,
) -> pd.DataFrame:
    """Compute a two-dimensional t-SNE projection.

    Parameters
    ----------
    by_features:
        If ``False``, embed observations.  If ``True``, embed features by using
        the transposed standardized observation matrix; this is useful for
        visualizing feature structure.
    max_rows:
        Optional row subsampling before t-SNE.  This is strongly recommended for
        minute-level crypto data.
    """

    df, x = _prepare_feature_matrix(data, features, scale=scale, max_rows=max_rows, random_state=random_state)
    labels = df.columns if by_features else df.index
    x_in = x.T if by_features else x
    n = x_in.shape[0]
    if n < 3:
        raise ValueError("t-SNE needs at least 3 samples.")
    perplexity = min(perplexity, max(1, n - 1) / 3)
    emb = TSNE(n_components=2, perplexity=perplexity, random_state=random_state, init="pca", learning_rate="auto").fit_transform(x_in)
    return pd.DataFrame(emb, index=labels, columns=["tsne_1", "tsne_2"])


def feature_relation_report(
    data: pd.DataFrame,
    features: Sequence[str],
    *,
    method: str = "pearson",
    scale: bool = True,
    max_rows: int | None = 10_000,
    plot: bool = True,
    run_tsne: bool = True,
) -> EDAResult:
    """Aggregate feature-feature diagnostics: correlation, VIF, clustering, PCA and t-SNE.

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame containing feature columns.
    features : sequence of str
        Numeric feature columns to analyze.
    method : {"pearson", "spearman", "kendall"}, default "pearson"
        Correlation method.
    scale : bool, default True
        Standardize features for PCA/t-SNE.
    max_rows : int, optional
        Optional row subsampling for PCA/t-SNE.
    plot : bool, default True
        Include heatmap, clustering and PCA figures.
    run_tsne : bool, default True
        Include feature-level t-SNE projection when feasible.

    Returns
    -------
    EDAResult
        Keys: ``correlation``, ``vif``, ``clustering``, ``pca``, ``tsne`` and
        optional ``tsne_warning``.

    Notes
    -----
    Rows with NaN/inf values are dropped for matrix diagnostics.  VIF and PCA
    use only numeric finite rows.

    Examples
    --------
    >>> feature_relation_report(df, ["x1", "x2", "x3"], plot=False)
    """

    out = {
        "correlation": heatmap_correlation_matrix(data, features, method=method, plot=plot),
        "vif": vif(data, features),
        "clustering": cluster_features(data, features, method=method, plot=plot),
        "pca": pca_analysis(data, features, scale=scale, max_rows=max_rows, plot=plot),
        "tsne": None,
    }
    if run_tsne:
        try:
            out["tsne"] = tsne_projection(data, features, scale=scale, by_features=True, max_rows=max_rows)
        except Exception as exc:
            out["tsne_warning"] = str(exc)
    return _result("Feature relation report", out)


def realized_volatility(returns: pd.Series, window: int, *, min_periods: int | None = None, annualization: float | None = None) -> pd.Series:
    """Compute trailing realized volatility ``sqrt(sum(r^2))``.

    Parameters
    ----------
    returns : pandas.Series
        Return series.
    window, min_periods : int
        Trailing rolling window controls.
    annualization : float, optional
        If provided, multiply by ``sqrt(annualization)``.

    Returns
    -------
    pandas.Series
        Realized volatility aligned to the input index.
    """

    window, min_periods = _validate_window(window, min_periods)
    rv = np.sqrt((_numeric_series(returns, dropna=False) ** 2).rolling(window, min_periods=min_periods).sum())
    return rv * np.sqrt(annualization) if annualization else rv


def bipower_variation(returns: pd.Series, window: int, *, min_periods: int | None = None) -> pd.Series:
    """Compute trailing bipower variation.

    Parameters are return series and trailing window controls.  The output is a
    ``pandas.Series`` aligned to the input index.

    Notes
    -----
    Bipower variation is commonly used as a jump-robust volatility diagnostic
    under high-frequency assumptions.
    """

    window, min_periods = _validate_window(window, min_periods)
    r = _numeric_series(returns, dropna=False).abs()
    return (np.pi / 2) * (r * r.shift(1)).rolling(window, min_periods=min_periods).sum()


def rolling_sharpe(
    returns: pd.Series,
    window: int,
    *,
    min_periods: int | None = None,
    periods_per_year: float | None = None,
) -> pd.Series:
    """Compute trailing Sharpe-like mean/std diagnostic for returns.

    Parameters
    ----------
    returns : pandas.Series
        Return series.
    window, min_periods : int
        Trailing rolling window controls.
    periods_per_year : float, optional
        Annualization multiplier applied as ``sqrt(periods_per_year)``.
    """

    window, min_periods = _validate_window(window, min_periods)
    r = _numeric_series(returns, dropna=False)
    out = r.rolling(window, min_periods=min_periods).mean() / r.rolling(window, min_periods=min_periods).std()
    return out * np.sqrt(periods_per_year) if periods_per_year else out


def drawdown_diagnostics(series: pd.Series, *, input_type: str = "returns") -> EDAResult:
    """Compute drawdown curve and summary from returns, price or equity.

    Parameters
    ----------
    series : pandas.Series
        Returns, price or equity curve.
    input_type : {"returns", "price", "equity"}, default "returns"
        Interpretation of ``series``.

    Returns
    -------
    EDAResult
        Keys: ``equity``, ``drawdown`` and ``summary``.
    """

    x = _numeric_series(series, dropna=True, min_obs=2, warn=False)
    if input_type == "returns":
        equity = (1 + x).cumprod()
    elif input_type in {"price", "equity"}:
        equity = x / x.iloc[0]
    else:
        raise ValueError("input_type must be 'returns', 'price' or 'equity'.")
    dd = equity / equity.cummax() - 1
    summary = pd.DataFrame(
        [
            {
                "total_return": equity.iloc[-1] - 1,
                "max_drawdown": dd.min(),
                "drawdown_end": dd.idxmin(),
                "current_drawdown": dd.iloc[-1],
            }
        ]
    )
    return _result("Drawdown diagnostics", {"equity": equity.rename("equity"), "drawdown": dd.rename("drawdown"), "summary": summary})


def tail_dependence(x: pd.Series, y: pd.Series, *, q: float = 0.95, tail: str = "upper") -> pd.DataFrame:
    """Estimate empirical tail dependence ``P(Y in tail | X in tail)``.

    Parameters
    ----------
    x, y : pandas.Series
        Aligned numeric series.
    q : float, default 0.95
        Tail quantile.
    tail : {"upper", "lower"}, default "upper"
        Tail side.

    Returns
    -------
    pandas.DataFrame
        One-row table with tail dependence estimate.
    """

    if not 0 < q < 1:
        raise ValueError("q must be in (0, 1).")
    df = _align_xy(_as_series(x, name="x"), _as_series(y, name="y"))
    xs, ys = df.iloc[:, 0], df["__target__"]
    if tail == "upper":
        mx, my = xs >= xs.quantile(q), ys >= ys.quantile(q)
    elif tail == "lower":
        mx, my = xs <= xs.quantile(1 - q), ys <= ys.quantile(1 - q)
    else:
        raise ValueError("tail must be 'upper' or 'lower'.")
    return pd.DataFrame([{"tail": tail, "q": q, "n_tail_x": int(mx.sum()), "tail_dependence": float(my[mx].mean())}])


def upside_downside_volatility(returns: pd.Series, *, threshold: float = 0.0) -> pd.DataFrame:
    """Compute upside and downside volatility split around a threshold.

    Returns a one-row ``pandas.DataFrame`` with volatilities, sample counts and
    downside share.

    Parameters
    ----------
    returns : pandas.Series
        Return series.
    threshold : float, default 0.0
        Split point for upside/downside observations.
    """

    r = _numeric_series(returns, dropna=True, min_obs=3, warn=False)
    up, down = r[r > threshold], r[r < threshold]
    return pd.DataFrame(
        [
            {
                "threshold": threshold,
                "upside_vol": up.std(),
                "downside_vol": down.std(),
                "upside_n": len(up),
                "downside_n": len(down),
                "downside_share": len(down) / len(r),
            }
        ]
    )


def hit_rate(prediction: pd.Series, target: pd.Series, *, threshold: float = 0.0) -> pd.DataFrame:
    """Compute directional hit rate between prediction and target signs.

    ``prediction`` and ``target`` are aligned by index; NaN/inf rows are
    dropped.  Returns a one-row ``pandas.DataFrame`` with hit rate and coverage.
    """

    df = _align_xy(_as_series(prediction, name="prediction"), _as_series(target, name="target"))
    pred_sign = np.sign(df.iloc[:, 0] - threshold)
    target_sign = np.sign(df["__target__"])
    active = pred_sign != 0
    return pd.DataFrame(
        [
            {
                "n": int(active.sum()),
                "coverage": float(active.mean()),
                "hit_rate": float((pred_sign[active] == target_sign[active]).mean()) if active.any() else np.nan,
            }
        ]
    )


def turnover_cost_diagnostics(
    positions: pd.Series,
    returns: pd.Series | None = None,
    *,
    cost: float = 0.0005,
) -> EDAResult:
    """Turnover and optional cost-aware strategy diagnostics.

    Parameters
    ----------
    positions : pandas.Series
        Position or signal series indexed by time.
    returns : pandas.Series, optional
        Return series aligned to the same index.
    cost : float, default 0.0005
        Per-unit turnover cost in return units.

    Returns
    -------
    EDAResult
        Keys: ``turnover`` and ``summary``.  If ``returns`` is provided, also
        includes ``strategy_returns`` with gross, cost and net returns.

    If returns are provided, position at ``t-1`` is applied to return at ``t`` to
    avoid look-ahead bias.

    Notes
    -----
    The function assumes ``positions`` represent holdings decided before the
    next return.  Turnover is ``abs(position.diff())`` and costs are charged on
    that turnover.

    Examples
    --------
    >>> turnover_cost_diagnostics(positions, returns=df["return"], cost=0.0005)["summary"]
    """

    p = _numeric_series(positions, dropna=False).fillna(0)
    turnover = p.diff().abs().fillna(p.abs()).rename("turnover")
    summary = {"avg_turnover": turnover.mean(), "total_turnover": turnover.sum(), "avg_cost": (turnover * cost).mean()}
    out = {"turnover": turnover, "summary": pd.DataFrame([summary])}
    if returns is not None:
        r = _numeric_series(returns, dropna=False).reindex(p.index)
        gross = p.shift(1).fillna(0) * r
        net = gross - turnover * cost
        out["strategy_returns"] = pd.DataFrame({"gross": gross, "cost": turnover * cost, "net": net})
        out["summary"] = pd.DataFrame([{**summary, "gross_mean": gross.mean(), "net_mean": net.mean(), "cost_share_abs_gross": (turnover * cost).sum() / gross.abs().sum()}])
    return _result("Turnover/cost diagnostics", out)


def ljung_box_tests(returns: pd.Series, *, lags: Sequence[int] = (10, 20, 50), squared: bool = True) -> pd.DataFrame:
    """Run Ljung-Box autocorrelation tests for returns and squared returns.

    Parameters
    ----------
    returns : pandas.Series
        Return series.
    lags : sequence of int
        Ljung-Box lags.
    squared : bool, default True
        Also test squared returns for volatility clustering.

    Returns
    -------
    pandas.DataFrame
        Rows for each requested lag and series type.
    """

    r = _numeric_series(returns, dropna=True, min_obs=max(lags) + 2, warn=False)
    frames = []
    lb = acorr_ljungbox(r, lags=list(lags), return_df=True).rename_axis("lag").reset_index()
    lb.insert(0, "series", "returns")
    frames.append(lb)
    if squared:
        lb2 = acorr_ljungbox(r**2, lags=list(lags), return_df=True).rename_axis("lag").reset_index()
        lb2.insert(0, "series", "squared_returns")
        frames.append(lb2)
    return pd.concat(frames, ignore_index=True)


def missingness_by_time_bucket(series: pd.Series, *, bucket: str = "hour") -> pd.DataFrame:
    """Missingness percentage by calendar bucket for DatetimeIndex series.

    ``bucket`` can be ``"hour"``, ``"dayofweek"`` or ``"month"``.

    Parameters
    ----------
    series : pandas.Series
        Series with a ``DatetimeIndex``.
    bucket : {"hour", "dayofweek", "month"}
        Calendar bucket used for grouping.

    Returns
    -------
    pandas.DataFrame
        Calendar bucket and missing percentage.
    """

    s = _as_series(series)
    if not isinstance(s.index, pd.DatetimeIndex):
        raise ValueError("missingness_by_time_bucket requires a DatetimeIndex.")
    keys = {"hour": s.index.hour, "dayofweek": s.index.dayofweek, "month": s.index.month}
    if bucket not in keys:
        raise ValueError("bucket must be 'hour', 'dayofweek' or 'month'.")
    return s.isna().groupby(keys[bucket]).mean().mul(100).rename("missing_pct").rename_axis(bucket).reset_index()


def calendar_seasonality(series: pd.Series, *, bucket: str = "hour") -> pd.DataFrame:
    """Compute mean, median, std and count by calendar bucket.

    Parameters
    ----------
    series : pandas.Series
        Numeric series with a ``DatetimeIndex``.
    bucket : {"hour", "dayofweek", "month"}, default "hour"
        Calendar grouping.

    Returns
    -------
    pandas.DataFrame
        Calendar bucket summary table.
    """

    s = _numeric_series(series, dropna=False, warn=False)
    if not isinstance(s.index, pd.DatetimeIndex):
        raise ValueError("calendar_seasonality requires a DatetimeIndex.")
    keys = {"hour": s.index.hour, "dayofweek": s.index.dayofweek, "month": s.index.month}
    if bucket not in keys:
        raise ValueError("bucket must be 'hour', 'dayofweek' or 'month'.")
    return s.groupby(keys[bucket]).agg(["count", "mean", "median", "std", "skew"]).rename_axis(bucket).reset_index()


class EDA:
    """Notebook-friendly facade around the standalone EDA functions.

    Parameters
    ----------
    df:
        Source DataFrame.
    time_col:
        Optional timestamp column.  If provided, it is converted with
        ``pd.to_datetime``, used as the index, and the frame is sorted by time.
        If ``None``, the existing index is preserved.
    price_col:
        Default price column used for forward-return target creation.
    target_col:
        Optional default target column for feature-target reports.
    cold_palette:
        If ``True``, install the module's cold blue matplotlib defaults.
    copy:
        If ``True``, keep a defensive copy of ``df``.

    Method Reference
    ----------------
    forward_return(horizon=1, *, price_col=None, log_return=True)
        Input: ``EDA.df`` and a positive price column, usually ``close``.
        Computes: ``log(price[t+h] / price[t])`` or simple forward return,
        indexed at decision time ``t``.  The last ``horizon`` rows are ``NaN``.
        Returns: ``pd.Series`` with the same index as ``EDA.df``.
        Use for: creating aligned targets before feature-target analysis.

    target_selection(horizons, *, price_col=None, cost=0.0005,
    cost_is_multiplier=False, log_return=True, rolling_window=None,
    min_periods=None, plot=True)
        Input: ``EDA.df`` and a price column.
        Computes: forward-return magnitude, tail and cost-coverage statistics
        for each horizon; optionally trailing rolling ``P(|r| > cost)``.
        Returns: ``EDAResult`` with ``summary``, ``targets``,
        ``rolling_probability`` and optional ``figure``.
        Use for: choosing economically meaningful prediction horizons.

    data_diagnostics(cols=None, *, rolling_windows=(10080, 43200, 86400),
    quantiles=(...), lags=40, step=1, plot=True, verbose=False)
        Input: selected numeric columns from ``EDA.df``.
        Computes: missingness, moments, quantiles, ADF/KPSS/Zivot-Andrews,
        ACF/PACF and trailing rolling moments.
        Returns: ``EDAResult`` with ``summary``, ``stationarity``,
        ``acf_pacf``, ``rolling``, ``figures`` and ``warnings``.
        Use for: first-pass data quality and stationarity checks.  Use
        ``step=h`` for overlapping ``h``-bar returns.

    distribution_report(col, *, q=0.95, tail="abs", arch_lags=10,
    step=1, plot=True)
        Input: one numeric column from ``EDA.df``.
        Computes: summary stats, density/QQ diagnostics, Hill tail index,
        GPD tail fit, normality tests, ARCH LM and optional class balance.
        Returns: ``EDAResult`` with ``summary``, ``density``, ``qq``, ``hill``,
        ``evt``, ``normality``, ``arch_lm`` and ``class_balance``.
        Use for: return distribution, tail risk and volatility clustering
        diagnostics.  Use ``step=h`` for overlapping returns.

    seasonality_report(col, *, period=None, fs=1.0, plot=True, verbose=False)
        Input: one numeric column, ideally with a ``DatetimeIndex``.
        Computes: periodogram, Lomb-Scargle, Hurst exponent, CUSUM, structural
        breaks and optional STL decomposition.
        Returns: ``EDAResult`` with ``periodogram``, ``lomb_scargle``,
        ``hurst_rs``, ``cusum``, ``structural_breaks`` and ``stl``.
        Use for: seasonal patterns, regime instability and structural-change
        diagnostics.

    feature_target_report(features, *, target=None, price_col=None,
    horizons=(1,), log_return=True, method="spearman", rolling_window=None,
    min_periods=None, rolling_step=1, quantiles=5, cost=None,
    hac_lags="auto", plot=True, run_granger=False, granger_maxlag=5,
    run_nonlinear=True, verbose=False)
        Input: feature columns from ``EDA.df`` plus either a target column or a
        price column used to create forward-return targets.
        Computes: IC, HAC/Newey-West t-statistics, cumulative IC, rolling IC
        diagnostics, feature quantile tables, mutual information, distance
        correlation and optional Granger causality.
        Returns: ``EDAResult`` with ``ic``, ``cumulative_ic``, ``rolling_ic``,
        ``rolling_ic_stats``, ``quantiles``, ``mutual_information``,
        ``distance_correlation``, ``granger``, ``figures`` and ``warnings``.
        Use for: primary feature-vs-target hypothesis checks.  When targets are
        built from ``horizons``, HAC lags are at least ``h - 1`` to account for
        mechanically overlapping returns.

    feature_relation_report(features, *, method="pearson", scale=True,
    max_rows=10000, plot=True, run_tsne=True)
        Input: feature columns from ``EDA.df``.
        Computes: correlation matrix/heatmap, VIF table, correlation
        clustering, PCA and optional feature-level t-SNE.
        Returns: ``EDAResult`` with ``correlation``, ``vif``, ``clustering``,
        ``pca``, ``tsne`` and optional ``tsne_warning``.
        Use for: redundancy, multicollinearity and latent structure checks.

    Notebook Display
    ----------------
    Report methods return :class:`EDAResult`, a ``dict`` subclass with a clean
    HTML representation in Jupyter.  Full DataFrames remain available through
    keys such as ``result["summary"]``.

    Standalone Functions
    --------------------
    The same functionality is available through ``import eda``.  Use
    ``help(eda)`` or ``help(eda.function_name)`` for the full standalone API,
    including lower-level functions such as ``acf_pacf``, ``ic_summary``,
    ``rolling_ic_stats``, ``normality_tests``, ``pca_analysis`` and
    ``turnover_cost_diagnostics``.

    Examples
    --------
    >>> from eda import EDA
    >>> eda = EDA(df, time_col="datetime", price_col="close")
    >>> eda.target_selection(horizons=[1, 5, 10], cost=0.0005)
    >>> eda.data_diagnostics(cols=["close"])
    >>> eda.distribution_report(col="return")
    >>> eda.seasonality_report(col="return", period=1440)
    >>> eda.feature_target_report(features=["volume", "turnover"], horizons=[1, 5])
    >>> eda.feature_relation_report(features=["volume", "turnover"])

    Notes
    -----
    The class does not duplicate analytical logic.  Methods delegate to
    functions in this module, so the same behavior is available via ``import
    eda`` and direct function calls.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        *,
        time_col: str | None = None,
        price_col: str = "close",
        target_col: str | None = None,
        cold_palette: bool = True,
        copy: bool = True,
    ) -> None:
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame.")
        data = df.copy() if copy else df
        if time_col is not None:
            if time_col not in data.columns:
                raise ValueError(f"time_col {time_col!r} is not in DataFrame.")
            data[time_col] = pd.to_datetime(data[time_col])
            data = data.set_index(time_col).sort_index()
        if price_col is not None and price_col not in data.columns:
            raise ValueError(f"price_col {price_col!r} is not in DataFrame.")
        if target_col is not None and target_col not in data.columns:
            raise ValueError(f"target_col {target_col!r} is not in DataFrame.")
        self.df = data
        self.price_col = price_col
        self.target_col = target_col
        if cold_palette:
            set_plot_style()

    def _series(self, col: str | pd.Series) -> pd.Series:
        if isinstance(col, str):
            if col not in self.df.columns:
                raise ValueError(f"Column {col!r} is not in DataFrame.")
            return self.df[col]
        return _as_series(col)

    def forward_return(self, horizon: int = 1, *, price_col: str | None = None, log_return: bool = True) -> pd.Series:
        """Compute a forward return aligned at decision time ``t``.

        Parameters
        ----------
        horizon : int, default 1
            Positive number of bars ahead.
        price_col : str, optional
            Price column in ``self.df``.  Defaults to the class-level
            ``price_col`` passed to :class:`EDA`.
        log_return : bool, default True
            If ``True``, compute ``log(price[t+h] / price[t])``.  Otherwise
            compute simple percentage return.

        Returns
        -------
        pandas.Series
            Forward return with the same index as ``self.df``.  The final
            ``horizon`` rows are ``NaN`` because the future price is unknown.

        Notes
        -----
        The returned target is indexed at the feature observation time ``t``.
        Join features at ``t`` to this series at the same index to avoid
        look-ahead bias.

        Examples
        --------
        >>> eda.forward_return(horizon=5)
        """

        return forward_return(self.df[price_col or self.price_col], horizon=horizon, log_return=log_return)

    def target_selection(
        self,
        horizons: Sequence[int],
        *,
        price_col: str | None = None,
        cost: float = 0.0005,
        cost_is_multiplier: bool = False,
        log_return: bool = True,
        rolling_window: int | None = None,
        min_periods: int | None = None,
        plot: bool = True,
    ) -> EDAResult:
        """Compare candidate forward-return targets across horizons.

        Parameters
        ----------
        horizons : sequence of int
            Positive forecast horizons in bars.
        price_col : str, optional
            Price column used to build forward returns.  Defaults to the class
            ``price_col``.
        cost : float, default 0.0005
            Cost threshold in return units.  Use ``cost_is_multiplier=True`` if
            passing an old-style multiplier such as ``1.002``.
        cost_is_multiplier : bool, default False
            Convert ``cost`` with ``log(cost)`` before comparisons.
        log_return : bool, default True
            Use log returns instead of simple percentage returns.
        rolling_window : int, optional
            Trailing window in rows for rolling probability and rolling std.
        min_periods : int, optional
            Minimum observations required inside rolling windows.
        plot : bool, default True
            Include a minimalist rolling probability figure.

        Returns
        -------
        EDAResult
            Keys: ``summary`` with horizon statistics, ``targets`` with aligned
            forward returns, ``rolling_probability`` if ``rolling_window`` is
            provided, and optional ``figure``.

        Notes
        -----
        Forward returns are aligned at time ``t`` and the final ``h`` rows are
        left as ``NaN``.  Rolling probability uses trailing windows only and
        does not count unavailable future targets as failures.

        Examples
        --------
        >>> eda.target_selection(horizons=[1, 5, 10], cost=0.0005, plot=False)
        """

        return target_selection(
            self.df[price_col or self.price_col],
            horizons=horizons,
            cost=cost,
            cost_is_multiplier=cost_is_multiplier,
            log_return=log_return,
            rolling_window=rolling_window,
            min_periods=min_periods,
            plot=plot,
        )

    def data_diagnostics(
        self,
        cols: Sequence[str] | None = None,
        *,
        rolling_windows: Sequence[int] = (10_080, 43_200, 86_400),
        quantiles: Sequence[float] = (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99),
        lags: int = 40,
        step: int = 1,
        plot: bool = True,
        verbose: bool = False,
    ) -> EDAResult:
        """Run data-quality, stationarity, ACF/PACF and rolling diagnostics.

        Parameters
        ----------
        cols : sequence of str, optional
            Columns from ``self.df``.  If omitted, all columns are considered.
        rolling_windows : sequence of int, default (10080, 43200, 86400)
            Trailing rolling windows in rows.
        quantiles : sequence of float
            Quantiles included in the summary table.
        lags : int, default 40
            Number of ACF/PACF lags.
        step : int, default 1
            Subsampling step used in stationarity tests and ACF/PACF.  Use
            ``step=h`` for overlapping ``h``-bar returns.
        plot : bool, default True
            Include ACF/PACF and rolling std figures.
        verbose : bool, default False
            Emit warnings when a sub-diagnostic fails.

        Returns
        -------
        EDAResult
            Keys: ``summary``, ``stationarity``, ``acf_pacf``, ``rolling``,
            ``figures`` and ``warnings``.

        Notes
        -----
        Rolling statistics are trailing.  NaN and infinite values are excluded
        from tests, while rolling outputs preserve the original index.

        Examples
        --------
        >>> eda.data_diagnostics(cols=["return"], lags=24, step=5, plot=False)
        """

        return data_diagnostics(
            self.df,
            cols=cols,
            rolling_windows=rolling_windows,
            quantiles=quantiles,
            lags=lags,
            step=step,
            plot=plot,
            verbose=verbose,
        )

    def distribution_report(
        self,
        col: str,
        *,
        q: float = 0.95,
        tail: str = "abs",
        arch_lags: int = 10,
        step: int = 1,
        plot: bool = True,
    ) -> EDAResult:
        """Run distribution, tail, normality and ARCH diagnostics.

        Parameters
        ----------
        col : str
            Numeric column in ``self.df``.
        q : float, default 0.95
            Tail threshold quantile for Hill and GPD diagnostics.
        tail : {"abs", "right", "left"}, default "abs"
            Tail definition for extreme-value diagnostics.
        arch_lags : int, default 10
            Lags for Engle ARCH LM test.
        step : int, default 1
            Test subsampling step.  Use ``step=h`` for overlapping ``h``-bar
            returns.
        plot : bool, default True
            Include density, QQ and optional GPD figures.

        Returns
        -------
        EDAResult
            Keys: ``summary``, ``density``, ``qq``, ``hill``, ``evt``,
            ``normality``, ``arch_lm`` and optional ``class_balance``.

        Notes
        -----
        Shapiro-Wilk is subsampled for large samples.  KS uses mean/std
        estimated from the sample and is therefore a diagnostic, not a strict
        Lilliefors-adjusted test.

        Examples
        --------
        >>> eda.distribution_report("return", q=0.99, step=10, plot=False)
        """

        return distribution_report(self._series(col), q=q, tail=tail, arch_lags=arch_lags, step=step, plot=plot)

    def seasonality_report(
        self,
        col: str,
        *,
        period: int | None = None,
        fs: float = 1.0,
        plot: bool = True,
        verbose: bool = False,
    ) -> EDAResult:
        """Run seasonality, persistence and regime-change diagnostics.

        Parameters
        ----------
        col : str
            Numeric column in ``self.df``.
        period : int, optional
            Seasonal period in rows for STL.  If omitted, a daily period is
            inferred when the index is datetime-like and regular enough.
        fs : float, default 1.0
            Sampling frequency for the regular periodogram.
        plot : bool, default True
            Include periodogram, Lomb-Scargle and STL figures where possible.
        verbose : bool, default False
            Emit warnings for optional diagnostics that cannot be computed.

        Returns
        -------
        EDAResult
            Keys: ``periodogram``, ``lomb_scargle``, ``hurst_rs``, ``cusum``,
            ``structural_breaks`` and ``stl``.

        Notes
        -----
        Lomb-Scargle is useful for irregular timestamps.  Structural breaks use
        the installed ``ruptures`` package as a practical Bai-Perron-style
        alternative.

        Examples
        --------
        >>> eda.seasonality_report("return", period=1440, plot=False)
        """

        return seasonality_report(self._series(col), period=period, fs=fs, plot=plot, verbose=verbose)

    def feature_target_report(
        self,
        features: Sequence[str],
        *,
        target: str | pd.Series | None = None,
        price_col: str | None = None,
        horizons: Sequence[int] = (1,),
        log_return: bool = True,
        method: str = "spearman",
        rolling_window: int | None = None,
        min_periods: int | None = None,
        rolling_step: int = 1,
        quantiles: int = 5,
        cost: float | None = None,
        hac_lags: int | str | None = "auto",
        plot: bool = True,
        run_granger: bool = False,
        granger_maxlag: int = 5,
        run_nonlinear: bool = True,
        verbose: bool = False,
    ) -> EDAResult:
        """Run feature-vs-target diagnostics over one or more horizons.

        Parameters
        ----------
        features : sequence of str
            Feature columns observed at time ``t``.
        target : str or pandas.Series, optional
            Already aligned target.  If omitted, forward returns are built from
            ``price_col``.
        price_col : str, optional
            Price column used for forward-return targets when ``target`` is not
            supplied.
        horizons : sequence of int, default (1,)
            Forward-return horizons in bars.
        log_return : bool, default True
            Use log forward returns when creating targets from price.
        method : {"pearson", "spearman", "kendall"}, default "spearman"
            Correlation method for IC summaries.  Rolling IC supports Pearson
            and Spearman only.
        rolling_window, min_periods : int, optional
            Trailing rolling window controls for rolling IC diagnostics.
        rolling_step : int, default 1
            Evaluate expensive rolling HAC summaries every ``rolling_step`` rows.
        quantiles : int, default 5
            Number of feature buckets for quantile diagnostics.
        cost : float, optional
            Cost threshold used in quantile target statistics.
        hac_lags : int, "auto" or None, default "auto"
            Newey-West lags.  When targets are created from horizon ``h``, the
            effective lag is at least ``h - 1``.
        plot : bool, default True
            Include cumulative and rolling IC figures.
        run_granger : bool, default False
            Include Granger causality tables.
        granger_maxlag : int, default 5
            Maximum lag for Granger tests.
        run_nonlinear : bool, default True
            Include mutual information and distance correlation diagnostics.
        verbose : bool, default False
            Emit warnings when optional diagnostics fail.

        Returns
        -------
        EDAResult
            Keys: ``ic``, ``cumulative_ic``, ``rolling_ic``,
            ``rolling_ic_stats``, ``quantiles``, ``mutual_information``,
            ``distance_correlation``, ``granger``, ``figures`` and
            ``warnings``.

        Notes
        -----
        If forward returns are created internally, features at ``t`` are aligned
        with ``price[t+h] / price[t]`` at the same index ``t``.  No future
        feature values are used.  Overlap in ``h``-bar returns is handled in HAC
        t-statistics by using at least ``h - 1`` Newey-West lags.

        Examples
        --------
        >>> eda.feature_target_report(
        ...     features=["volume", "turnover"],
        ...     horizons=[1, 5, 10],
        ...     rolling_window=500,
        ...     plot=False,
        ... )
        """

        chosen_target = target if target is not None else self.target_col
        price = None if chosen_target is not None else (price_col or self.price_col)
        return feature_target_report(
            self.df,
            features,
            target=chosen_target,
            price=price,
            horizons=horizons,
            log_return=log_return,
            method=method,
            rolling_window=rolling_window,
            min_periods=min_periods,
            rolling_step=rolling_step,
            quantiles=quantiles,
            cost=cost,
            hac_lags=hac_lags,
            plot=plot,
            run_granger=run_granger,
            granger_maxlag=granger_maxlag,
            run_nonlinear=run_nonlinear,
            verbose=verbose,
        )

    def feature_relation_report(
        self,
        features: Sequence[str],
        *,
        method: str = "pearson",
        scale: bool = True,
        max_rows: int | None = 10_000,
        plot: bool = True,
        run_tsne: bool = True,
    ) -> EDAResult:
        """Run feature-feature diagnostics for selected columns.

        Parameters
        ----------
        features : sequence of str
            Numeric feature columns from ``self.df``.
        method : {"pearson", "spearman", "kendall"}, default "pearson"
            Correlation method.
        scale : bool, default True
            Standardize features before PCA and t-SNE.
        max_rows : int, optional
            Optional row subsampling for PCA/t-SNE.
        plot : bool, default True
            Include heatmap, dendrogram and PCA explained-variance figures.
        run_tsne : bool, default True
            Compute feature-level t-SNE when enough features are available.

        Returns
        -------
        EDAResult
            Keys: ``correlation``, ``vif``, ``clustering``, ``pca``, ``tsne``
            and optional ``tsne_warning``.

        Notes
        -----
        Rows with missing or infinite feature values are dropped for matrix
        diagnostics.  PCA and t-SNE use standardized features by default.

        Examples
        --------
        >>> eda.feature_relation_report(["x1", "x2", "x3"], plot=False)
        """

        return feature_relation_report(
            self.df,
            features,
            method=method,
            scale=scale,
            max_rows=max_rows,
            plot=plot,
            run_tsne=run_tsne,
        )


if __name__ == "__main__":
    idx = pd.date_range("2024-01-01", periods=500, freq="min")
    rng = np.random.default_rng(42)
    demo = pd.DataFrame(
        {
            "datetime": idx,
            "close": 100 * np.exp(np.cumsum(rng.normal(0, 0.001, len(idx)))),
            "volume": rng.lognormal(2, 0.5, len(idx)),
        }
    )
    demo_eda = EDA(demo, time_col="datetime", price_col="close")
    print(demo_eda.target_selection([1, 5, 10], cost=0.0005, rolling_window=50, plot=False)["summary"])
