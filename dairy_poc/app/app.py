"""
Dairy Process Investigation PoC — Streamlit app skeleton.

Run from the dairy_poc/ directory:
    streamlit run app/app.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

try:
    import sklearn  # noqa: F401
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False

# ── Paths ─────────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).parents[1]
_RAW  = _ROOT / "data_raw"
_PROC = _ROOT / "data_processed"

# ── App config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Dairy Process Investigation",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================================
# DATA LOADING  (cached so files are read once per session)
# =============================================================================

@st.cache_data
def load_runs() -> pd.DataFrame:
    return pd.read_csv(_RAW / "runs.csv")


@st.cache_data
def load_timeseries() -> pd.DataFrame:
    return pd.read_csv(_RAW / "timeseries.csv", parse_dates=["timestamp"])


@st.cache_data
def load_lab() -> pd.DataFrame:
    return pd.read_csv(_RAW / "lab_results.csv")


@st.cache_data
def load_events() -> pd.DataFrame:
    return pd.read_csv(_RAW / "events.csv", parse_dates=["timestamp"])


@st.cache_data
def load_demo_cases() -> dict:
    with open(_PROC / "demo_cases.json", encoding="utf-8") as fh:
        return json.load(fh)


# =============================================================================
# SIGNAL CHARTS
# =============================================================================

# (col_name, y-axis label, unit, line colour)
_QUARK_SIGNALS: list[tuple[str, str, str, str]] = [
    ("pH",                  "pH",              "–",    "#2196F3"),
    ("temperature_C",       "Temperature",     "°C",   "#FF5722"),
    ("separator_speed_rpm", "Centrifuge speed","rpm",  "#9C27B0"),
    ("separation_deltaP",   "Sep. ΔP",         "bar",  "#FF9800"),
]

_PUDDING_SIGNALS: list[tuple[str, str, str, str]] = [
    ("temperature_C",         "Temperature", "°C",    "#FF5722"),
    ("pressure_bar",          "Pressure",    "bar",   "#607D8B"),
    ("deltaT_heat_exchanger", "HX ΔT",       "°C",    "#F44336"),
    ("flow_rate_lpm",         "Flow rate",   "L/min", "#4CAF50"),
]

# Event types to show per product (others are filtered out to reduce clutter)
_QUARK_EVENTS: frozenset[str] = frozenset({
    "inoculation", "rennet_addition", "gel_break_start",
    "separation_start", "standardization_start", "filling_start",
})
_PUDDING_EVENTS: frozenset[str] = frozenset({
    "heating_start", "holding_start", "cooling_start",
    "filling_start", "CIP", "maintenance",
})

# Repeating color palette for step bands in the process timeline
_STEP_PALETTE: list[str] = [
    "#BBDEFB", "#C8E6C9", "#FFE0B2", "#F8BBD9",
    "#E1BEE7", "#B2EBF2", "#FFF9C4", "#FFCCBC",
    "#D7CCC8", "#CFD8DC",
]

# Short display labels for annotation text on the chart
_EVENT_SHORT: dict[str, str] = {
    "inoculation":           "Inoc.",
    "rennet_addition":       "Rennet",
    "gel_break_start":       "Gel brk",
    "separation_start":      "Sep.",
    "standardization_start": "Std.",
    "filling_start":         "Fill",
    "heating_start":         "Heat",
    "holding_start":         "Hold",
    "cooling_start":         "Cool",
    "CIP":                   "CIP",
    "maintenance":           "Maint.",
}


def _find_baseline_run_id(
    runs:    pd.DataFrame,
    product: str,
    scale:   str,
) -> str | None:
    """Return the run_id of a NORMAL run for the same product.

    Prefers the same scale; falls back to any NORMAL run for the product.
    """
    normals = runs[(runs["product"] == product) & (runs["scenario"] == "NORMAL")]
    if normals.empty:
        return None
    same_scale = normals[normals["scale"] == scale]
    pool = same_scale if not same_scale.empty else normals
    return str(pool.iloc[0]["run_id"])


def _compute_step_windows(
    run_ts: pd.DataFrame,
) -> list[tuple[str, float, float]]:
    """Return (step, start_t_min, end_t_min) for each step in first-appearance order."""
    if run_ts.empty:
        return []
    first_t = run_ts.groupby("step")["t_min"].min()
    ordered = first_t.sort_values().index.tolist()
    bounds  = run_ts.groupby("step")["t_min"].agg(start="min", end="max")
    return [
        (step, float(bounds.loc[step, "start"]), float(bounds.loc[step, "end"]))
        for step in ordered
    ]


def _add_event_markers(
    fig:    go.Figure,
    evts:   pd.DataFrame,
    col:    str,
    df_run: pd.DataFrame,
) -> None:
    """Overlay one vertical marker per event on an existing figure.

    Each marker is a dotted grey vline with a short annotation label and an
    invisible scatter point at the top of the signal range that carries a
    hover tooltip with the full event_type name and t_min.
    """
    if evts.empty:
        return

    # Hover target sits at the 97th-percentile of the signal to avoid being
    # pushed off-chart by SENSOR_FAULT spikes while staying near the top.
    valid = df_run[col].dropna()
    y_top = float(valid.quantile(0.97)) if not valid.empty else 1.0

    for _, row in evts.sort_values("t_min").iterrows():
        t     = float(row["t_min"])
        etype = str(row["event_type"])
        short = _EVENT_SHORT.get(etype, etype[:7])

        fig.add_vline(
            x=t,
            line_dash="dot",
            line_color="#888888",
            line_width=1.0,
            annotation_text=short,
            annotation_position="top right",
            annotation_font_size=8,
            annotation_font_color="#555555",
        )
        # Invisible marker — sole purpose is a richer hover tooltip
        fig.add_trace(go.Scatter(
            x=[t],
            y=[y_top],
            mode="markers",
            marker=dict(size=7, symbol="triangle-down", color="#888888", opacity=0.6),
            hovertemplate=(
                f"<b>{etype}</b><br>t = {t:.1f} min<extra></extra>"
            ),
            showlegend=False,
        ))


def _make_signal_chart(
    df_run:      pd.DataFrame,
    col:         str,
    label:       str,
    unit:        str,
    color:       str,
    show_xaxis:  bool = False,
    evts:        pd.DataFrame | None = None,
    x_range:     tuple[float, float] | None = None,
    df_baseline: pd.DataFrame | None = None,
) -> go.Figure:
    """Return a single compact Plotly line chart for one sensor signal."""
    has_baseline = (
        df_baseline is not None
        and col in df_baseline.columns
        and not df_baseline[col].isna().all()
    )

    fig = go.Figure()

    # Baseline drawn first so it sits visually behind the selected run
    if has_baseline:
        fig.add_trace(go.Scatter(
            x=df_baseline["t_min"],
            y=df_baseline[col],
            mode="lines",
            line=dict(color=color, width=1.4, dash="dash"),
            opacity=0.45,
            name="Baseline (NORMAL)",
            connectgaps=False,
        ))

    fig.add_trace(go.Scatter(
        x=df_run["t_min"],
        y=df_run[col],
        mode="lines",
        line=dict(color=color, width=1.8),
        name="Selected run",
        connectgaps=False,
    ))

    fig.update_layout(
        height=185,
        margin=dict(l=8, r=8, t=6, b=36 if show_xaxis else 6),
        xaxis=dict(
            title="t (min)" if show_xaxis else None,
            showticklabels=show_xaxis,
            showgrid=True,
            gridcolor="#ebebeb",
            zeroline=False,
        ),
        yaxis=dict(
            title=f"{label} ({unit})",
            showgrid=True,
            gridcolor="#ebebeb",
            zeroline=False,
        ),
        showlegend=has_baseline,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0,
            font=dict(size=9),
            bgcolor="rgba(255,255,255,0.75)",
            bordercolor="#dddddd",
            borderwidth=1,
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    if x_range is not None:
        fig.update_xaxes(range=list(x_range))
    if evts is not None and not evts.empty:
        _add_event_markers(fig, evts, col, df_run)
    return fig


def render_signal_charts(
    df_run:   pd.DataFrame,
    product:  str,
    run_evts: pd.DataFrame,
    runs:     pd.DataFrame,
    ts:       pd.DataFrame,
) -> None:
    """Render one Plotly line chart per signal, stacked vertically."""
    signals = _QUARK_SIGNALS if product == "QUARK" else _PUDDING_SIGNALS

    renderable = [
        sig for sig in signals
        if sig[0] in df_run.columns and not df_run[sig[0]].isna().all()
    ]

    st.subheader("Sensor charts")

    if not renderable:
        st.info("No signal data available for this run.")
        return

    # ── Step selector ─────────────────────────────────────────────────────────
    windows    = _compute_step_windows(df_run)
    run_id_key = df_run["run_id"].iloc[0]
    step_opts  = ["full_run"] + [step for step, _, _ in windows]

    sel_step = st.radio(
        "Step zoom",
        options=step_opts,
        format_func=lambda s: "Full run" if s == "full_run"
                              else s.replace("_", " ").capitalize(),
        horizontal=True,
        key=f"step_zoom_{run_id_key}",
        label_visibility="collapsed",
    )

    # ── Baseline toggle ────────────────────────────────────────────────────────
    show_baseline = st.checkbox(
        "Compare to baseline (NORMAL)",
        key=f"baseline_{run_id_key}",
    )

    df_baseline: pd.DataFrame | None = None
    if show_baseline:
        run_meta = runs[runs["run_id"] == run_id_key]
        scale    = str(run_meta["scale"].iloc[0]) if not run_meta.empty else "PRODUCTION"
        baseline_run_id = _find_baseline_run_id(runs, product, scale)
        if baseline_run_id:
            df_baseline = ts[ts["run_id"] == baseline_run_id].copy()
        else:
            st.caption("No NORMAL baseline found for this product.")

    # ── X-range for selected step (1-min padding each side) ───────────────────
    if sel_step == "full_run":
        x_range: tuple[float, float] | None = None
    else:
        win_map        = {step: (s, e) for step, s, e in windows}
        step_s, step_e = win_map[sel_step]
        x_range        = (step_s - 1.0, step_e + 1.0)

    # ── Event filtering — keep only events visible in the current x-range ─────
    evt_filter   = _QUARK_EVENTS if product == "QUARK" else _PUDDING_EVENTS
    evts_to_show = run_evts[run_evts["event_type"].isin(evt_filter)]

    if x_range is not None:
        evts_to_show = evts_to_show[
            (evts_to_show["t_min"] >= x_range[0]) &
            (evts_to_show["t_min"] <= x_range[1])
        ]

    # ── Chart loop — same x_range and baseline applied to every panel ─────────
    for i, (col, label, unit, color) in enumerate(renderable):
        is_last = (i == len(renderable) - 1)
        fig = _make_signal_chart(
            df_run, col, label, unit, color,
            show_xaxis=is_last,
            evts=evts_to_show,
            x_range=x_range,
            df_baseline=df_baseline,
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# =============================================================================
# ML EARLY WARNING  (IsolationForest, trained on NORMAL runs at startup)
# =============================================================================

_ML_WINDOW:        int   = 10    # rolling window for slope/std features (minutes)
_ML_THRESHOLD_PCT: float = 95.0  # percentile of NORMAL scores used as alert level
_ML_WATCH_THRESH:  float = 0.90  # run-level model score: Watch starts here (calibrated)
_ML_HIGH_THRESH:   float = 1.00  # run-level model score: High = at/above 95th-pct NORMAL
_ML_TREND_WINDOW:  int   = 30    # sliding window width in minutes for the score trend chart
_ML_TREND_STEP:    int   = 5     # step between successive windows in minutes

# Product-aware cosine-similarity weights: [mean, std, range, slope] × 4 signals (16 values)
# QUARK signal order: pH, temperature_C, separator_speed_rpm, separation_deltaP
_QUARK_SIM_WEIGHTS: np.ndarray = np.array([
    2.0, 2.0, 2.0, 2.0,   # pH — primary fermentation quality indicator
    1.5, 1.5, 1.5, 1.5,   # temperature_C — process control
    0.5, 0.5, 0.5, 0.5,   # separator_speed_rpm — separation step only
    0.5, 0.5, 0.5, 0.5,   # separation_deltaP — separation step only
])

# PUDDING signal order: temperature_C, pressure_bar, deltaT_heat_exchanger, flow_rate_lpm
_PUDDING_SIM_WEIGHTS: np.ndarray = np.array([
    1.5, 1.5, 1.5, 1.5,   # temperature_C
    2.0, 2.0, 2.0, 2.0,   # pressure_bar — fouling-sensitive
    2.0, 2.0, 2.0, 2.0,   # deltaT_heat_exchanger — primary fouling signal
    1.5, 1.5, 1.5, 1.5,   # flow_rate_lpm
])


def _extract_run_summary(
    run_ts:  pd.DataFrame,
    signals: list[tuple[str, str, str, str]],
) -> list[float]:
    """One fixed-length feature vector per run: [mean, std, range, slope] per signal.

    Used to train and score the run-level IsolationForest.
    All four statistics are 0.0 when a signal has fewer than 3 valid rows.
    """
    feats: list[float] = []
    for col, *_ in signals:
        vals = run_ts[col].dropna() if col in run_ts.columns else pd.Series(dtype=float)
        t    = run_ts.loc[vals.index, "t_min"].values.astype(float) if not vals.empty else np.array([])
        v    = vals.values.astype(float)
        if len(v) >= 3:
            feats += [
                float(v.mean()),
                float(v.std()),
                float(v.max() - v.min()),
                float(np.polyfit(t, v, 1)[0]),
            ]
        else:
            feats += [0.0, 0.0, 0.0, 0.0]
    return feats


def _extract_ml_features(
    df_ts:   pd.DataFrame,
    signals: list[tuple[str, str, str, str]],
    window:  int = _ML_WINDOW,
) -> pd.DataFrame:
    """Raw value + rolling slope + rolling std per signal.

    Slope is approximated as (y[t] − y[t−window+1]) / (window−1), which is
    O(n) vectorised and sufficient for anomaly detection.  NaN values (signals
    inactive in the current step) are left as NaN and imputed by the pipeline.
    """
    cols: dict = {}
    for col, *_ in signals:
        if col not in df_ts.columns:
            continue
        s = df_ts[col]
        cols[col]            = s
        cols[f"{col}_slope"] = (s - s.shift(window - 1)) / (window - 1)
        cols[f"{col}_rstd"]  = s.rolling(window, min_periods=2).std()
    return pd.DataFrame(cols, index=df_ts.index)


@st.cache_resource
def _load_ml_models() -> dict:
    """Train one IsolationForest per product on all NORMAL runs.

    Uses @st.cache_resource so training runs once per app-server start.
    Restart the app to pick up new training data.
    Returns {} when scikit-learn is unavailable or data is insufficient.
    """
    if not _SKLEARN_AVAILABLE:
        return {}

    from sklearn.ensemble import IsolationForest
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    runs_df = pd.read_csv(_RAW / "runs.csv")
    ts_df   = pd.read_csv(_RAW / "timeseries.csv")

    models: dict = {}
    for product, signals in [
        ("QUARK",                _QUARK_SIGNALS),
        ("HIGH_PROTEIN_PUDDING", _PUDDING_SIGNALS),
    ]:
        normal_ids = runs_df.loc[
            (runs_df["product"] == product) & (runs_df["scenario"] == "NORMAL"), "run_id"
        ]
        normal_ts = ts_df[ts_df["run_id"].isin(normal_ids)]
        if normal_ts.empty:
            continue

        feat_df = _extract_ml_features(normal_ts, signals)
        X       = feat_df.values.astype(float)

        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
            ("scaler",  StandardScaler()),
            ("iso",     IsolationForest(
                n_estimators=100, contamination="auto", random_state=42,
            )),
        ])
        pipe.fit(X)

        # Alert threshold = Nth-percentile anomaly score on the training data
        normal_scores = (-pipe.score_samples(X)).clip(min=0)
        threshold     = float(np.percentile(normal_scores, _ML_THRESHOLD_PCT))

        models[product] = {"pipeline": pipe, "threshold": threshold}

        # --- Run-level model: one summary vector per normal run ---
        summaries = [
            _extract_run_summary(ts_df[ts_df["run_id"] == rid], signals)
            for rid in normal_ids
        ]
        if len(summaries) >= 5:
            Xr = np.array(summaries, dtype=float)
            run_pipe = Pipeline([
                ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
                ("scaler",  StandardScaler()),
                ("iso",     IsolationForest(
                    n_estimators=100, contamination="auto", random_state=42,
                )),
            ])
            run_pipe.fit(Xr)
            run_scores  = (-run_pipe.score_samples(Xr)).clip(min=0)
            run_thr     = float(np.percentile(run_scores, _ML_THRESHOLD_PCT))
            models[product]["run_pipeline"]  = run_pipe
            models[product]["run_threshold"] = run_thr

    return models


def _render_ml_warning(
    df_run:        pd.DataFrame,
    product:       str,
    x_range:       tuple[float, float] | None,
    selected_step: str,
    run_id:        str,
) -> None:
    """Toggle-gated anomaly score chart with first-exceedance marker.

    Scores the full run for proper rolling context, then clips the display to
    the current zoom window.  Does not imply causality.
    """
    show = st.checkbox(
        "Show ML-assisted early warning",
        key=f"ml_warning_{run_id}",
    )
    if not show:
        return

    if not _SKLEARN_AVAILABLE:
        st.warning(
            "scikit-learn is required for ML early warning.  "
            "Install it with:  "
            "`uv pip install scikit-learn --python .venv/Scripts/python.exe`"
        )
        return

    if df_run.empty:
        return

    models = _load_ml_models()
    if product not in models:
        st.caption("ML model not available for this product.")
        return

    m         = models[product]
    pipe      = m["pipeline"]
    threshold = m["threshold"]
    signals   = _QUARK_SIGNALS if product == "QUARK" else _PUDDING_SIGNALS

    # Score the full run (preserves rolling context at step boundaries)
    feat_df     = _extract_ml_features(df_run, signals)
    X           = feat_df.values.astype(float)
    raw_scores  = (-pipe.score_samples(X)).clip(min=0)
    # Normalise so the alert threshold sits at y = 1.0
    norm_scores = raw_scores / max(threshold, 1e-9)
    t_arr       = df_run["t_min"].values

    # Clip display to the current zoom window
    if x_range is not None:
        lo, hi = x_range
        mask = (t_arr >= lo) & (t_arr <= hi)
    else:
        mask = np.ones(len(t_arr), dtype=bool)

    t_vis     = t_arr[mask]
    score_vis = norm_scores[mask]
    if len(t_vis) == 0:
        return

    # First timepoint in the visible window where score exceeds the threshold
    exceed          = score_vis > 1.0
    first_t: float | None = float(t_vis[exceed][0]) if exceed.any() else None
    first_step_lbl: str | None = None
    if first_t is not None:
        sr = df_run[(df_run["t_min"] - first_t).abs() < 0.5]
        if not sr.empty:
            first_step_lbl = sr["step"].iloc[0].replace("_", " ")

    # ── Chart ─────────────────────────────────────────────────────────────────
    st.subheader("ML-assisted early warning")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=t_vis, y=score_vis,
        mode="lines",
        line=dict(color="#FF9800", width=1.5),
        fill="tozeroy",
        fillcolor="rgba(255,152,0,0.08)",
        hovertemplate="t = %{x:.1f} min<br>Score = %{y:.2f}× threshold<extra></extra>",
        showlegend=False,
    ))
    fig.add_hline(
        y=1.0,
        line_dash="dash", line_color="#E53935", line_width=1.5,
        annotation_text="Alert threshold",
        annotation_position="top right",
        annotation_font_size=9, annotation_font_color="#E53935",
    )
    if first_t is not None:
        fig.add_vline(
            x=first_t, line_dash="dot", line_color="#E53935", line_width=1.5,
        )
        fig.add_annotation(
            x=first_t,
            y=max(float(score_vis.max()), 1.15),
            text=f"t = {first_t:.0f}",
            showarrow=False,
            font=dict(size=9, color="#E53935"),
            xanchor="left",
        )
    fig.update_layout(
        height=135,
        margin=dict(l=8, r=8, t=4, b=36),
        xaxis=dict(title="t (min)", showgrid=True, gridcolor="#ebebeb", zeroline=False),
        yaxis=dict(
            title="Score (× threshold)",
            showgrid=True, gridcolor="#ebebeb", zeroline=False,
            range=[0, max(float(score_vis.max()) * 1.1, 1.5)],
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # ── Explanatory text ──────────────────────────────────────────────────────
    if first_t is not None:
        step_clause = f" (step: *{first_step_lbl}*)" if first_step_lbl else ""
        st.warning(
            f"ML-assisted early warning: unusual pattern begins around "
            f"**t = {first_t:.0f} min**{step_clause}.  "
            f"This does not imply a root cause — it indicates the sensor-signal "
            f"combination is unusual relative to NORMAL runs."
        )
    else:
        st.success("No anomaly detected above the alert threshold in this window.")

    st.caption(
        "Model: IsolationForest trained on NORMAL runs only (one model per product).  "
        "Score > 1.0 = sensor pattern unusual vs reference behaviour.  "
        "Observational signal only — not a diagnosis."
    )


def _compute_score_trend(
    t_arr:      np.ndarray,
    row_scores: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Sliding-window mean of normalised row-level scores.

    For each window ending at time t (stepping by _ML_TREND_STEP minutes),
    takes the mean of row_scores for all rows with t_min in [t-window, t].
    Returns (t_ends, window_means).  Windows with < 3 rows are skipped.
    Always computed on the full run; callers clip for display.
    """
    if len(t_arr) == 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    t_start = float(t_arr.min())
    t_end   = float(t_arr.max())
    t_ends: list[float] = []
    scores:  list[float] = []

    tc = t_start + _ML_TREND_WINDOW
    while True:
        tc = min(tc, t_end)
        mask = (t_arr >= tc - _ML_TREND_WINDOW) & (t_arr <= tc)
        if mask.sum() >= 3:
            t_ends.append(tc)
            scores.append(float(row_scores[mask].mean()))
        if tc >= t_end:
            break
        tc += _ML_TREND_STEP

    return np.array(t_ends, dtype=float), np.array(scores, dtype=float)


def _compute_ml_run_score(
    df_run:  pd.DataFrame,
    product: str,
) -> tuple[float, float] | None:
    """Single normalised run-level anomaly score using per-run summary features.

    The run-level IsolationForest is trained on [mean, std, range, slope] per signal
    across the full run — one vector per normal run.  This avoids the aggregation
    instability of per-row scores.

    Returns (normalized_score, run_threshold).  Score >= 1.0 means the run profile
    is as unusual as the 95th-pct normal run.
    Returns None when sklearn is unavailable or the product has no run-level model.
    """
    if not _SKLEARN_AVAILABLE:
        return None
    models = _load_ml_models()
    m = models.get(product)
    if m is None or "run_pipeline" not in m:
        return None
    signals = _QUARK_SIGNALS if product == "QUARK" else _PUDDING_SIGNALS
    feats   = np.array([_extract_run_summary(df_run, signals)], dtype=float)
    raw     = float((-m["run_pipeline"].score_samples(feats)).clip(min=0)[0])
    norm    = raw / max(float(m["run_threshold"]), 1e-9)
    return norm, float(m["run_threshold"])


def _compute_ml_row_scores(
    df_ts:   pd.DataFrame,
    product: str,
) -> np.ndarray | None:
    """Per-row normalised scores from the row-level model.  Used for step drill-down.

    Returns a numpy array aligned with df_ts rows, or None if unavailable.
    """
    if not _SKLEARN_AVAILABLE:
        return None
    models = _load_ml_models()
    m = models.get(product)
    if m is None:
        return None
    signals = _QUARK_SIGNALS if product == "QUARK" else _PUDDING_SIGNALS
    feat_df = _extract_ml_features(df_ts, signals)
    X       = feat_df.values.astype(float)
    raw     = (-m["pipeline"].score_samples(X)).clip(min=0)
    return raw / max(float(m["threshold"]), 1e-9)


def _render_ml_score_panel(
    df_run:        pd.DataFrame,
    product:       str,
    x_range:       tuple[float, float] | None,
    selected_step: str,
    run_id:        str,
) -> None:
    """Always-visible ML Early Warning summary panel, rendered above signal charts.

    Uses a run-level IsolationForest (per-run summary features) for the main score.
    When a step is zoomed, also shows the mean row-level anomaly score for that step.
    """
    st.subheader("ML Early Warning")

    if not _SKLEARN_AVAILABLE:
        st.caption("scikit-learn not installed — ML early warning unavailable.")
        return

    if df_run.empty:
        return

    result = _compute_ml_run_score(df_run, product)
    if result is None:
        st.caption("ML model not available for this product.")
        return

    run_score, run_threshold = result

    if run_score < _ML_WATCH_THRESH:
        status    = "Normal"
        flag_text = "✅ Normal"
        show_flag = st.success
    elif run_score < _ML_HIGH_THRESH:
        status    = "Watch"
        flag_text = "⚠️ Watch"
        show_flag = st.warning
    else:
        status    = "High"
        flag_text = "🚨 High Risk"
        show_flag = st.error

    with st.container(border=True):
        c1, c2 = st.columns(2)
        c1.metric("Anomaly score", f"{run_score:.2f}")
        c2.metric("Status",        status)
        show_flag(flag_text)
        st.caption(
            f"Threshold (Watch): {_ML_WATCH_THRESH:.2f}  ·  "
            f"Threshold (High Risk): {_ML_HIGH_THRESH:.2f}  ·  "
            f"Training alert level (95th-pct NORMAL): {run_threshold:.4f}  ·  "
            f"Level 1 indicative signal — not a definitive alarm"
        )

        # Step-level score: mean row-level anomaly score for the zoomed step
        if selected_step != "full_run" and "step" in df_run.columns:
            row_scores = _compute_ml_row_scores(df_run, product)
            if row_scores is not None:
                step_mask = df_run["step"].values == selected_step
                if step_mask.any():
                    step_score = float(row_scores[step_mask].mean())
                    step_label = selected_step.replace("_", " ").capitalize()
                    st.caption(
                        f"Selected step score ({step_label}): {step_score:.2f}"
                        f"  (mean row-level score)"
                    )


def _render_score_trend_chart(
    df_run:        pd.DataFrame,
    product:       str,
    run_evts:      pd.DataFrame,
    x_range:       tuple[float, float] | None,
    selected_step: str,
    run_id:        str,
) -> None:
    """Sliding-window anomaly score trend chart.

    Shows mean row-level anomaly score in a _ML_TREND_WINDOW-minute rolling
    window, stepping every _ML_TREND_STEP minutes.  The full run is always
    scored; the display clips to x_range when a step is zoomed.

    The computed trend is cached in session_state (keyed by run_id) so it
    survives Streamlit reruns triggered by other widget interactions.
    """
    if not _SKLEARN_AVAILABLE or df_run.empty:
        return

    # Cache full-run trend — keyed by run_id only so zoom doesn't retrigger
    cache_key = f"score_trend_{run_id}"
    if cache_key not in st.session_state:
        row_scores = _compute_ml_row_scores(df_run, product)
        if row_scores is None:
            return
        t_all, s_all = _compute_score_trend(df_run["t_min"].values, row_scores)
        st.session_state[cache_key] = (t_all, s_all)

    t_all, s_all = st.session_state[cache_key]
    if len(t_all) == 0:
        return

    # Clip to zoom window for display
    if x_range is not None:
        lo, hi  = x_range
        vis_mask = (t_all >= lo) & (t_all <= hi)
        t_vis, s_vis = t_all[vis_mask], s_all[vis_mask]
    else:
        t_vis, s_vis = t_all, s_all

    if len(t_vis) == 0:
        return

    # First crossings within the visible window
    first_watch = next((float(t) for t, s in zip(t_vis, s_vis) if s >= _ML_WATCH_THRESH), None)
    first_high  = next((float(t) for t, s in zip(t_vis, s_vis) if s >= _ML_HIGH_THRESH),  None)

    # y-axis ceiling: 10% above the taller of the peak score and the High threshold,
    # plus 0.15 absolute so threshold labels drawn at "top right" never touch the axis edge.
    y_max = max(float(s_vis.max()), _ML_HIGH_THRESH) * 1.1 + 0.15

    # ── Figure ────────────────────────────────────────────────────────────────
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=t_vis, y=s_vis,
        mode="lines",
        line=dict(color="#FF6F00", width=2.0),
        fill="tozeroy",
        fillcolor="rgba(255,111,0,0.08)",
        hovertemplate="t = %{x:.1f} min<br>Window score = %{y:.3f}<extra></extra>",
        showlegend=False,
    ))

    # Threshold reference lines — labels go into the right margin (r=100)
    fig.add_hline(
        y=_ML_WATCH_THRESH,
        line_dash="dash", line_color="#FF9800", line_width=1.2,
        annotation_text=f"Watch ({_ML_WATCH_THRESH:.2f})",
        annotation_position="top right",
        annotation_font_size=8, annotation_font_color="#FF9800",
    )
    fig.add_hline(
        y=_ML_HIGH_THRESH,
        line_dash="dash", line_color="#E53935", line_width=1.5,
        annotation_text=f"High ({_ML_HIGH_THRESH:.2f})",
        annotation_position="top right",
        annotation_font_size=8, annotation_font_color="#E53935",
    )

    # First-warning annotation — High takes precedence over Watch.
    # Anchor in paper coordinates (yref="paper") so the text stays in the
    # visible area regardless of the data scale; ax/ay are pixel offsets.
    first_warn = first_high if first_high is not None else first_watch
    if first_warn is not None:
        fig.add_vline(x=first_warn, line_dash="dot", line_color="#E53935", line_width=1.5)
        near = df_run[(df_run["t_min"] - first_warn).abs() < (_ML_TREND_STEP + 1.0)]
        step_lbl = near["step"].iloc[0].replace("_", " ") if not near.empty else ""
        ann = f"first warning{' · ' + step_lbl if step_lbl else ''}"
        fig.add_annotation(
            x=first_warn,
            xref="x",
            y=0.88,
            yref="paper",
            text=ann,
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=1,
            arrowcolor="#E53935",
            ax=12,
            ay=-28,
            axref="pixel",
            ayref="pixel",
            font=dict(size=9, color="#E53935"),
            bgcolor="rgba(255,255,255,0.82)",
        )

    # Event markers — same filter and style as signal charts
    evt_filter   = _QUARK_EVENTS if product == "QUARK" else _PUDDING_EVENTS
    evts_visible = run_evts[run_evts["event_type"].isin(evt_filter)]
    if x_range is not None:
        evts_visible = evts_visible[
            (evts_visible["t_min"] >= x_range[0]) & (evts_visible["t_min"] <= x_range[1])
        ]
    y_evt = float(np.percentile(s_vis, 90)) if len(s_vis) >= 2 else _ML_WATCH_THRESH
    for _, erow in evts_visible.iterrows():
        t_e   = float(erow["t_min"])
        etype = str(erow["event_type"])
        short = _EVENT_SHORT.get(etype, etype[:7])
        fig.add_vline(
            x=t_e, line_dash="dot", line_color="#888888", line_width=1.0,
            annotation_text=short,
            annotation_position="top right",
            annotation_font_size=8, annotation_font_color="#555555",
        )
        fig.add_trace(go.Scatter(
            x=[t_e], y=[y_evt],
            mode="markers",
            marker=dict(size=7, symbol="triangle-down", color="#888888", opacity=0.6),
            hovertemplate=f"<b>{etype}</b><br>t = {t_e:.1f} min<extra></extra>",
            showlegend=False,
        ))

    fig.update_layout(
        height=290,
        margin=dict(l=62, r=100, t=50, b=40),
        xaxis=dict(title="t (min)", showgrid=True, gridcolor="#ebebeb", zeroline=False),
        yaxis=dict(
            title=f"Score ({_ML_TREND_WINDOW}-min window)",
            showgrid=True, gridcolor="#ebebeb", zeroline=False,
            range=[0, y_max],
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=False,
    )
    if x_range is not None:
        fig.update_xaxes(range=list(x_range))

    st.markdown("**Early Warning Score trend**")
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    if first_warn is not None:
        cross_type = "High Risk" if first_high is not None else "Watch"
        st.caption(
            f"Indicative early warning: score crosses {cross_type} threshold around "
            f"t = {first_warn:.0f} min.  "
            f"Observational signal only — not a definitive alarm."
        )
    else:
        st.caption(
            f"Score remains below Watch threshold in this window.  "
            f"Indicative signal only — not a definitive alarm."
        )

    if product == "QUARK":
        st.caption(
            "QUARK note: separation-step signals (centrifuge speed, ΔP) activate from zero "
            "at the fermentation→separation transition, which consistently elevates the "
            "row-level score for all runs — including normal ones.  "
            "Use the trend shape and timing to compare runs; "
            "absolute threshold crossings are unreliable for QUARK in this view."
        )


# =============================================================================
# SIMILAR RUNS  (cosine similarity on product-aware run feature vectors)
# =============================================================================

@st.cache_data
def _build_similarity_vectors(
    _ts:   pd.DataFrame,
    _runs: pd.DataFrame,
) -> dict[str, np.ndarray]:
    """Weighted feature vector per run_id for cosine similarity.

    Reuses _extract_run_summary (16-dim: [mean, std, range, slope] × 4 signals)
    then applies product-specific signal weights so fermentation signals dominate
    for QUARK and heat-exchanger signals dominate for PUDDING.

    Returns {} when all runs lack timeseries data.
    """
    result: dict[str, np.ndarray] = {}
    for product, signals, weights in [
        ("QUARK",                _QUARK_SIGNALS,  _QUARK_SIM_WEIGHTS),
        ("HIGH_PROTEIN_PUDDING", _PUDDING_SIGNALS, _PUDDING_SIM_WEIGHTS),
    ]:
        prod_run_ids = _runs.loc[_runs["product"] == product, "run_id"].tolist()
        for rid in prod_run_ids:
            run_ts = _ts[_ts["run_id"] == rid]
            if run_ts.empty:
                continue
            feats = np.array(_extract_run_summary(run_ts, signals), dtype=float)
            feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
            result[rid] = feats * weights
    return result


def _compute_similar_runs(
    target_run_id: str,
    runs:          pd.DataFrame,
    sim_vectors:   dict[str, np.ndarray],
    top_n:         int = 5,
) -> list[dict]:
    """Top-N most similar runs by cosine similarity, same product only.

    Adds a 0.05 boost to runs on the same scale before ranking so that
    same-scale runs are preferred when similarity scores are close.
    Caps displayed similarity at 1.0 after the boost.
    """
    if target_run_id not in sim_vectors:
        return []
    target_vec = sim_vectors[target_run_id]
    norm_t = float(np.linalg.norm(target_vec))

    target_meta = runs[runs["run_id"] == target_run_id]
    if target_meta.empty:
        return []
    target_meta    = target_meta.iloc[0]
    target_product = target_meta["product"]
    target_scale   = target_meta["scale"]

    results: list[dict] = []
    for rid, vec in sim_vectors.items():
        if rid == target_run_id:
            continue
        run_meta = runs[runs["run_id"] == rid]
        if run_meta.empty or run_meta.iloc[0]["product"] != target_product:
            continue
        run_meta = run_meta.iloc[0]

        norm_v = float(np.linalg.norm(vec))
        sim = float(np.dot(target_vec, vec) / (norm_t * norm_v)) if (norm_t > 1e-9 and norm_v > 1e-9) else 0.0
        if run_meta["scale"] == target_scale:
            sim += 0.05

        results.append({
            "run_id":           rid,
            "scenario":         run_meta["scenario"],
            "scale":            run_meta["scale"],
            "similarity":       min(sim, 1.0),
            "downtime_minutes": float(run_meta["downtime_minutes"]),
            "yield_loss_pct":   float(run_meta["yield_loss_pct"]),
            "extra_cleaning":   bool(run_meta["extra_cleaning"]),
            "fouling_grade":    str(run_meta["fouling_grade"]),
        })

    results.sort(key=lambda x: x["similarity"], reverse=True)
    return results[:top_n]


def _on_open_similar_run(
    target_run_id:  str,
    target_product: str,
    current_mode:   str,
    demo_cases:     dict,
) -> None:
    """on_click callback for the 'Open' button in the Similar Runs panel.

    Guided mode + run in demo_cases.json → stay guided, jump to that story.
    All other cases → switch to Explore mode for the target run.
    """
    stories   = demo_cases.get("stories", [])
    story_ids = [s["run_id"] for s in stories]
    _clear_run_widgets(target_run_id)

    if current_mode == "guided" and target_run_id in story_ids:
        st.session_state["guided_idx"] = story_ids.index(target_run_id)
    else:
        st.session_state["app_mode"]          = "Explore all runs"
        st.session_state["explore_product"]   = target_product
        st.session_state["explore_scenario"]  = "(all)"
        st.session_state["explore_run"]       = target_run_id


def _render_similar_runs_panel(
    run_id:     str,
    mode:       str,
    runs:       pd.DataFrame,
    ts:         pd.DataFrame,
    demo_cases: dict,
) -> None:
    """Cosine-similarity case-based reasoning panel in the right column."""
    st.subheader("Similar runs")

    sim_vectors = _build_similarity_vectors(ts, runs)
    if not sim_vectors:
        st.caption("Similarity data unavailable.")
        return

    similar = _compute_similar_runs(run_id, runs, sim_vectors)
    if not similar:
        st.caption("No similar runs found for this product.")
        return

    target_product = runs.loc[runs["run_id"] == run_id, "product"].iloc[0]

    with st.container(border=True):
        for sr in similar:
            sim_pct = int(sr["similarity"] * 100)
            col_info, col_btn = st.columns([3, 1])
            with col_info:
                st.markdown(
                    f"**{sr['run_id']}** · {sr['scenario']} · {sr['scale']}  \n"
                    f"Similarity: **{sim_pct}%**  ·  "
                    f"Downtime: {sr['downtime_minutes']:.0f} min  ·  "
                    f"Yield loss: {sr['yield_loss_pct']:.1f}%"
                )
                badges: list[str] = []
                if sr["extra_cleaning"]:
                    badges.append("extra CIP")
                if sr["fouling_grade"] not in ("NONE", "nan", ""):
                    badges.append(f"fouling: {sr['fouling_grade']}")
                if badges:
                    st.caption("  ·  ".join(badges))
            with col_btn:
                st.button(
                    "Open",
                    key=f"open_sim_{run_id}_{sr['run_id']}",
                    on_click=_on_open_similar_run,
                    kwargs={
                        "target_run_id":  sr["run_id"],
                        "target_product": target_product,
                        "current_mode":   mode,
                        "demo_cases":     demo_cases,
                    },
                    use_container_width=True,
                )
            st.divider()

        st.caption(
            "Similarity is pattern-based (helps investigation); "
            "it does not imply root cause."
        )


# =============================================================================
# SIDEBAR — mode toggle + selectors
# =============================================================================

def render_sidebar(
    runs:       pd.DataFrame,
    demo_cases: dict,
) -> tuple[str, str | None]:
    """
    Render sidebar controls.

    Returns
    -------
    mode     : "guided" or "explore"
    run_id   : the selected run_id, or None if nothing is selectable yet
    """
    st.sidebar.title("Controls")

    mode = st.sidebar.radio(
        "Mode",
        options=["Guided story mode", "Explore all runs"],
        key="app_mode",
        help=(
            "Guided: step through curated demo cases with process narratives.\n"
            "Explore: browse any run in the dataset."
        ),
    )
    guided = mode == "Guided story mode"

    st.sidebar.divider()

    if guided:
        run_id = _sidebar_guided(demo_cases)
    else:
        run_id = _sidebar_explore(runs)

    return ("guided" if guided else "explore"), run_id


def _clear_run_widgets(run_id: str) -> None:
    """Remove per-run session state so the destination run starts fresh."""
    for key in (f"step_zoom_{run_id}", f"baseline_{run_id}", f"score_trend_{run_id}"):
        st.session_state.pop(key, None)


def _on_guided_nav(stories: list, new_idx: int) -> None:
    """Button callback: move to new_idx and reset the destination run's widgets."""
    st.session_state["guided_idx"] = new_idx
    _clear_run_widgets(stories[new_idx]["run_id"])


def _on_guided_select(stories: list) -> None:
    """Selectbox on_change callback: reset widgets for the newly selected run."""
    _clear_run_widgets(stories[st.session_state["guided_idx"]]["run_id"])


def _sidebar_guided(demo_cases: dict) -> str | None:
    """Story-mode sidebar: selectbox + Prev / Next navigation buttons."""
    stories = demo_cases.get("stories", [])
    if not stories:
        st.sidebar.warning("No demo cases found in demo_cases.json.")
        return None

    n = len(stories)

    # Initialise index on first visit (or after a hot-reload)
    if "guided_idx" not in st.session_state:
        st.session_state["guided_idx"] = 0

    labels = [f"{s['run_id']}  ·  {s['short_title']}" for s in stories]

    # Selectbox — key="guided_idx" makes session_state the single source of truth
    st.sidebar.selectbox(
        "Select story",
        options=range(n),
        format_func=lambda i: labels[i],
        label_visibility="collapsed",
        key="guided_idx",
        on_change=_on_guided_select,
        kwargs={"stories": stories},
    )

    idx = st.session_state["guided_idx"]

    # Prev / Next buttons
    col_prev, col_next = st.sidebar.columns(2)
    col_prev.button(
        "◀  Prev",
        disabled=(idx == 0),
        on_click=_on_guided_nav,
        kwargs={"stories": stories, "new_idx": idx - 1},
        use_container_width=True,
        key="guided_prev",
    )
    col_next.button(
        "Next  ▶",
        disabled=(idx == n - 1),
        on_click=_on_guided_nav,
        kwargs={"stories": stories, "new_idx": idx + 1},
        use_container_width=True,
        key="guided_next",
    )

    st.sidebar.caption(f"Story {idx + 1} of {n}")

    return stories[idx]["run_id"]


def _on_explore_product_change() -> None:
    """Clear downstream keys when the user manually changes the product."""
    st.session_state.pop("explore_scenario", None)
    st.session_state.pop("explore_run", None)


def _on_explore_scenario_change() -> None:
    """Clear run key when the user manually changes the scenario."""
    st.session_state.pop("explore_run", None)


def _sidebar_explore(runs: pd.DataFrame) -> str | None:
    """Explore-mode sidebar: cascading product → scenario → run selectors.

    All three selectboxes are keyed so that _on_open_similar_run can
    programmatically navigate by setting st.session_state before the rerun.
    on_change callbacks clear downstream keys when the user changes a selector
    manually, preventing stale values from causing invalid-option errors.
    """
    products = sorted(runs["product"].unique())
    product_display = {p: p.replace("HIGH_PROTEIN_PUDDING", "High-Protein Pudding") for p in products}

    sel_product = st.sidebar.selectbox(
        "Product",
        options=products,
        format_func=lambda p: product_display[p],
        key="explore_product",
        on_change=_on_explore_product_change,
    )

    filtered_by_product = runs[runs["product"] == sel_product]
    scenarios = ["(all)"] + sorted(filtered_by_product["scenario"].unique())

    # Guard: if session_state holds a scenario that isn't valid for the current
    # product (e.g., after a programmatic product switch), reset it.
    if st.session_state.get("explore_scenario") not in scenarios:
        st.session_state["explore_scenario"] = "(all)"

    sel_scenario = st.sidebar.selectbox(
        "Scenario",
        options=scenarios,
        key="explore_scenario",
        on_change=_on_explore_scenario_change,
    )

    if sel_scenario == "(all)":
        filtered = filtered_by_product
    else:
        filtered = filtered_by_product[filtered_by_product["scenario"] == sel_scenario]

    if filtered.empty:
        st.sidebar.info("No runs match the current filters.")
        return None

    run_list = list(filtered.sort_values("run_id")["run_id"])
    run_labels = {
        row["run_id"]: f"{row['run_id']}  ·  {row['scenario']}  ·  {row['scale']}"
        for _, row in filtered.sort_values("run_id").iterrows()
    }

    # Guard: if session_state holds a run that isn't in the filtered list reset it.
    if st.session_state.get("explore_run") not in run_list:
        st.session_state["explore_run"] = run_list[0]

    sel_run = st.sidebar.selectbox(
        "Run",
        options=run_list,
        format_func=lambda rid: run_labels[rid],
        key="explore_run",
    )
    return sel_run


# =============================================================================
# MAIN AREA
# =============================================================================

def render_main(
    mode:       str,
    run_id:     str | None,
    runs:       pd.DataFrame,
    ts:         pd.DataFrame,
    lab:        pd.DataFrame,
    events:     pd.DataFrame,
    demo_cases: dict,
) -> None:
    """Render the centre column and right column."""
    if run_id is None:
        st.info("Select a run from the sidebar to begin.")
        return

    # Fetch metadata for the selected run
    run_row = runs[runs["run_id"] == run_id]
    if run_row.empty:
        st.error(f"run_id {run_id!r} not found in runs.csv.")
        return
    run_row = run_row.iloc[0]

    lab_row = lab[lab["run_id"] == run_id]
    lab_row = lab_row.iloc[0] if not lab_row.empty else None

    story    = _find_story(demo_cases, run_id)       # None in explore mode
    run_ts   = ts[ts["run_id"] == run_id]            # filter once; shared by chart renders
    run_evts = events[events["run_id"] == run_id]    # events for this run

    # Read widget states from session_state before any widget is drawn.
    # Both the timeline (above the radio) and the divergence panel (right col)
    # need these values, so they must be resolved here.
    selected_step = st.session_state.get(f"step_zoom_{run_id}", "full_run")
    show_baseline = st.session_state.get(f"baseline_{run_id}", False)

    # Resolve baseline DataFrame (mirrors logic in render_signal_charts)
    df_baseline: pd.DataFrame | None = None
    if show_baseline:
        run_meta = runs[runs["run_id"] == run_id]
        scale    = str(run_meta["scale"].iloc[0]) if not run_meta.empty else "PRODUCTION"
        b_rid    = _find_baseline_run_id(runs, run_row["product"], scale)
        if b_rid:
            df_baseline = ts[ts["run_id"] == b_rid].copy()

    # Resolve x_range from step-zoom state (mirrors logic in render_signal_charts)
    windows = _compute_step_windows(run_ts)
    if selected_step == "full_run":
        x_range: tuple[float, float] | None = None
    else:
        win_map = {s: (s_t, e_t) for s, s_t, e_t in windows}
        if selected_step in win_map:
            step_s, step_e = win_map[selected_step]
            x_range = (step_s - 1.0, step_e + 1.0)
        else:
            x_range = None

    # Build two-column layout: main content | right panel
    col_main, col_right = st.columns([2, 1], gap="large")

    with col_main:
        _render_run_header(run_row, story)
        _render_process_timeline(run_ts, run_evts, selected_step)
        _render_ml_score_panel(run_ts, run_row["product"], x_range, selected_step, run_id)
        _render_score_trend_chart(run_ts, run_row["product"], run_evts, x_range, selected_step, run_id)
        render_signal_charts(run_ts, run_row["product"], run_evts, runs, ts)

    with col_right:
        _render_right_panel(mode, run_row, lab_row, story)
        if df_baseline is not None:
            _render_divergence_panel(
                run_ts, df_baseline, run_row["product"], x_range, selected_step,
            )
        _render_similar_runs_panel(run_id, mode, runs, ts, demo_cases)


# ── Run header ────────────────────────────────────────────────────────────────

def _render_run_header(run_row: pd.Series, story: dict | None) -> None:
    product_label = run_row["product"].replace("HIGH_PROTEIN_PUDDING", "High-Protein Pudding")

    title = story["short_title"] if story else f"{run_row['run_id']} — {run_row['scenario']}"
    st.header(title)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Run ID",   run_row["run_id"])
    c2.metric("Product",  product_label)
    c3.metric("Scenario", run_row["scenario"])
    c4.metric("Scale",    run_row["scale"])

    st.caption(
        f"Duration: {run_row['total_duration_hr']:.1f} h  ·  "
        f"Start: {run_row['run_start_ts']}  ·  "
        f"Batch: {run_row['batch_size_L']:.0f} L"
    )
    st.divider()


# ── Process timeline ──────────────────────────────────────────────────────────

def _render_process_timeline(
    run_ts:        pd.DataFrame,
    run_evts:      pd.DataFrame,
    selected_step: str,
) -> None:
    """Horizontal step-band timeline with event tick markers.

    Each step is a colored rectangle; width = duration in t_min.
    The currently zoomed step (selected_step) is highlighted; all others dim.
    Events appear as thin vertical ticks with hover tooltips.
    """
    st.subheader("Process timeline")

    if run_ts.empty:
        st.caption("No timeseries data for this run.")
        return

    windows = _compute_step_windows(run_ts)
    if not windows:
        return

    t_max      = float(run_ts["t_min"].max())
    label_min  = t_max * 0.045   # only label steps wider than 4.5% of full run

    fig = go.Figure()

    # ── Step bands ────────────────────────────────────────────────────────────
    for idx, (step, start, end) in enumerate(windows):
        color      = _STEP_PALETTE[idx % len(_STEP_PALETTE)]
        is_sel     = (selected_step == step)
        # dim non-selected when a specific step is zoomed
        opacity    = 1.0 if (selected_step == "full_run" or is_sel) else 0.30
        border_col = "#1a1a1a" if is_sel else "#aaaaaa"
        border_w   = 2.0       if is_sel else 0.5

        fig.add_shape(
            type="rect",
            x0=start, x1=end, y0=0, y1=1,
            fillcolor=color,
            opacity=opacity,
            line=dict(color=border_col, width=border_w),
            layer="below",
        )

        # Label: only when the band is wide enough to hold text
        if (end - start) >= label_min:
            raw   = step.replace("_", " ")
            label = raw if len(raw) <= 15 else raw[:13] + "…"
            fig.add_annotation(
                x=(start + end) / 2, y=0.5,
                text=label,
                showarrow=False,
                font=dict(size=9, color="#333333"),
                opacity=1.0 if (selected_step == "full_run" or is_sel) else 0.45,
                xanchor="center",
                yanchor="middle",
            )

    # ── Event tick markers ────────────────────────────────────────────────────
    if not run_evts.empty:
        for _, erow in run_evts.iterrows():
            fig.add_shape(
                type="line",
                x0=float(erow["t_min"]), x1=float(erow["t_min"]),
                y0=0, y1=1,
                line=dict(color="#333333", width=1.2),
                layer="above",
            )

        # Invisible scatter — one point per event, positioned above the band
        # for hover tooltips (shapes don't emit hover events in Plotly)
        fig.add_trace(go.Scatter(
            x=[float(r["t_min"])  for _, r in run_evts.iterrows()],
            y=[1.06]              * len(run_evts),
            mode="markers",
            marker=dict(symbol="triangle-down", size=7, color="#333333"),
            hovertemplate=[
                f"<b>{r['event_type']}</b><br>t = {r['t_min']:.1f} min<extra></extra>"
                for _, r in run_evts.iterrows()
            ],
            showlegend=False,
        ))

    fig.update_layout(
        height=95,
        margin=dict(l=8, r=8, t=2, b=26),
        xaxis=dict(
            range=[-t_max * 0.005, t_max * 1.01],
            showticklabels=True,
            ticksuffix=" min",
            showgrid=False,
            zeroline=False,
            title=None,
        ),
        yaxis=dict(
            range=[0, 1.18],      # headroom for the event triangle markers
            showticklabels=False,
            showgrid=False,
            zeroline=False,
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=False,
    )

    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# ── Baseline divergence panel ────────────────────────────────────────────────

# Divergence is flagged when a metric exceeds 1.5× the baseline noise floor.
_Z_THRESH: float = 1.5

# Process-step categories used for product-aware narrative language.
_QUARK_FERM_STEPS: frozenset[str] = frozenset({
    "inoculation_mixing", "early_fermentation", "late_fermentation", "gel_break_mixing",
})
_QUARK_SEP_STEPS: frozenset[str] = frozenset({
    "separation", "standardization_mix",
})
_PUDDING_THERMAL_STEPS: frozenset[str] = frozenset({
    "mixing", "heating", "holding",
})


def _residual_std(t: np.ndarray, y: np.ndarray) -> float:
    """Std of residuals after removing the linear trend within a step.

    Detrending is critical for signals like pH that change monotonically
    during fermentation: raw std would include trend variance and make real
    run-vs-baseline differences look small relative to the noise level.
    Falls back to raw std when fewer than 3 points are available.
    """
    if len(t) < 3:
        return float(y.std()) if len(y) > 1 else 0.001
    slope, intercept = np.polyfit(t, y, 1)
    return max(float((y - (slope * t + intercept)).std()), 0.001)


def _norm_denom(t: np.ndarray, y: np.ndarray) -> float:
    """Normalisation denominator: detrended residual std, floored at 1% of |mean|."""
    res_std  = _residual_std(t, y)
    mean_abs = abs(float(y.mean()))
    return max(res_std, 0.01 * mean_abs if mean_abs > 0 else 0.0, 0.001)


def _analyze_step_divergence(
    df_run:      pd.DataFrame,
    df_baseline: pd.DataFrame,
    signals:     list[tuple[str, str, str, str]],
) -> list[dict]:
    """Scan steps in chronological order and compute per-signal divergence.

    Steps are matched by the 'step' column (names are standardised per product).
    A step is flagged divergent when any signal exceeds _Z_THRESH after
    normalising by the detrended baseline noise floor.

    Returns a list of step dicts:
      {step, diverges, sigs: [{col, label, unit, mean_diff, slope_diff,
                                z_mean, z_drift, diverges}, ...]}
    """
    if df_run.empty or "step" not in df_run.columns:
        return []

    first_t      = df_run.groupby("step")["t_min"].min()
    ordered_steps = first_t.sort_values().index.tolist()

    results: list[dict] = []
    for step in ordered_steps:
        sel = df_run[df_run["step"] == step]
        bas = df_baseline[df_baseline["step"] == step]

        sig_results: list[dict] = []
        step_diverges = False

        for col, label, unit, _ in signals:
            if col not in sel.columns:
                continue

            sv = sel[col].dropna()
            bv = bas[col].dropna() if not bas.empty else pd.Series([], dtype=float)
            if len(sv) < 3 or len(bv) < 3:
                continue

            sv_arr = sv.values.astype(float)
            bv_arr = bv.values.astype(float)
            st_t   = sel.loc[sv.index, "t_min"].values.astype(float)
            bt_t   = bas.loc[bv.index, "t_min"].values.astype(float)

            mean_diff  = float(sv_arr.mean()) - float(bv_arr.mean())
            std_diff   = float(sv_arr.std())  - float(bv_arr.std())
            s_slope    = float(np.polyfit(st_t, sv_arr, 1)[0]) if len(st_t) > 1 else 0.0
            b_slope    = float(np.polyfit(bt_t, bv_arr, 1)[0]) if len(bt_t) > 1 else 0.0
            slope_diff = s_slope - b_slope

            denom    = _norm_denom(bt_t, bv_arr)
            step_dur = float(st_t.max() - st_t.min()) if len(st_t) > 1 else 1.0
            z_mean   = abs(mean_diff)  / denom
            z_drift  = abs(slope_diff) * step_dur / denom  # total slope-drift normalised

            sig_div = z_mean > _Z_THRESH or z_drift > _Z_THRESH
            if sig_div:
                step_diverges = True

            sig_results.append({
                "col": col, "label": label, "unit": unit,
                "mean_diff": mean_diff, "std_diff": std_diff, "slope_diff": slope_diff,
                "z_mean": z_mean, "z_drift": z_drift, "diverges": sig_div,
            })

        results.append({"step": step, "diverges": step_diverges, "sigs": sig_results})

    return results


def _step_bullets(
    step:       str,
    product:    str,
    sigs:       list[dict],
    is_first:   bool,
    first_step: str,
) -> list[str]:
    """Return 1–2 process-language bullet strings for a divergent step.

    Separation and cooling signals are framed as observations consistent with
    upstream differences, not as causes.  "caused by", "root cause", and
    "primary driver" are never used.
    """
    diverging = sorted(
        [s for s in sigs if s["diverges"]],
        key=lambda s: s["z_mean"], reverse=True,
    )
    if not diverging:
        return []

    def d(v: float, pos: str = "above", neg: str = "below") -> str:
        return pos if v > 0 else neg

    def fmt(v: float, unit: str) -> str:
        return f"{v:+.3g} {unit}".rstrip()

    bullets: list[str] = []

    if product == "QUARK":
        ph   = next((s for s in diverging if s["col"] == "pH"), None)
        temp = next((s for s in diverging if s["col"] == "temperature_C"), None)
        spd  = next((s for s in diverging if s["col"] == "separator_speed_rpm"), None)
        dp   = next((s for s in diverging if s["col"] == "separation_deltaP"), None)

        if step in _QUARK_FERM_STEPS:
            if ph:
                dr = d(ph["mean_diff"], "higher", "lower")
                slope_note = ""
                if ph["z_drift"] > _Z_THRESH:
                    slope_note = (
                        ", with acidification proceeding faster than reference"
                        if ph["slope_diff"] < 0
                        else ", with acidification proceeding more slowly than reference"
                    )
                bullets.append(
                    f"pH is {dr} than reference ({fmt(ph['mean_diff'], ph['unit'])})"
                    f"{slope_note}, consistent with a difference in fermentation progression."
                )
            if temp and len(bullets) < 2:
                dr = d(temp["mean_diff"], "higher", "lower")
                bullets.append(
                    f"Incubation temperature is {dr} than reference "
                    f"({fmt(temp['mean_diff'], temp['unit'])})."
                )

        elif step in _QUARK_SEP_STEPS:
            upstream_note = (
                ", observed after fermentation differences in earlier steps"
                if (not is_first and first_step in _QUARK_FERM_STEPS)
                else ""
            )
            parts = []
            if spd:
                parts.append(
                    f"centrifuge speed {d(spd['mean_diff'], 'higher', 'lower')} "
                    f"({fmt(spd['mean_diff'], spd['unit'])})"
                )
            if dp:
                parts.append(
                    f"separation ΔP {d(dp['mean_diff'], 'higher', 'lower')} "
                    f"({fmt(dp['mean_diff'], dp['unit'])})"
                )
            if parts:
                bullets.append(
                    f"Separation metrics differ from reference: {'; '.join(parts)}"
                    + upstream_note + "."
                )
            if ph and len(bullets) < 2:
                dr = d(ph["mean_diff"], "higher", "lower")
                bullets.append(
                    f"Post-separation pH is {dr} ({fmt(ph['mean_diff'], ph['unit'])})."
                )

        else:
            sig = diverging[0]
            bullets.append(
                f"{sig['label']} is {d(sig['mean_diff'])} than reference "
                f"({fmt(sig['mean_diff'], sig['unit'])}) in the "
                f"{step.replace('_', ' ')} step."
            )

    elif product == "HIGH_PROTEIN_PUDDING":
        dT       = next((s for s in diverging if s["col"] == "deltaT_heat_exchanger"), None)
        pressure = next((s for s in diverging if s["col"] == "pressure_bar"), None)
        flow     = next((s for s in diverging if s["col"] == "flow_rate_lpm"), None)
        temp     = next((s for s in diverging if s["col"] == "temperature_C"), None)

        if step in _PUDDING_THERMAL_STEPS:
            if dT:
                dr = d(dT["mean_diff"], "higher", "lower")
                bullets.append(
                    f"Heat exchanger ΔT is {dr} than reference "
                    f"({fmt(dT['mean_diff'], dT['unit'])}), consistent with a difference "
                    f"in heat transfer during this step."
                )
            hyd: list[str] = []
            if pressure:
                hyd.append(
                    f"pressure {d(pressure['mean_diff'], 'elevated', 'reduced')} "
                    f"({fmt(pressure['mean_diff'], pressure['unit'])})"
                )
            if flow:
                hyd.append(
                    f"flow {d(flow['mean_diff'], 'elevated', 'reduced')} "
                    f"({fmt(flow['mean_diff'], flow['unit'])})"
                )
            if hyd and len(bullets) < 2:
                bullets.append(
                    f"Hydraulic signals suggest a process difference: {'; '.join(hyd)}."
                )
            if temp and not dT and len(bullets) < 2:
                dr = d(temp["mean_diff"], "higher", "lower")
                bullets.append(
                    f"Process temperature is {dr} than reference "
                    f"({fmt(temp['mean_diff'], temp['unit'])})."
                )

        else:
            upstream_note = (
                ", consistent with effects propagating from earlier thermal steps"
                if (not is_first and first_step in _PUDDING_THERMAL_STEPS)
                else ""
            )
            parts = [
                f"{s['label']} {d(s['mean_diff'])} ({fmt(s['mean_diff'], s['unit'])})"
                for s in diverging[:2]
            ]
            if parts:
                bullets.append(
                    f"Downstream differences: {', '.join(parts)}"
                    + upstream_note + "."
                )

    return bullets[:2]


def _render_divergence_panel(
    df_run:        pd.DataFrame,
    df_baseline:   pd.DataFrame,
    product:       str,
    x_range:       tuple[float, float] | None,  # kept for call-site compat; not used
    selected_step: str,
) -> None:
    """Step-anchored divergence narrative across all process steps.

    Every step is shown in chronological order with a status icon:
      ⭐  first step where divergence is detected
      ⚠️  subsequent divergent step (downstream)
      ✅  no divergence detected in this step
    """
    signals      = _QUARK_SIGNALS if product == "QUARK" else _PUDDING_SIGNALS
    step_results = _analyze_step_divergence(df_run, df_baseline, signals)
    if not step_results:
        return

    st.subheader("Where this run diverges from reference (and what happens next)")

    first_div  = next((sr for sr in step_results if sr["diverges"]), None)
    first_step = first_div["step"] if first_div else None

    with st.container(border=True):
        # Legend
        st.caption("⭐ First divergence  ·  ⚠️ Downstream divergence  ·  ✅ Signals stable")

        if first_step is None:
            st.info(
                "No significant divergence from the NORMAL reference detected across all steps."
            )
        else:
            for sr in step_results:
                is_first_div = (sr["step"] == first_step)
                is_zoomed    = (selected_step == sr["step"])
                sl           = sr["step"].replace("_", " ").capitalize()

                if is_first_div:
                    icon = "⭐"
                elif sr["diverges"]:
                    icon = "⚠️"
                else:
                    icon = "✅"

                zoomed_tag = " *(zoomed)*" if is_zoomed else ""
                st.markdown(f"{icon} **{sl}**{zoomed_tag}")

                if not sr["diverges"]:
                    st.markdown("  - Signals stable vs reference.")
                else:
                    bullets = _step_bullets(
                        sr["step"], product, sr["sigs"], is_first_div, first_step,
                    )
                    if bullets:
                        for b in bullets:
                            st.markdown(f"  - {b}")
                    else:
                        for sig in [s for s in sr["sigs"] if s["diverges"]][:2]:
                            dr = "above" if sig["mean_diff"] > 0 else "below"
                            st.markdown(
                                f"  - {sig['label']} differs from reference "
                                f"({sig['mean_diff']:+.3g} {sig['unit']})."
                            )

                    # Per-signal metric table hidden behind expander
                    measured = [s for s in sr["sigs"] if s["z_mean"] > 0 or s["z_drift"] > 0]
                    if measured:
                        with st.expander("Advanced details"):
                            adv_rows = [
                                {
                                    "Signal":    s["label"],
                                    "Mean diff": f"{s['mean_diff']:+.3g} {s['unit']}",
                                    "z-mean":    f"{s['z_mean']:.2f}",
                                    "z-drift":   f"{s['z_drift']:.2f}",
                                    "Flagged":   "yes" if s["diverges"] else "—",
                                }
                                for s in measured
                            ]
                            st.dataframe(
                                pd.DataFrame(adv_rows),
                                hide_index=True,
                                use_container_width=True,
                            )

        st.divider()
        st.caption(
            "Observations are relative to the NORMAL reference run. "
            "Differences in an earlier step are *consistent with*, but do not "
            "establish, a causal relationship with changes observed later."
        )
        st.caption(
            "✅ No divergence = no divergence detected in monitored signals for that step."
        )


# ── Right panel ───────────────────────────────────────────────────────────────

def _render_right_panel(
    mode:    str,
    run_row: pd.Series,
    lab_row: pd.Series | None,
    story:   dict | None,
) -> None:
    # Outcomes
    st.subheader("Run outcomes")
    with st.container(border=True):
        if lab_row is not None:
            flag = lab_row["result_flag"]
            color = {"PASS": "normal", "WARN": "off", "FAIL": "inverse"}.get(flag, "normal")
            st.metric("Result flag", flag, delta=None, delta_color=color)
        else:
            st.caption("Lab result not found.")

        o1, o2 = st.columns(2)
        o1.metric("Yield loss",    f"{run_row['yield_loss_pct']:.1f} %")
        o2.metric("Downtime",      f"{run_row['downtime_minutes']:.0f} min")
        o1.metric("Fouling grade", str(run_row["fouling_grade"]))
        o2.metric("Extra CIP",     "Yes" if run_row["extra_cleaning"] else "No")

    # Lab results
    st.subheader("Lab results")
    with st.container(border=True):
        if lab_row is not None:
            _render_lab_metrics(lab_row, run_row["product"])
        else:
            st.info("No lab result for this run.")

    # Narrative (guided mode only)
    if mode == "guided" and story is not None:
        st.subheader("Process narrative")
        with st.container(border=True):
            st.write(story["narrative"])
            if story.get("what_to_watch"):
                st.markdown("**What to watch:**")
                for bullet in story["what_to_watch"]:
                    st.markdown(f"- {bullet}")


def _render_lab_metrics(lab_row: pd.Series, product: str) -> None:
    """Show the most relevant lab values as compact metrics."""
    st.metric("Protein",      f"{lab_row['protein_pct']:.2f} %")
    st.metric("Total solids", f"{lab_row['total_solids_pct']:.2f} %")
    st.metric("Viscosity",    f"{lab_row['viscosity_cP']:.0f} cP")
    st.metric("Texture score",f"{lab_row['texture_score']:.1f} / 100")

    if product == "QUARK" and pd.notna(lab_row.get("final_pH")):
        c1, c2 = st.columns(2)
        c1.metric("Final pH",        f"{lab_row['final_pH']:.3f}")
        ferm = lab_row.get("fermentation_time_hr")
        c2.metric("Ferm. time", "DNF" if ferm == 99.0 else f"{ferm:.1f} h")
        wpl = lab_row.get("whey_protein_loss_proxy")
        if pd.notna(wpl):
            st.metric("Whey protein loss", f"{wpl:.4f}")

    if product == "HIGH_PROTEIN_PUDDING" and pd.notna(lab_row.get("fouling_index_end")):
        st.metric("Fouling index (end)", f"{lab_row['fouling_index_end']:.4f}")


# =============================================================================
# HELPERS
# =============================================================================

def _find_story(demo_cases: dict, run_id: str) -> dict | None:
    """Return the demo story for a given run_id, or None."""
    for s in demo_cases.get("stories", []):
        if s["run_id"] == run_id:
            return s
    return None


# =============================================================================
# ENTRY POINT
# =============================================================================

def main() -> None:
    runs       = load_runs()
    ts         = load_timeseries()
    lab        = load_lab()
    events     = load_events()
    demo_cases = load_demo_cases()

    mode, run_id = render_sidebar(runs, demo_cases)
    render_main(mode, run_id, runs, ts, lab, events, demo_cases)


main()
