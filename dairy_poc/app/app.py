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


def _check_demo_cases_sync(demo_cases: dict, runs: pd.DataFrame) -> list[dict]:
    """Return one dict per story that is out of sync with runs.csv (empty = all OK)."""
    run_index = runs.set_index("run_id")[["product", "scenario"]]
    issues: list[dict] = []
    for story in demo_cases.get("stories", []):
        sid = story.get("story_id", "?")
        rid = story.get("run_id", "")
        if rid not in run_index.index:
            issues.append({"story_id": sid, "run_id": rid, "field": "run_id", "issue": "not in runs.csv"})
            continue
        row = run_index.loc[rid]
        if row["product"] != story.get("product"):
            issues.append({
                "story_id": sid, "run_id": rid, "field": "product",
                "issue": f"story={story.get('product')}  runs={row['product']}",
            })
        if row["scenario"] != story.get("scenario"):
            issues.append({
                "story_id": sid, "run_id": rid, "field": "scenario",
                "issue": f"story={story.get('scenario')}  runs={row['scenario']}",
            })
    return issues


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
_ML_WATCH_THRESH:  float = 0.90  # normalised score: Watch starts here
_ML_HIGH_THRESH:   float = 1.00  # normalised score: High = at/above 95th-pct NORMAL
_ML_TREND_WINDOW:  int   = 30    # sliding window width in minutes for the score trend chart
_ML_TREND_STEP:    int   = 5     # step between successive windows in minutes

# QUARK step + signal partitioning — keeps fermentation and separation models completely
# independent, eliminating false positives from the zero-to-active step boundary.
_QUARK_ML_FERM_STEPS:   frozenset[str] = frozenset({
    "inoculation_mixing", "early_fermentation", "late_fermentation",
})
_QUARK_ML_SEP_STEP:     str   = "separation"
_QUARK_ML_SPEED_THRESH: float = 1000.0  # rpm; below = centrifuge ramp-up, excluded from sep model
_QUARK_ML_PH_HI:        float = 5.2     # pH threshold: frac below this measures ferm progress
_QUARK_ML_PH_LO:        float = 4.6     # pH threshold: frac below this measures ferm completion

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


def _extract_quark_ferm_features(run_ts: pd.DataFrame) -> list[float]:
    """6-dim fermentation health vector for one QUARK run (fermentation steps only).

    Only uses pH over inoculation_mixing + early_fermentation + late_fermentation.
    Separation signals never enter this vector, so zero-to-active step boundaries
    cannot affect fermentation scores.

    Features:
      [pH_mean, late_ferm_slope, pH_min, pH_range, frac_below_5.2, frac_below_4.6]

    Returns [0.0]*6 when fewer than 10 valid pH rows are found.
    """
    if "step" not in run_ts.columns:
        return [0.0] * 6
    ferm = run_ts[run_ts["step"].isin(_QUARK_ML_FERM_STEPS)]
    late = run_ts[run_ts["step"] == "late_fermentation"]

    ph = ferm["pH"].dropna() if "pH" in ferm.columns else pd.Series(dtype=float)
    if len(ph) < 10:
        return [0.0] * 6

    t_all = ferm.loc[ph.index, "t_min"].values.astype(float)
    v_all = ph.values.astype(float)

    ph_late = late["pH"].dropna() if "pH" in late.columns else pd.Series(dtype=float)
    t_late  = late.loc[ph_late.index, "t_min"].values.astype(float) if not ph_late.empty else np.array([])
    v_late  = ph_late.values.astype(float)
    late_slope = (
        float(np.polyfit(t_late, v_late, 1)[0]) if len(v_late) > 3
        else float(np.polyfit(t_all,  v_all,  1)[0])
    )

    return [
        float(v_all.mean()),
        late_slope,
        float(v_all.min()),
        float(v_all.max() - v_all.min()),
        float((v_all < _QUARK_ML_PH_HI).mean()),
        float((v_all < _QUARK_ML_PH_LO).mean()),
    ]


def _extract_quark_sep_features(run_ts: pd.DataFrame) -> list[float]:
    """6-dim separation stability vector (steady-state separation rows only).

    Filters to rows where separator_speed_rpm > _QUARK_ML_SPEED_THRESH to exclude the
    centrifuge ramp-up phase, which activates from zero at the step boundary and would
    otherwise create false anomaly signals.

    Features:
      [deltaP_mean, deltaP_std, deltaP_max, deltaP_slope, speed_mean, speed_std]

    Returns [0.0]*6 when fewer than 5 valid rows are found.
    """
    if "step" not in run_ts.columns:
        return [0.0] * 6
    sep = run_ts[
        (run_ts["step"] == _QUARK_ML_SEP_STEP) &
        (run_ts["separator_speed_rpm"].fillna(0.0) > _QUARK_ML_SPEED_THRESH)
    ]
    dp = sep["separation_deltaP"].dropna() if "separation_deltaP" in sep.columns else pd.Series(dtype=float)
    if len(dp) < 5:
        return [0.0] * 6

    t    = sep.loc[dp.index, "t_min"].values.astype(float)
    v_dp = dp.values.astype(float)
    v_sp = sep.loc[dp.index, "separator_speed_rpm"].fillna(0.0).values.astype(float)
    dp_slope = float(np.polyfit(t, v_dp, 1)[0]) if len(t) > 3 else 0.0

    return [
        float(v_dp.mean()),
        float(v_dp.std()),
        float(v_dp.max()),
        dp_slope,
        float(v_sp.mean()),
        float(v_sp.std()),
    ]


@st.cache_resource
def _load_ml_models() -> dict:
    """Train product-aware IsolationForest models on NORMAL runs.

    QUARK — four step-scoped models:
      ferm_run_pipeline / ferm_run_threshold  run-level Fermentation Health (6 features)
      sep_run_pipeline  / sep_run_threshold   run-level Separation Stability (6 features)
      ferm_row_pipeline / ferm_row_threshold  row-level ferm (pH+temp, ferm steps only)
      sep_row_pipeline  / sep_row_threshold   row-level sep  (ΔP+speed, steady sep only)

    HIGH_PROTEIN_PUDDING — two models (unchanged):
      pipeline     / threshold      row-level, all signals
      run_pipeline / run_threshold  run-level, all signals

    Uses @st.cache_resource so training runs once per app-server start.
    """
    if not _SKLEARN_AVAILABLE:
        return {}

    from sklearn.ensemble import IsolationForest
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    def _make_pipe() -> Pipeline:
        return Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
            ("scaler",  StandardScaler()),
            ("iso",     IsolationForest(n_estimators=100, contamination="auto", random_state=42)),
        ])

    def _fit_thr(pipe: Pipeline, X: np.ndarray) -> float:
        pipe.fit(X)
        return float(np.percentile((-pipe.score_samples(X)).clip(min=0), _ML_THRESHOLD_PCT))

    runs_df = pd.read_csv(_RAW / "runs.csv")
    ts_df   = pd.read_csv(_RAW / "timeseries.csv")
    models: dict = {}

    # ── QUARK: step-scoped, product-aware models ──────────────────────────────
    q_ids = runs_df.loc[
        (runs_df["product"] == "QUARK") & (runs_df["scenario"] == "NORMAL"), "run_id"
    ]
    q_ts = ts_df[ts_df["run_id"].isin(q_ids)]

    if not q_ts.empty:
        models["QUARK"] = {}

        # Run-level Fermentation Health
        fv = [_extract_quark_ferm_features(ts_df[ts_df["run_id"] == r]) for r in q_ids]
        if len(fv) >= 5:
            Xf = np.array(fv, dtype=float)
            p = _make_pipe()
            models["QUARK"]["ferm_run_pipeline"]  = p
            models["QUARK"]["ferm_run_threshold"] = _fit_thr(p, Xf)

        # Run-level Separation Stability
        sv = [_extract_quark_sep_features(ts_df[ts_df["run_id"] == r]) for r in q_ids]
        if len(sv) >= 5:
            Xs = np.array(sv, dtype=float)
            p = _make_pipe()
            models["QUARK"]["sep_run_pipeline"]  = p
            models["QUARK"]["sep_run_threshold"] = _fit_thr(p, Xs)

        # Row-level Fermentation (pH + temperature, fermentation steps only)
        ferm_rows = q_ts[q_ts["step"].isin(_QUARK_ML_FERM_STEPS)]
        ferm_sigs = [s for s in _QUARK_SIGNALS if s[0] in ("pH", "temperature_C")]
        if not ferm_rows.empty:
            Xfr = _extract_ml_features(ferm_rows, ferm_sigs).values.astype(float)
            p = _make_pipe()
            models["QUARK"]["ferm_row_pipeline"]  = p
            models["QUARK"]["ferm_row_threshold"] = _fit_thr(p, Xfr)

        # Row-level Separation (ΔP + speed, steady-state rows only)
        sep_rows = q_ts[
            (q_ts["step"] == _QUARK_ML_SEP_STEP) &
            (q_ts["separator_speed_rpm"].fillna(0.0) > _QUARK_ML_SPEED_THRESH)
        ]
        sep_sigs = [s for s in _QUARK_SIGNALS if s[0] in ("separation_deltaP", "separator_speed_rpm")]
        if not sep_rows.empty:
            Xsr = _extract_ml_features(sep_rows, sep_sigs).values.astype(float)
            p = _make_pipe()
            models["QUARK"]["sep_row_pipeline"]  = p
            models["QUARK"]["sep_row_threshold"] = _fit_thr(p, Xsr)

    # ── HIGH_PROTEIN_PUDDING: unchanged ───────────────────────────────────────
    p_ids = runs_df.loc[
        (runs_df["product"] == "HIGH_PROTEIN_PUDDING") & (runs_df["scenario"] == "NORMAL"), "run_id"
    ]
    p_ts = ts_df[ts_df["run_id"].isin(p_ids)]

    if not p_ts.empty:
        signals = _PUDDING_SIGNALS
        Xp = _extract_ml_features(p_ts, signals).values.astype(float)
        p = _make_pipe()
        models["HIGH_PROTEIN_PUDDING"] = {
            "pipeline":  p,
            "threshold": _fit_thr(p, Xp),
        }
        summaries = [_extract_run_summary(ts_df[ts_df["run_id"] == r], signals) for r in p_ids]
        if len(summaries) >= 5:
            Xr = np.array(summaries, dtype=float)
            rp = _make_pipe()
            models["HIGH_PROTEIN_PUDDING"]["run_pipeline"]  = rp
            models["HIGH_PROTEIN_PUDDING"]["run_threshold"] = _fit_thr(rp, Xr)

    return models


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


# ── Scoring helpers ───────────────────────────────────────────────────────────

def _score_status(score: float) -> tuple[str, str]:
    """(status_label, badge_text) for a normalised anomaly score."""
    if score < _ML_WATCH_THRESH:
        return "Normal", "✅ Normal"
    if score < _ML_HIGH_THRESH:
        return "Watch", "⚠️ Watch"
    return "High", "🚨 High Risk"


_STATUS_FN = {"Normal": st.success, "Watch": st.warning, "High": st.error}


def _anomaly_to_stability(score: float) -> int:
    """Convert a normalised ML anomaly score to a 0–100 Run Stability Score.

    Mapping (anchored to model thresholds):
      score ≤ Watch (0.90) → stability = 100  (pattern within normal band)
      score = 0.95 (mid Watch–High) → stability ≈ 50
      score ≥ High (1.00) → stability = 0     (strong anomaly pattern)
    Linear between Watch and High; clamped outside that range.
    """
    band = _ML_HIGH_THRESH - _ML_WATCH_THRESH  # = 0.10
    normalised_risk = max(0.0, min(1.0, (score - _ML_WATCH_THRESH) / band))
    return round(100 - 100 * normalised_risk)


def _compute_run_stability(
    df_run:  pd.DataFrame,
    product: str,
) -> tuple[int, str, str]:
    """(stability_0_100, status, badge) for a run, using ML Early Warning scores.

    QUARK: average of Fermentation Health and Separation Stability scores.
           Averaging avoids one sub-system's false positives dominating the summary.
    PUDDING: single run-level anomaly score.
    Returns (100, 'Normal', '✅ Normal') when model is unavailable.
    """
    if not _SKLEARN_AVAILABLE or df_run.empty:
        return 100, "Normal", "✅ Normal"

    if product == "QUARK":
        scores = []
        for result in (_compute_quark_ferm_score(df_run), _compute_quark_sep_score(df_run)):
            if result is not None:
                scores.append(result[0])
        ml_score = sum(scores) / len(scores) if scores else 0.0
    else:
        result = _compute_ml_run_score(df_run, product)
        ml_score = result[0] if result is not None else 0.0

    status, badge = _score_status(ml_score)
    return _anomaly_to_stability(ml_score), status, badge


def _compute_quark_ferm_score(df_run: pd.DataFrame) -> tuple[float, float] | None:
    """Normalised QUARK Fermentation Health score.  Returns (score, threshold) or None."""
    if not _SKLEARN_AVAILABLE:
        return None
    m = _load_ml_models().get("QUARK")
    if m is None or "ferm_run_pipeline" not in m:
        return None
    feats = np.array([_extract_quark_ferm_features(df_run)], dtype=float)
    raw   = float((-m["ferm_run_pipeline"].score_samples(feats)).clip(min=0)[0])
    return raw / max(m["ferm_run_threshold"], 1e-9), m["ferm_run_threshold"]


def _compute_quark_sep_score(df_run: pd.DataFrame) -> tuple[float, float] | None:
    """Normalised QUARK Separation Stability score.  Returns (score, threshold) or None."""
    if not _SKLEARN_AVAILABLE:
        return None
    m = _load_ml_models().get("QUARK")
    if m is None or "sep_run_pipeline" not in m:
        return None
    feats = np.array([_extract_quark_sep_features(df_run)], dtype=float)
    raw   = float((-m["sep_run_pipeline"].score_samples(feats)).clip(min=0)[0])
    return raw / max(m["sep_run_threshold"], 1e-9), m["sep_run_threshold"]


def _compute_ml_run_score(df_run: pd.DataFrame, product: str) -> tuple[float, float] | None:
    """Normalised run-level score for PUDDING (single run-level model, all signals)."""
    if not _SKLEARN_AVAILABLE:
        return None
    m = _load_ml_models().get(product)
    if m is None or "run_pipeline" not in m:
        return None
    feats = np.array([_extract_run_summary(df_run, _PUDDING_SIGNALS)], dtype=float)
    raw   = float((-m["run_pipeline"].score_samples(feats)).clip(min=0)[0])
    return raw / max(m["run_threshold"], 1e-9), m["run_threshold"]


def _compute_ml_row_scores(df_ts: pd.DataFrame, product: str) -> np.ndarray | None:
    """Per-row normalised scores for PUDDING (row-level, all signals)."""
    if not _SKLEARN_AVAILABLE:
        return None
    m = _load_ml_models().get(product)
    if m is None or "pipeline" not in m:
        return None
    X   = _extract_ml_features(df_ts, _PUDDING_SIGNALS).values.astype(float)
    raw = (-m["pipeline"].score_samples(X)).clip(min=0)
    return raw / max(float(m["threshold"]), 1e-9)


def _compute_quark_ferm_row_scores(
    df_run: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Per-row Fermentation Health scores aligned to df_run index.

    Returns (t_arr, scores) where scores is NaN for rows outside fermentation steps.
    Only pH + temperature signals; only _QUARK_ML_FERM_STEPS rows are scored.
    """
    if not _SKLEARN_AVAILABLE:
        return None
    m = _load_ml_models().get("QUARK")
    if m is None or "ferm_row_pipeline" not in m:
        return None
    if "step" not in df_run.columns:
        return None
    ferm_mask = df_run["step"].isin(_QUARK_ML_FERM_STEPS)
    ferm_rows = df_run[ferm_mask]
    if ferm_rows.empty:
        return None
    sigs = [s for s in _QUARK_SIGNALS if s[0] in ("pH", "temperature_C")]
    X    = _extract_ml_features(ferm_rows, sigs).values.astype(float)
    raw  = (-m["ferm_row_pipeline"].score_samples(X)).clip(min=0)
    norm = raw / max(float(m["ferm_row_threshold"]), 1e-9)
    out  = np.full(len(df_run), np.nan)
    out[ferm_mask.values] = norm
    return df_run["t_min"].values.astype(float), out


def _compute_quark_sep_row_scores(
    df_run: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Per-row Separation Stability scores aligned to df_run index.

    Returns (t_arr, scores) where scores is NaN for ramp-up or non-separation rows.
    Only ΔP + speed signals; only steady-state separation rows are scored.
    """
    if not _SKLEARN_AVAILABLE:
        return None
    m = _load_ml_models().get("QUARK")
    if m is None or "sep_row_pipeline" not in m:
        return None
    if "step" not in df_run.columns:
        return None
    steady_mask = (
        (df_run["step"] == _QUARK_ML_SEP_STEP) &
        (df_run["separator_speed_rpm"].fillna(0.0) > _QUARK_ML_SPEED_THRESH)
    )
    steady_rows = df_run[steady_mask]
    if steady_rows.empty:
        return None
    sigs = [s for s in _QUARK_SIGNALS if s[0] in ("separation_deltaP", "separator_speed_rpm")]
    X    = _extract_ml_features(steady_rows, sigs).values.astype(float)
    raw  = (-m["sep_row_pipeline"].score_samples(X)).clip(min=0)
    norm = raw / max(float(m["sep_row_threshold"]), 1e-9)
    out  = np.full(len(df_run), np.nan)
    out[steady_mask.values] = norm
    return df_run["t_min"].values.astype(float), out


# ── ML score panel ────────────────────────────────────────────────────────────

def _ml_why_bullets(
    df_run:        "pd.DataFrame",
    df_baseline:   "pd.DataFrame",
    product:       str,
    selected_step: str,
) -> list:
    """Top-2 descriptive bullet strings comparing run vs baseline in the active window.

    Returns an empty list when data is insufficient or differences are negligible.
    Avoids causal language; describes pattern differences only.
    """
    if df_run.empty or df_baseline.empty or "step" not in df_run.columns:
        return []

    findings = []  # (z_score, sentence)

    def _analyse(run_rows, bas_rows, sig_col, sig_label, step_lbl):
        if sig_col not in run_rows.columns:
            return
        sv = run_rows[sig_col].dropna()
        bv = bas_rows[sig_col].dropna() if not bas_rows.empty else pd.Series([], dtype=float)
        if len(sv) < 3 or len(bv) < 3:
            return

        sv_arr = sv.values.astype(float)
        bv_arr = bv.values.astype(float)
        st_t = run_rows.loc[sv.index, "t_min"].values.astype(float)
        bt_t = bas_rows.loc[bv.index, "t_min"].values.astype(float)

        mean_diff  = float(sv_arr.mean()) - float(bv_arr.mean())
        std_run    = float(sv_arr.std())
        std_bas    = float(bv_arr.std()) if len(bv_arr) > 1 else 0.0
        std_diff   = std_run - std_bas
        s_slope    = float(np.polyfit(st_t, sv_arr, 1)[0]) if len(st_t) > 1 else 0.0
        b_slope    = float(np.polyfit(bt_t, bv_arr, 1)[0]) if len(bt_t) > 1 else 0.0
        slope_diff = s_slope - b_slope

        denom     = _norm_denom(bt_t, bv_arr)
        step_dur  = float(st_t.max() - st_t.min()) if len(st_t) > 1 else 1.0
        z_mean    = abs(mean_diff) / denom
        z_drift   = abs(slope_diff) * step_dur / denom
        z_var     = abs(std_diff) / max(std_bas, denom * 0.1, 0.001) if std_bas > 0 else 0.0
        dominant  = max(z_mean, z_drift, z_var)
        if dominant < 0.4:
            return

        lbl = step_lbl.replace("_", " ")
        if z_mean >= z_drift and z_mean >= z_var:
            dr = "higher" if mean_diff > 0 else "lower"
            if sig_col == "pH":
                sentence = f"pH level {dr} than reference during {lbl}"
            elif sig_col == "temperature_C":
                sentence = f"temperature {dr} than reference during {lbl}"
            elif sig_col == "separation_deltaP":
                sentence = f"separation ΔP {dr} than reference during {lbl}"
            elif sig_col == "separator_speed_rpm":
                sentence = f"centrifuge speed {dr} than reference during {lbl}"
            elif sig_col == "pressure_bar":
                sentence = f"pressure {dr} than reference during {lbl}"
            elif sig_col == "deltaT_heat_exchanger":
                sentence = f"heat exchanger ΔT {dr} than reference during {lbl}"
            else:
                sentence = f"{sig_label} level {dr} than reference during {lbl}"
        elif z_drift >= z_var:
            if sig_col == "pH":
                dr = "faster" if slope_diff < 0 else "slower"
                sentence = f"pH acidification {dr} than reference during {lbl}"
            elif sig_col == "temperature_C":
                dr = "steeper" if slope_diff > 0 else "flatter"
                sentence = f"temperature slope {dr} than reference during {lbl}"
            elif sig_col == "separation_deltaP":
                dr = "increasing" if slope_diff > 0 else "decreasing"
                sentence = f"separation ΔP trending {dr} vs reference during {lbl}"
            elif sig_col == "flow_rate_lpm":
                dr = "declining" if slope_diff < 0 else "rising"
                sentence = f"flow rate {dr} relative to reference during {lbl}"
            else:
                dr = "steeper" if slope_diff > 0 else "flatter"
                sentence = f"{sig_label} slope {dr} than reference during {lbl}"
        else:
            dr = "higher" if std_diff > 0 else "lower"
            if sig_col == "separation_deltaP":
                sentence = f"separation ΔP variability {dr} than reference during {lbl}"
            elif sig_col == "pressure_bar":
                sentence = f"pressure variability {dr} than reference during {lbl}"
            else:
                sentence = f"{sig_label} variability {dr} than reference during {lbl}"

        findings.append((dominant, sentence))

    def _run_scope(step_filter, sig_cols):
        if step_filter is None:
            rr, br = df_run, df_baseline
            slbl = "the run"
        elif isinstance(step_filter, (set, frozenset)):
            rr = df_run[df_run["step"].isin(step_filter)]
            br = df_baseline[df_baseline["step"].isin(step_filter)]
            slbl = "fermentation" if step_filter & _QUARK_ML_FERM_STEPS else "the selected window"
        else:
            rr = df_run[df_run["step"] == step_filter]
            br = df_baseline[df_baseline["step"] == step_filter]
            slbl = step_filter
        all_sigs = _QUARK_SIGNALS if product == "QUARK" else _PUDDING_SIGNALS
        for col, label, _, _ in all_sigs:
            if col in sig_cols:
                _analyse(rr, br, col, label, slbl)

    if product == "QUARK":
        ferm_sigs = {"pH", "temperature_C"}
        sep_sigs  = {"separation_deltaP", "separator_speed_rpm"}
        if selected_step in _QUARK_ML_FERM_STEPS:
            _run_scope(selected_step, ferm_sigs)
        elif selected_step == _QUARK_ML_SEP_STEP:
            _run_scope(_QUARK_ML_SEP_STEP, sep_sigs)
        else:  # full_run or unknown -> both contexts
            _run_scope(_QUARK_ML_FERM_STEPS, ferm_sigs)
            _run_scope(_QUARK_ML_SEP_STEP,   sep_sigs)
    else:
        step = selected_step if selected_step not in ("full_run", None) else None
        pudding_sigs = {s[0] for s in _PUDDING_SIGNALS}
        _run_scope(step, pudding_sigs)

    findings.sort(key=lambda x: x[0], reverse=True)
    return [s for _, s in findings[:2]]


def _render_ml_score_panel(
    df_run:        "pd.DataFrame",
    product:       str,
    x_range:       "tuple[float, float] | None",
    selected_step: str,
    run_id:        str,
) -> None:
    """Always-visible ML Early Warning panel, product-aware.

    QUARK: two side-by-side metric cards + ferm/sep explainers.
    PUDDING: single metric card + process-specific explainer.
    """
    st.subheader("ML Early Warning")

    if not _SKLEARN_AVAILABLE:
        st.caption("scikit-learn not installed — ML early warning unavailable.")
        return
    if df_run.empty:
        return

    # ── Metric cards ──────────────────────────────────────────────────────────
    if product == "QUARK":
        _render_quark_ml_cards(df_run)
        score_elevated = _quark_score_elevated(df_run)
    else:
        _render_pudding_ml_card(df_run, product, selected_step)
        score_elevated = _pudding_score_elevated(df_run, product)

    # ── Explanation expanders ─────────────────────────────────────────────────
    _render_ml_explanation_expanders(product)

    # ── Why it triggered ─────────────────────────────────────────────────────
    # Always attempts to load a NORMAL baseline for this run (uses cached loaders).
    try:
        all_runs = load_runs()
        all_ts   = load_timeseries()
        run_meta = all_runs[all_runs["run_id"] == run_id]
        scale    = str(run_meta["scale"].iloc[0]) if not run_meta.empty else "PRODUCTION"
        b_rid    = _find_baseline_run_id(all_runs, product, scale)
        df_baseline = all_ts[all_ts["run_id"] == b_rid].copy() if b_rid else None
    except Exception:
        df_baseline = None

    if df_baseline is not None and score_elevated:
        bullets = _ml_why_bullets(df_run, df_baseline, product, selected_step)
        if bullets:
            step_lbl = selected_step.replace("_", " ") if selected_step != "full_run" else "full run"
            with st.container(border=True):
                st.caption(
                    f"⚠️ **Pattern differences vs normal reference"
                    f" ({step_lbl})**"
                )
                for b in bullets:
                    st.caption(f"• {b}")
                st.caption(
                    "_These observations describe signal patterns that differ from "
                    "typical NORMAL runs. They are indicative of where to investigate "
                    "— not a diagnosis or root-cause statement._"
                )


def _quark_score_elevated(df_run: "pd.DataFrame") -> bool:
    """True when either QUARK sub-score is Watch or High."""
    ferm = _compute_quark_ferm_score(df_run)
    sep  = _compute_quark_sep_score(df_run)
    for result in (ferm, sep):
        if result is not None:
            score, _ = result
            if score >= _ML_WATCH_THRESH:
                return True
    return False


def _pudding_score_elevated(df_run: "pd.DataFrame", product: str) -> bool:
    """True when PUDDING run score is Watch or High."""
    result = _compute_ml_run_score(df_run, product)
    return result is not None and result[0] >= _ML_WATCH_THRESH


def _render_ml_explanation_expanders(product: str) -> None:
    """Two collapsible expanders: interpretation guide + score calculation."""
    is_quark = product == "QUARK"

    with st.expander("What does this early warning mean?"):
        st.caption(
            "✅ **Normal** — signal patterns are consistent with stable "
            "historical runs for this product and step. No action needed."
        )
        st.caption(
            "⚠️ **Watch** — a deviation pattern is emerging. "
            "Review the highlighted step or window to see which signals are shifting."
        )
        st.caption(
            "🚨 **High** — a strong deviation pattern is present. "
            "Treat as a priority for process review; compare against reference runs."
        )
        st.caption(
            "_These are indicative signals from a pattern-matching model — "
            "not definitive process alarms. Always apply process and domain "
            "knowledge before acting._"
        )

    calc_text_intro = (
        "The score measures how different this run’s signal patterns are "
        "from patterns seen in NORMAL runs (trained and calibrated on NORMAL data only)."
    )
    if is_quark:
        score_detail = (
            "**Fermentation Early Warning (ML)** captures pH trajectory and temperature patterns "
            "during inoculation and fermentation. "
            "**Separation Early Warning (ML)** captures centrifuge speed and ΔP patterns "
            "during the separation phase. "
            "These are independent scores — separation early warning reflects "
            "downstream centrifugation behaviour, not fermentation outcomes."
        )
    else:
        score_detail = (
            "The score captures temperature, pressure, heat-exchanger ΔT, and "
            "flow rate patterns across heating, holding, and downstream steps. "
            "A high score during holding often reflects thermal uniformity or "
            "fouling-related patterns."
        )

    with st.expander("How is the score calculated (simplified)?"):
        st.caption(calc_text_intro)
        st.caption(
            "• Patterns detected: level shifts, drift (slope change), and "
            "variability relative to typical NORMAL runs."
        )
        st.caption(
            "• The model is trained and threshold-calibrated on NORMAL runs only — "
            "anomalous patterns produce higher scores."
        )
        st.caption(
            "• The trend chart uses a 30-min sliding window updated every 5 min, "
            "showing how the score evolves through the run."
        )
        st.caption(score_detail)


def _render_quark_ml_cards(df_run: "pd.DataFrame") -> None:
    ferm = _compute_quark_ferm_score(df_run)
    sep  = _compute_quark_sep_score(df_run)

    c_ferm, c_sep = st.columns(2)
    for col, label, result in [
        (c_ferm, "Fermentation Early Warning (ML)",  ferm),
        (c_sep,  "Separation Early Warning (ML)",   sep),
    ]:
        with col:
            with st.container(border=True):
                st.markdown(f"**{label}**")
                if result is None:
                    st.caption("Model unavailable.")
                else:
                    score, _ = result
                    status, badge = _score_status(score)
                    st.metric("Score",  f"{score:.2f}")
                    st.metric("Status", status)
                    _STATUS_FN[status](badge)
                    st.caption(
                        f"Watch ≥ {_ML_WATCH_THRESH:.2f}  ·  High ≥ {_ML_HIGH_THRESH:.2f}"
                    )

    st.caption(
        "Fermentation Early Warning (ML) indicates upstream deviation during acidification.  "
        "Separation Early Warning (ML) indicates downstream behaviour during centrifugation.  "
        "Neither is a root-cause diagnosis — observational signals only."
    )


def _render_pudding_ml_card(
    df_run:        "pd.DataFrame",
    product:       str,
    selected_step: str,
) -> None:
    result = _compute_ml_run_score(df_run, product)
    if result is None:
        st.caption("ML model not available for this product.")
        return
    run_score, run_threshold = result
    status, badge = _score_status(run_score)

    with st.container(border=True):
        c1, c2 = st.columns(2)
        c1.metric("Anomaly score", f"{run_score:.2f}")
        c2.metric("Status",        status)
        _STATUS_FN[status](badge)
        st.caption(
            f"Threshold (Watch): {_ML_WATCH_THRESH:.2f}  ·  "
            f"Threshold (High Risk): {_ML_HIGH_THRESH:.2f}  ·  "
            f"Training alert level (95th-pct NORMAL): {run_threshold:.4f}  ·  "
            f"Level 1 indicative signal — not a definitive alarm"
        )
        if selected_step != "full_run" and "step" in df_run.columns:
            row_scores = _compute_ml_row_scores(df_run, product)
            if row_scores is not None:
                mask = df_run["step"].values == selected_step
                if mask.any():
                    st.caption(
                        f"Step score ({selected_step.replace('_', ' ').capitalize()}): "
                        f"{float(row_scores[mask].mean()):.2f}  (mean row-level)"
                    )

def _draw_trend_figure(
    t_vis:       "np.ndarray",
    s_vis:       "np.ndarray",
    df_run:      "pd.DataFrame",
    run_evts:    "pd.DataFrame",
    x_range:     "tuple[float, float] | None",
    product:     str,
    title:       str,
    trace_color: str,
    fill_color:  str,
):
    """Shared chart builder for all score-trend variants."""
    first_watch = next((float(t) for t, s in zip(t_vis, s_vis) if s >= _ML_WATCH_THRESH), None)
    first_high  = next((float(t) for t, s in zip(t_vis, s_vis) if s >= _ML_HIGH_THRESH),  None)

    y_max = max(float(s_vis.max()), _ML_HIGH_THRESH) * 1.1 + 0.15

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=t_vis, y=s_vis,
        mode="lines",
        line=dict(color=trace_color, width=2.0),
        fill="tozeroy",
        fillcolor=fill_color,
        hovertemplate="t = %{x:.1f} min<br>Window score = %{y:.3f}<extra></extra>",
        showlegend=False,
    ))

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

    first_warn = first_high if first_high is not None else first_watch
    if first_warn is not None:
        fig.add_vline(x=first_warn, line_dash="dot", line_color="#E53935", line_width=1.5)
        near = df_run[(df_run["t_min"] - first_warn).abs() < (_ML_TREND_STEP + 1.0)]
        step_lbl = near["step"].iloc[0].replace("_", " ") if not near.empty else ""
        ann = f"first warning{(' · ' + step_lbl) if step_lbl else ''}"
        fig.add_annotation(
            x=first_warn, xref="x",
            y=0.88, yref="paper",
            text=ann,
            showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1,
            arrowcolor="#E53935",
            ax=12, ay=-28, axref="pixel", ayref="pixel",
            font=dict(size=9, color="#E53935"),
            bgcolor="rgba(255,255,255,0.82)",
        )

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

    return fig, first_warn, first_high


def _render_quark_trend(
    df_run:        "pd.DataFrame",
    run_evts:      "pd.DataFrame",
    x_range:       "tuple[float, float] | None",
    selected_step: str,
    run_id:        str,
) -> None:
    """Step-aware QUARK trend: fermentation vs separation dispatch."""
    if selected_step == _QUARK_ML_SEP_STEP:
        is_ferm = False
    elif selected_step in _QUARK_ML_FERM_STEPS:
        is_ferm = True
    else:
        is_ferm = True  # "all" or unknown -> fermentation health

    if is_ferm:
        cache_key   = f"ferm_trend_{run_id}"
        chart_title = "Fermentation Early Warning (ML) — Score Trend"
        trace_color = "#1565C0"
        fill_color  = "rgba(21,101,192,0.08)"
        note        = (
            "Fermentation Early Warning (ML) score uses pH and temperature from fermentation steps only. "
            "Elevated score indicates pH trajectory or temperature outside normal bounds."
        )
        if cache_key not in st.session_state:
            result = _compute_quark_ferm_row_scores(df_run)
            if result is None:
                return
            t_raw, s_raw = result
            valid = ~np.isnan(s_raw)
            if valid.sum() == 0:
                return
            t_all, s_all = _compute_score_trend(t_raw[valid], s_raw[valid])
            st.session_state[cache_key] = (t_all, s_all)
    else:
        cache_key   = f"sep_trend_{run_id}"
        chart_title = "Separation Early Warning (ML) — Score Trend"
        trace_color = "#6A1B9A"
        fill_color  = "rgba(106,27,154,0.08)"
        note        = (
            "Separation Early Warning (ML) score uses centrifuge speed and ΔP from steady-state "
            "separation rows only (speed > 1 000 rpm). "
            "Elevated score indicates pressure or speed instability during separation."
        )
        if cache_key not in st.session_state:
            result = _compute_quark_sep_row_scores(df_run)
            if result is None:
                return
            t_raw, s_raw = result
            valid = ~np.isnan(s_raw)
            if valid.sum() == 0:
                return
            t_all, s_all = _compute_score_trend(t_raw[valid], s_raw[valid])
            st.session_state[cache_key] = (t_all, s_all)

    t_all, s_all = st.session_state[cache_key]
    if len(t_all) == 0:
        return

    if x_range is not None:
        lo, hi   = x_range
        vis_mask = (t_all >= lo) & (t_all <= hi)
        t_vis, s_vis = t_all[vis_mask], s_all[vis_mask]
    else:
        t_vis, s_vis = t_all, s_all

    if len(t_vis) == 0:
        st.caption(f"No {chart_title.split('—')[0].strip()} data in the selected zoom window.")
        return

    fig, first_warn, first_high = _draw_trend_figure(
        t_vis, s_vis, df_run, run_evts, x_range, "QUARK",
        chart_title, trace_color, fill_color,
    )

    st.markdown(f"**{chart_title}**")
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    if first_warn is not None:
        cross_type = "High Risk" if first_high is not None else "Watch"
        st.caption(
            f"Indicative early warning: score crosses {cross_type} threshold around "
            f"t = {first_warn:.0f} min.  Observational signal only — not a definitive alarm."
        )
    else:
        st.caption(
            "Score remains below Watch threshold in this window.  "
            "Indicative signal only — not a definitive alarm."
        )
    st.caption(note)


def _render_pudding_trend(
    df_run:   "pd.DataFrame",
    product:  str,
    run_evts: "pd.DataFrame",
    x_range:  "tuple[float, float] | None",
    run_id:   str,
) -> None:
    """PUDDING trend chart — unchanged single-model behaviour."""
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

    if x_range is not None:
        lo, hi   = x_range
        vis_mask = (t_all >= lo) & (t_all <= hi)
        t_vis, s_vis = t_all[vis_mask], s_all[vis_mask]
    else:
        t_vis, s_vis = t_all, s_all

    if len(t_vis) == 0:
        return

    fig, first_warn, first_high = _draw_trend_figure(
        t_vis, s_vis, df_run, run_evts, x_range, product,
        "Early Warning Score trend", "#FF6F00", "rgba(255,111,0,0.08)",
    )

    st.markdown("**Early Warning Score trend**")
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    if first_warn is not None:
        cross_type = "High Risk" if first_high is not None else "Watch"
        st.caption(
            f"Indicative early warning: score crosses {cross_type} threshold around "
            f"t = {first_warn:.0f} min.  Observational signal only — not a definitive alarm."
        )
    else:
        st.caption(
            "Score remains below Watch threshold in this window.  "
            "Indicative signal only — not a definitive alarm."
        )


def _render_score_trend_chart(
    df_run:        "pd.DataFrame",
    product:       str,
    run_evts:      "pd.DataFrame",
    x_range:       "tuple[float, float] | None",
    selected_step: str,
    run_id:        str,
) -> None:
    """Dispatch to product-specific trend chart renderer."""
    if not _SKLEARN_AVAILABLE or df_run.empty:
        return

    if product == "QUARK":
        _render_quark_trend(df_run, run_evts, x_range, selected_step, run_id)
    else:
        _render_pudding_trend(df_run, product, run_evts, x_range, run_id)


# =============================================================================
# SOFT SENSOR — Predicted Final Lab Outcomes
# =============================================================================

_SOFT_SENSOR_TARGETS: list[str] = [
    "protein_pct", "total_solids_pct", "viscosity_value", "final_pH_offline", "d50_um",
]

_SS_TARGET_DISPLAY: list[tuple[str, str, str]] = [
    ("protein_pct",      "Protein",      "{:.2f} %"),
    ("total_solids_pct", "Total solids", "{:.2f} %"),
    ("viscosity_value",  "Viscosity",    "{:.0f} cP"),
    ("final_pH_offline", "pH (offline)", "{:.3f}"),
    ("d50_um",           "D50",          "{:.1f} µm"),
]

_QUARK_CHECKPOINTS:   list[str] = ["early_fermentation", "late_fermentation", "separation"]
_PUDDING_CHECKPOINTS: list[str] = ["mixing", "heating", "holding"]

# Signals used to build features at each checkpoint
_SS_CHECKPOINT_SIGNALS: dict[str, dict[str, list[str]]] = {
    "QUARK": {
        "early_fermentation": ["pH", "temperature_C", "pressure_bar", "flow_rate_lpm"],
        "late_fermentation":  ["pH", "temperature_C", "pressure_bar", "flow_rate_lpm"],
        "separation":         ["pH", "temperature_C", "pressure_bar",
                               "separator_speed_rpm", "separation_deltaP"],
    },
    "HIGH_PROTEIN_PUDDING": {
        "mixing":  ["temperature_C", "pressure_bar", "flow_rate_lpm"],
        "heating": ["temperature_C", "pressure_bar", "flow_rate_lpm", "deltaT_heat_exchanger"],
        "holding": ["temperature_C", "pressure_bar", "flow_rate_lpm", "deltaT_heat_exchanger"],
    },
}

# Step order for resolving "nearest checkpoint at or before selected step"
_SS_QUARK_STEP_ORDER: list[str] = [
    "inoculation_mixing", "gel_break_mixing", "early_fermentation",
    "late_fermentation", "separation", "standardization_mix",
    "filling_packaging", "cooling",
]
_SS_PUDDING_STEP_ORDER: list[str] = [
    "mixing", "heating", "holding", "cooling", "filling_packaging",
]

_SS_WINDOW_MIN: float = 30.0   # look-back window (minutes) for feature stats


def _ss_resolve_checkpoint(product: str, selected_step: str) -> str | None:
    """Return the last checkpoint at or before selected_step; full_run → latest."""
    if product == "QUARK":
        checkpoints = _QUARK_CHECKPOINTS
        step_order  = _SS_QUARK_STEP_ORDER
    else:
        checkpoints = _PUDDING_CHECKPOINTS
        step_order  = _SS_PUDDING_STEP_ORDER

    if selected_step == "full_run":
        return checkpoints[-1]

    sel_idx = step_order.index(selected_step) if selected_step in step_order else len(step_order) - 1
    for cp in reversed(checkpoints):
        if cp in step_order and step_order.index(cp) <= sel_idx:
            return cp
    return None


def _ss_extract_features(
    run_ts:          pd.DataFrame,
    checkpoint_step: str,
    product:         str,
) -> "np.ndarray | None":
    """Feature vector from run_ts up to end of checkpoint_step.

    [elapsed_t_end] + per-signal [mean, std, min, max, slope] over the last
    _SS_WINDOW_MIN minutes (or full available window if shorter).
    Returns None when the checkpoint step has no rows in run_ts.
    """
    if run_ts.empty or "step" not in run_ts.columns:
        return None

    step_mask = run_ts["step"] == checkpoint_step
    if not step_mask.any():
        return None

    t_end  = float(run_ts.loc[step_mask, "t_min"].max())
    scope  = run_ts[run_ts["t_min"] <= t_end]
    t_win  = max(float(scope["t_min"].min()), t_end - _SS_WINDOW_MIN)
    window = scope[scope["t_min"] >= t_win]

    if window.empty:
        return None

    signals = _SS_CHECKPOINT_SIGNALS.get(product, {}).get(checkpoint_step, [])
    if not signals:
        return None

    feats: list[float] = [t_end]   # elapsed time at checkpoint

    for sig in signals:
        if sig not in window.columns:
            feats.extend([0.0] * 5)
            continue

        vals = window[sig].dropna()
        if vals.empty:
            feats.extend([0.0] * 5)
            continue

        v_arr = vals.to_numpy(dtype=float)
        t_arr = window.loc[vals.index, "t_min"].to_numpy(dtype=float)

        feats.append(float(np.mean(v_arr)))
        feats.append(float(np.std(v_arr)))
        feats.append(float(np.min(v_arr)))
        feats.append(float(np.max(v_arr)))
        slope = (
            float(np.polyfit(t_arr - t_arr[0], v_arr, 1)[0])
            if len(t_arr) >= 2 and np.ptp(t_arr) > 0
            else 0.0
        )
        feats.append(slope)

    arr = np.array(feats, dtype=float)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


@st.cache_resource
def _train_soft_sensor_models() -> dict:
    """Train per-(product, checkpoint, target) Ridge regressors.

    Loads data internally and trains once per session.
    Returns nested dict: result[product][checkpoint][target] = fitted Pipeline.
    """
    if not _SKLEARN_AVAILABLE:
        return {}

    from sklearn.linear_model import Ridge           # noqa: PLC0415
    from sklearn.pipeline import Pipeline            # noqa: PLC0415
    from sklearn.preprocessing import StandardScaler  # noqa: PLC0415

    runs = load_runs()
    ts   = load_timeseries()
    lab  = load_lab()

    result: dict = {}

    for product, checkpoints in [
        ("QUARK",                _QUARK_CHECKPOINTS),
        ("HIGH_PROTEIN_PUDDING", _PUDDING_CHECKPOINTS),
    ]:
        result[product] = {}
        prod_runs = runs[runs["product"] == product]

        for checkpoint in checkpoints:
            result[product][checkpoint] = {}

            X_rows: list[np.ndarray] = []
            y_cols: dict[str, list[float]] = {t: [] for t in _SOFT_SENSOR_TARGETS}

            for rid in prod_runs["run_id"]:
                run_ts  = ts[ts["run_id"] == rid]
                lab_sub = lab[lab["run_id"] == rid]
                if run_ts.empty or lab_sub.empty:
                    continue

                feats = _ss_extract_features(run_ts, checkpoint, product)
                if feats is None:
                    continue

                X_rows.append(feats)
                lab_vals = lab_sub.iloc[0]
                for t in _SOFT_SENSOR_TARGETS:
                    v = lab_vals.get(t, np.nan)
                    y_cols[t].append(
                        float(v) if (v is not None and not pd.isna(v)) else np.nan
                    )

            if len(X_rows) < 5:
                continue

            X = np.array(X_rows, dtype=float)

            for target in _SOFT_SENSOR_TARGETS:
                y = np.array(y_cols[target], dtype=float)
                valid = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
                if valid.sum() < 5:
                    continue

                pipe = Pipeline([
                    ("scaler", StandardScaler()),
                    ("ridge",  Ridge(alpha=1.0)),
                ])
                pipe.fit(X[valid], y[valid])
                result[product][checkpoint][target] = pipe

    return result


def _ss_predict(
    run_ts:     pd.DataFrame,
    product:    str,
    checkpoint: str,
) -> "dict[str, float | None]":
    """Return predicted lab values for run_ts up to checkpoint."""
    models    = _train_soft_sensor_models()
    cp_models = models.get(product, {}).get(checkpoint, {})
    if not cp_models:
        return {}

    feats = _ss_extract_features(run_ts, checkpoint, product)
    if feats is None:
        return {}

    X = np.array([feats], dtype=float)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    preds: dict[str, float | None] = {}
    for target, model in cp_models.items():
        try:
            preds[target] = float(model.predict(X)[0])
        except Exception:
            preds[target] = None
    return preds


def _render_soft_sensor_panel(
    run_ts:        "pd.DataFrame",
    product:       str,
    selected_step: str,
    lab_row:       "pd.Series | None",
    df_baseline:   "pd.DataFrame | None",
) -> None:
    """Render Predicted Final Lab Outcomes (Soft Sensor) section."""
    st.subheader("Predicted Final Lab Outcomes (Soft Sensor)")

    if not _SKLEARN_AVAILABLE:
        st.caption("scikit-learn not installed — soft sensor unavailable.")
        return

    if run_ts.empty:
        return

    checkpoint = _ss_resolve_checkpoint(product, selected_step)
    if checkpoint is None:
        st.caption(
            "Process has not yet reached the first prediction checkpoint for this product."
        )
        return

    preds = _ss_predict(run_ts, product, checkpoint)
    if not preds:
        st.caption("Soft sensor model not available for this product/checkpoint.")
        return

    cp_label = checkpoint.replace("_", " ")
    st.caption(
        f"Checkpoint: **end of {cp_label}**  ·  "
        "_Forecast based on early process patterns; supports early "
        "adjustment/hypothesis, not a release decision._"
    )

    baseline_preds: dict[str, float | None] = {}
    if df_baseline is not None and not df_baseline.empty:
        baseline_preds = _ss_predict(df_baseline, product, checkpoint)

    cols = st.columns(len(_SS_TARGET_DISPLAY))
    for col_ui, (target, label, fmt) in zip(cols, _SS_TARGET_DISPLAY):
        pred = preds.get(target)

        actual: float | None = None
        if lab_row is not None:
            v = (
                lab_row.get(target) if hasattr(lab_row, "get")
                else (lab_row[target] if target in lab_row.index else None)
            )
            if v is not None and not (isinstance(v, float) and pd.isna(v)):
                actual = float(v)

        pred_str  = fmt.format(pred) if pred is not None else "n/a"
        delta_str: str | None = None
        if pred is not None and actual is not None:
            delta_str = f"{pred - actual:+.2f} vs actual"

        col_ui.metric(label, pred_str, delta=delta_str)

        if actual is not None:
            col_ui.caption(f"Actual: {fmt.format(actual)}")

        bp = baseline_preds.get(target)
        if bp is not None:
            col_ui.caption(f"Normal pred: {fmt.format(bp)}")

    st.divider()


# =============================================================================
# MEASUREMENT QUALITY PANEL
# =============================================================================

# Per-product signal configs: (col, label, step_scope)
# step_scope = None means "all rows where the column is not structurally null"
_MQ_QUARK_SIGNALS: list[tuple[str, str, str | None]] = [
    ("temperature_C",       "Temperature",      None),
    ("pressure_bar",        "Pressure",         None),
    ("pH",                  "pH",               None),
    ("flow_rate_lpm",       "Flow rate",        None),
    ("separator_speed_rpm", "Centrifuge speed", "separation"),
    ("separation_deltaP",   "Sep. ΔP",          "separation"),
]

_MQ_PUDDING_SIGNALS: list[tuple[str, str, str | None]] = [
    ("temperature_C",         "Temperature", None),
    ("pressure_bar",          "Pressure",    None),
    ("pH",                    "pH",          None),
    ("flow_rate_lpm",         "Flow rate",   None),
    ("deltaT_heat_exchanger", "HX ΔT",       None),
]

_MQ_SPIKE_Z       = 3.5   # z-score threshold for spike detection
_MQ_SPIKE_SUSPECT = 0.01  # 1 % spike rate → Suspect
_MQ_SPIKE_POOR    = 0.05  # 5 % spike rate → Poor
_MQ_MISS_SUSPECT  = 0.05  # 5 % missingness → Suspect
_MQ_MISS_POOR     = 0.20  # 20 % missingness → Poor


def _compute_measurement_quality(
    run_ts:  pd.DataFrame,
    product: str,
) -> list[dict]:
    """Return one dict per signal with spike/missingness statistics.

    Each dict: {col, label, n_rows, spike_count, miss_count,
                spike_rate, miss_rate, status, triggered}
    status in {"OK", "Suspect", "Poor"}; triggered is a human-readable note.
    """
    configs = _MQ_QUARK_SIGNALS if product == "QUARK" else _MQ_PUDDING_SIGNALS
    results = []

    for col, label, step_scope in configs:
        # Scope to the relevant step rows if specified
        if step_scope is not None:
            scope = run_ts[run_ts["step"] == step_scope]
        else:
            scope = run_ts

        if scope.empty or col not in scope.columns:
            continue

        series = scope[col]
        n_rows    = len(series)
        miss_count = int(series.isna().sum())
        valid      = series.dropna()

        # Per-step z-score spike detection on valid values
        step_col = scope.loc[valid.index, "step"]
        step_mean = step_col.map(valid.groupby(step_col).mean())
        step_std  = step_col.map(valid.groupby(step_col).std())
        z = (valid - step_mean).abs() / step_std.replace(0, np.nan)
        spike_count = int((z > _MQ_SPIKE_Z).sum())

        spike_rate = spike_count / n_rows if n_rows > 0 else 0.0
        miss_rate  = miss_count  / n_rows if n_rows > 0 else 0.0

        # Status logic: worst of spike / missingness
        triggered_parts: list[str] = []
        if spike_rate >= _MQ_SPIKE_POOR:
            spike_status = "Poor"
            triggered_parts.append(f"{label} spikes ({spike_rate:.1%})")
        elif spike_rate >= _MQ_SPIKE_SUSPECT:
            spike_status = "Suspect"
            triggered_parts.append(f"{label} spikes ({spike_rate:.1%})")
        else:
            spike_status = "OK"

        if miss_rate >= _MQ_MISS_POOR:
            miss_status = "Poor"
            triggered_parts.append(f"{label} dropouts ({miss_rate:.1%})")
        elif miss_rate >= _MQ_MISS_SUSPECT:
            miss_status = "Suspect"
            triggered_parts.append(f"{label} dropouts ({miss_rate:.1%})")
        else:
            miss_status = "OK"

        rank = {"OK": 0, "Suspect": 1, "Poor": 2}
        status = max(spike_status, miss_status, key=lambda s: rank[s])

        results.append({
            "col":         col,
            "label":       label,
            "n_rows":      n_rows,
            "spike_count": spike_count,
            "miss_count":  miss_count,
            "spike_rate":  spike_rate,
            "miss_rate":   miss_rate,
            "status":      status,
            "triggered":   "; ".join(triggered_parts) if triggered_parts else "",
        })

    return results


def _render_measurement_quality_panel(
    run_ts:  pd.DataFrame,
    product: str,
    story:   "dict | None",
) -> None:
    """Render the Measurement Quality panel."""
    if run_ts.empty:
        return

    st.subheader("Measurement Quality (in-/at-line)")

    if story and story.get("story_id") == "cross_sensor_fault":
        st.info("This story highlights measurement integrity.")

    st.caption(
        "This panel assesses sensor data quality (spikes, dropouts) and does **not** "
        "imply a process fault. Measurement issues are handled separately from process "
        "Early Warning scores."
    )

    signal_rows = _compute_measurement_quality(run_ts, product)
    if not signal_rows:
        st.caption("No in-/at-line signal data available.")
        return

    # Overall status = worst individual status
    rank = {"OK": 0, "Suspect": 1, "Poor": 2}
    overall = max(signal_rows, key=lambda r: rank[r["status"]])["status"]
    colour  = {"OK": "normal", "Suspect": "off", "Poor": "inverse"}
    icon    = {"OK": "checkmark", "Suspect": "warning", "Poor": "error"}
    st.metric(
        "Overall measurement quality",
        overall,
        delta=None,
        label_visibility="visible",
    )

    # List triggered signals
    flagged = [r for r in signal_rows if r["triggered"]]
    if flagged:
        bullets = "  \n".join(f"- {r['triggered']}" for r in flagged)
        st.markdown(bullets)

    # Per-signal breakdown in an expander
    with st.expander("Per-signal breakdown", expanded=(overall != "OK")):
        display_rows = []
        for r in signal_rows:
            status_icon = {"OK": "✅", "Suspect": "⚠️", "Poor": "🔴"}.get(r["status"], "")
            display_rows.append({
                "Signal":       r["label"],
                "Status":       f"{status_icon} {r['status']}",
                "Spike rate":   f"{r['spike_rate']:.1%}",
                "Dropout rate": f"{r['miss_rate']:.1%}",
                "Rows checked": r["n_rows"],
            })
        st.dataframe(
            pd.DataFrame(display_rows),
            hide_index=True,
            use_container_width=True,
        )

    st.divider()


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
            df_sim  = ts[ts["run_id"] == sr["run_id"]]
            stab, _, sim_badge = _compute_run_stability(df_sim, target_product)
            col_info, col_btn = st.columns([3, 1])
            with col_info:
                st.markdown(
                    f"**{sr['run_id']}** · {sr['scenario']} · {sr['scale']}  \n"
                    f"Similarity: **{sim_pct}%**  ·  "
                    f"Stability: **{stab}/100** {sim_badge}  ·  "
                    f"Downtime: {sr['downtime_minutes']:.0f} min  ·  "
                    f"Yield loss: {sr['yield_loss_pct']:.1f}%"
                )
                badges: list[str] = []
                if sr["extra_cleaning"]:
                    badges.append("extra CIP")
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
    runs:        pd.DataFrame,
    demo_cases:  dict,
    sync_issues: list[dict] | None = None,
) -> tuple[str, str | None]:
    """
    Render sidebar controls.

    Returns
    -------
    mode     : "guided" or "explore"
    run_id   : the selected run_id, or None if nothing is selectable yet
    """
    st.sidebar.title("Controls")

    if sync_issues:
        st.sidebar.error(
            "Guided story list is out of sync with the current dataset.  \n"
            "Please regenerate demo_cases.json."
        )
        with st.sidebar.expander("Sync issues", expanded=False):
            st.dataframe(
                pd.DataFrame(sync_issues),
                hide_index=True,
                use_container_width=True,
            )

    mode = st.sidebar.radio(
        "Mode",
        options=["Guided story mode", "Explore all runs"],
        key="app_mode",
        disabled=bool(sync_issues),
        help=(
            "Guided: step through curated demo cases with process narratives.\n"
            "Explore: browse any run in the dataset."
        ),
    )

    # Force explore when stories are out of sync
    guided = (mode == "Guided story mode") and not sync_issues

    if sync_issues:
        st.sidebar.caption("Guided mode unavailable until demo_cases.json is regenerated.")

    st.sidebar.divider()

    if guided:
        run_id = _sidebar_guided(demo_cases)
    else:
        run_id = _sidebar_explore(runs)

    st.sidebar.divider()
    st.sidebar.markdown("**Advanced panels**")
    st.sidebar.toggle(
        "Show soft sensor predictions (advanced)",
        key="show_soft_sensor",
        value=False,
    )
    st.sidebar.toggle(
        "Show similar runs (advanced)",
        key="show_similar_runs",
        value=False,
    )

    return ("guided" if guided else "explore"), run_id


def _clear_run_widgets(run_id: str) -> None:
    """Remove per-run session state so the destination run starts fresh."""
    for key in (
        f"step_zoom_{run_id}", f"baseline_{run_id}",
        f"score_trend_{run_id}", f"ferm_trend_{run_id}", f"sep_trend_{run_id}",
    ):
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
        if st.session_state.get("show_soft_sensor", False):
            _render_ml_score_panel(run_ts, run_row["product"], x_range, selected_step, run_id)
            _render_score_trend_chart(run_ts, run_row["product"], run_evts, x_range, selected_step, run_id)
            _render_soft_sensor_panel(run_ts, run_row["product"], selected_step, lab_row, df_baseline)
        _render_measurement_quality_panel(run_ts, run_row["product"], story)
        render_signal_charts(run_ts, run_row["product"], run_evts, runs, ts)

    with col_right:
        _render_right_panel(mode, run_row, lab_row, story, df_run=run_ts)
        if df_baseline is not None:
            _render_divergence_panel(
                run_ts, df_baseline, run_row["product"], x_range, selected_step,
            )
        if st.session_state.get("show_similar_runs", False):
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
    df_run:  pd.DataFrame | None = None,
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
        o1.metric("Yield loss", f"{run_row['yield_loss_pct']:.1f} %")
        o2.metric("Downtime",   f"{run_row['downtime_minutes']:.0f} min")

        if df_run is not None and not df_run.empty and _SKLEARN_AVAILABLE:
            product = run_row["product"]
            if product == "QUARK":
                ferm = _compute_quark_ferm_score(df_run)
                sep  = _compute_quark_sep_score(df_run)
                f_stab = _anomaly_to_stability(ferm[0]) if ferm else 100
                s_stab = _anomaly_to_stability(sep[0])  if sep  else 100
                o1.metric(
                    "Fermentation Stability (0–100)",
                    f"{f_stab} / 100",
                    help="Derived from the Fermentation Early Warning (ML) score, normalised for reporting.",
                )
                o2.metric(
                    "Separation Stability (0–100)",
                    f"{s_stab} / 100",
                    help="Derived from the Separation Early Warning (ML) score, normalised for reporting.",
                )
                st.caption("ML status details are shown in the Early Warning section below.")
            else:
                stab, _, _ = _compute_run_stability(df_run, product)
                o1.metric(
                    "Run Stability Score (0–100)",
                    f"{stab} / 100",
                    help="Derived from the ML early warning score, normalised for reporting.",
                )
            o2.metric("Extra CIP", "Yes" if run_row["extra_cleaning"] else "No")
        else:
            o2.metric("Extra CIP", "Yes" if run_row["extra_cleaning"] else "No")

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


def _render_lab_metrics(lab_row: pd.Series, product: str) -> None:  # noqa: ARG001
    """Show the 5 core lab QC fields — identical layout for both products."""
    _NA = "Not available in this PoC"

    def _fmt(col: str, fmt: str) -> str:
        v = lab_row.get(col)
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return _NA
        return fmt.format(v)

    st.metric("Protein",                     _fmt("protein_pct",      "{:.2f} %"))
    st.metric("Total solids",                _fmt("total_solids_pct", "{:.2f} %"))
    st.metric("Viscosity",                   _fmt("viscosity_value",  "{:.0f} cP"))
    st.metric("pH (offline)",                _fmt("final_pH_offline", "{:.3f}"))
    st.metric("D50 (laser diffraction, µm)", _fmt("d50_um",           "{:.1f}"))


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

    sync_issues = _check_demo_cases_sync(demo_cases, runs)

    mode, run_id = render_sidebar(runs, demo_cases, sync_issues)
    render_main(mode, run_id, runs, ts, lab, events, demo_cases)


main()
