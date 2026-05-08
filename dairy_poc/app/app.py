"""
Dairy Process Investigation PoC — Streamlit app skeleton.

Run from the dairy_poc/ directory:
    streamlit run app/app.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

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
    df_run:     pd.DataFrame,
    col:        str,
    label:      str,
    unit:       str,
    color:      str,
    show_xaxis: bool = False,
    evts:       pd.DataFrame | None = None,
) -> go.Figure:
    """Return a single compact Plotly line chart for one sensor signal."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_run["t_min"],
        y=df_run[col],
        mode="lines",
        line=dict(color=color, width=1.8),
        name=label,
        connectgaps=False,      # NaN → visible gap in the line
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
        showlegend=False,
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    if evts is not None and not evts.empty:
        _add_event_markers(fig, evts, col, df_run)
    return fig


def render_signal_charts(
    df_run:   pd.DataFrame,
    product:  str,
    run_evts: pd.DataFrame,
) -> None:
    """Render one Plotly line chart per signal, stacked vertically."""
    signals = _QUARK_SIGNALS if product == "QUARK" else _PUDDING_SIGNALS

    # Keep only signals that exist in the DataFrame and have at least one value
    renderable = [
        sig for sig in signals
        if sig[0] in df_run.columns and not df_run[sig[0]].isna().all()
    ]

    st.subheader("Sensor charts")

    if not renderable:
        st.info("No signal data available for this run.")
        return

    evt_filter  = _QUARK_EVENTS if product == "QUARK" else _PUDDING_EVENTS
    evts_to_show = run_evts[run_evts["event_type"].isin(evt_filter)]

    for i, (col, label, unit, color) in enumerate(renderable):
        is_last = (i == len(renderable) - 1)
        fig = _make_signal_chart(
            df_run, col, label, unit, color,
            show_xaxis=is_last,
            evts=evts_to_show,
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


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
        index=0,
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


def _sidebar_guided(demo_cases: dict) -> str | None:
    """Story-mode sidebar: one selectbox over curated runs."""
    stories = demo_cases.get("stories", [])
    if not stories:
        st.sidebar.warning("No demo cases found in demo_cases.json.")
        return None

    labels = [
        f"{s['run_id']}  ·  {s['short_title']}"
        for s in stories
    ]
    idx = st.sidebar.selectbox(
        "Select story",
        options=range(len(labels)),
        format_func=lambda i: labels[i],
        label_visibility="collapsed",
    )
    return stories[idx]["run_id"]


def _sidebar_explore(runs: pd.DataFrame) -> str | None:
    """Explore-mode sidebar: cascading product → scenario → run selectors."""
    products = sorted(runs["product"].unique())
    product_display = {p: p.replace("HIGH_PROTEIN_PUDDING", "High-Protein Pudding") for p in products}

    sel_product = st.sidebar.selectbox(
        "Product",
        options=products,
        format_func=lambda p: product_display[p],
    )

    filtered_by_product = runs[runs["product"] == sel_product]
    scenarios = ["(all)"] + sorted(filtered_by_product["scenario"].unique())
    sel_scenario = st.sidebar.selectbox("Scenario", options=scenarios)

    if sel_scenario == "(all)":
        filtered = filtered_by_product
    else:
        filtered = filtered_by_product[filtered_by_product["scenario"] == sel_scenario]

    if filtered.empty:
        st.sidebar.info("No runs match the current filters.")
        return None

    run_labels = {
        row["run_id"]: f"{row['run_id']}  ·  {row['scenario']}  ·  {row['scale']}"
        for _, row in filtered.sort_values("run_id").iterrows()
    }
    sel_run = st.sidebar.selectbox(
        "Run",
        options=list(run_labels.keys()),
        format_func=lambda rid: run_labels[rid],
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

    # Build two-column layout: main content | right panel
    col_main, col_right = st.columns([2, 1], gap="large")

    with col_main:
        _render_run_header(run_row, story)
        _render_timeline_placeholder(run_ts)
        render_signal_charts(run_ts, run_row["product"], run_evts)

    with col_right:
        _render_right_panel(mode, run_row, lab_row, story)


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


# ── Timeline placeholder ──────────────────────────────────────────────────────

def _render_timeline_placeholder(run_ts: pd.DataFrame) -> None:
    steps    = run_ts["step"].unique().tolist() if not run_ts.empty else []
    n_rows   = len(run_ts)
    dur_min  = float(run_ts["t_min"].max()) if not run_ts.empty else 0.0

    st.subheader("Process timeline")
    with st.container(border=True):
        st.caption(
            f"Placeholder — {n_rows:,} timeseries rows  ·  "
            f"{dur_min:.0f} min  ·  steps: {', '.join(steps)}"
        )
        st.info(
            "Chart will go here: step-shaded timeline with sensor overlays "
            "(temperature, pH, separator_speed_rpm, fouling_index …)."
        )
        if not run_ts.empty:
            st.dataframe(
                run_ts[["t_min", "step", "temperature_C", "pH", "anomaly_flag"]]
                .head(5),
                use_container_width=True,
                hide_index=True,
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
