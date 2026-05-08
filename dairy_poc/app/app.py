"""
Dairy Process Investigation PoC — Streamlit app skeleton.

Run from the dairy_poc/ directory:
    streamlit run app/app.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
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

    story = _find_story(demo_cases, run_id)  # None in explore mode

    # Build two-column layout: main content | right panel
    col_main, col_right = st.columns([2, 1], gap="large")

    with col_main:
        _render_run_header(run_row, story)
        _render_timeline_placeholder(run_id, ts)
        _render_charts_placeholder(run_id, ts, run_row)

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

def _render_timeline_placeholder(run_id: str, ts: pd.DataFrame) -> None:
    run_ts   = ts[ts["run_id"] == run_id]
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
        # Debug wire-check
        st.write("**Selected run_id (wire check):**", run_id)
        if not run_ts.empty:
            st.dataframe(
                run_ts[["t_min", "step", "temperature_C", "pH", "anomaly_flag"]]
                .head(5),
                use_container_width=True,
                hide_index=True,
            )


# ── Charts placeholder ────────────────────────────────────────────────────────

def _render_charts_placeholder(
    run_id:  str,
    ts:      pd.DataFrame,
    run_row: pd.Series,
) -> None:
    product  = run_row["product"]
    scenario = run_row["scenario"]

    # Decide which sensor panels are relevant so the placeholder is informative
    panels: list[str] = ["temperature_C", "pressure_bar + flow_rate_lpm"]
    if product == "QUARK":
        panels += ["pH (full run)", "separator_speed_rpm + separation_deltaP"]
    else:
        panels += ["fouling_index + deltaT_heat_exchanger", "viscosity_proxy"]

    st.subheader("Sensor charts")
    with st.container(border=True):
        st.caption(f"Placeholder — {scenario} · {product.replace('HIGH_PROTEIN_PUDDING','Pudding')}")
        st.info(
            "Charts will go here. Planned panels:\n"
            + "\n".join(f"  • {p}" for p in panels)
        )
        run_ts = ts[ts["run_id"] == run_id]
        if not run_ts.empty:
            anom_count = int(run_ts["anomaly_flag"].sum())
            st.write(f"anomaly_flag rows: **{anom_count}** of {len(run_ts):,}")


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
