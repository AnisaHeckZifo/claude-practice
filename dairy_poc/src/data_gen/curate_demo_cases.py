"""
Curate a fixed demo-case set from the dairy PoC synthetic dataset.

Selects 10–12 illustrative runs covering key failure modes and process stories,
then writes data_processed/demo_cases.json.

Usage (from dairy_poc/ directory):
    python -m src.data_gen.curate_demo_cases
    python -m src.data_gen.curate_demo_cases --demo_mode
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

_ROOT     = Path(__file__).parents[2]
_DATA_DIR = _ROOT / "data_raw"
_OUT_DIR  = _ROOT / "data_processed"


# =============================================================================
# STORY PLAN
# Each tuple: (story_id, product | None, scenario_priority_list)
# None product = any product; scenarios tried in order, first match wins.
# =============================================================================

_STORY_PLAN: list[tuple[str, str | None, list[str]]] = [
    ("quark_normal",       "QUARK",                ["NORMAL"]),
    ("quark_over_acid",    "QUARK",                ["OVER_ACID"]),
    ("quark_stall",        "QUARK",                ["STALL_FERMENT", "SLOW_FERMENT"]),
    ("quark_separation",   "QUARK",                ["SEPARATION_ISSUE", "STANDARDIZATION_OFFTARGET"]),
    ("quark_raw_mat",      "QUARK",                ["RAW_MAT_VAR"]),
    ("pudding_normal",     "HIGH_PROTEIN_PUDDING", ["NORMAL"]),
    ("pudding_block",      "HIGH_PROTEIN_PUDDING", ["BLOCK"]),
    ("pudding_foul_1",     "HIGH_PROTEIN_PUDDING", ["FOUL", "DRIFT"]),
    ("pudding_foul_2",     "HIGH_PROTEIN_PUDDING", ["FOUL", "DRIFT"]),
    ("cross_sensor_fault", None,                   ["SENSOR_FAULT"]),
    ("cross_drift",        None,                   ["DRIFT"]),
]


# =============================================================================
# CASE METADATA
# Key: (product_key, scenario)  where product_key in
#      {"QUARK", "HIGH_PROTEIN_PUDDING", "ANY"}
# "ANY" is the fallback when no product-specific entry exists.
# =============================================================================

_CASE_META: dict[tuple[str, str], dict] = {

    # ── Quark ────────────────────────────────────────────────────────────────

    ("QUARK", "NORMAL"): {
        "short_title": "Quark: Healthy fermentation baseline",
        "narrative": (
            "Fermentation proceeds on schedule; pH reaches the 4.5 target within the "
            "expected 8–12 h window. Gel break and centrifuge separation complete cleanly, "
            "and the product leaves standardisation within spec."
        ),
        "what_to_watch": [
            "pH in late_fermentation: smooth sigmoid descent to ~4.5 over 8–12 h",
            "separator_speed_rpm: sigmoidal ramp-up at the start of the separation step",
            "separation_deltaP: stable near 0.6 bar with no upward drift",
            "lab: protein_pct in the 10–14 % band and result_flag = PASS",
        ],
        "key_events_expected": [
            "inoculation", "rennet_addition", "gel_break_start",
            "separation_start", "standardization_start", "filling_start",
        ],
        "critical_steps": ["late_fermentation", "gel_break_mixing", "separation"],
    },

    ("QUARK", "OVER_ACID"): {
        "short_title": "Quark: Over-acidified curd",
        "narrative": (
            "Culture activity runs faster than target, driving final pH below 4.4. "
            "The over-acidified curd shows reduced viscosity and a lower texture score at lab sampling."
        ),
        "what_to_watch": [
            "pH: steeper-than-normal slope; final pH in the 4.10–4.35 range",
            "fermentation_time_hr: shorter than the NORMAL baseline (~10–11 h)",
            "lab: viscosity_cP and texture_score below normal range",
            "result_flag: FAIL on final_pH spec limit",
        ],
        "key_events_expected": [
            "inoculation", "rennet_addition", "gel_break_start", "separation_start",
        ],
        "critical_steps": ["late_fermentation"],
    },

    ("QUARK", "STALL_FERMENT"): {
        "short_title": "Quark: Stalled fermentation — run scrapped",
        "narrative": (
            "Acidification halts at pH ~5.5 roughly one-third of the way through late fermentation. "
            "The batch is abandoned; yield_loss exceeds 80 % and fermentation_time_hr is recorded "
            "as 99 (did-not-complete sentinel)."
        ),
        "what_to_watch": [
            "pH in late_fermentation: drops to ~5.5 then flat-lines for the remainder",
            "anomaly_flag = 1 across the entire late_fermentation step",
            "lab: fermentation_time_hr = 99.0 sentinel; protein_pct far below spec",
            "yield_loss_pct: 80–100 % in run metadata",
        ],
        "key_events_expected": ["inoculation", "rennet_addition"],
        "critical_steps": ["late_fermentation"],
    },

    ("QUARK", "SLOW_FERMENT"): {
        "short_title": "Quark: Sluggish culture — extended timeline",
        "narrative": (
            "Culture activity is below normal, slowing the pH descent in late fermentation. "
            "The run completes but runs 2–3 h long, compressing the downstream schedule."
        ),
        "what_to_watch": [
            "pH: shallower slope than NORMAL baseline; final pH 4.55–4.72",
            "fermentation_time_hr: 15–16 h vs. ~13 h for a normal run",
            "total_duration_hr: visibly extended run in the step timeline",
        ],
        "key_events_expected": [
            "inoculation", "rennet_addition", "gel_break_start",
        ],
        "critical_steps": ["late_fermentation"],
    },

    ("QUARK", "SEPARATION_ISSUE"): {
        "short_title": "Quark: Centrifuge underperformance",
        "narrative": (
            "Centrifuge back-pressure drifts upward during separation, restricting throughput "
            "and reducing whey removal efficiency. Final protein content falls below the 10 % "
            "lower spec limit."
        ),
        "what_to_watch": [
            "separation_deltaP: rising trend across the full separation step",
            "flow_rate_lpm: declining as back-pressure increases",
            "anomaly_flag = 1 throughout the separation step",
            "lab: protein_pct below 10 %; whey_protein_loss_proxy elevated",
        ],
        "key_events_expected": ["gel_break_start", "separation_start"],
        "critical_steps": ["gel_break_mixing", "separation"],
    },

    ("QUARK", "STANDARDIZATION_OFFTARGET"): {
        "short_title": "Quark: Off-target standardisation dosing",
        "narrative": (
            "Dosing pump variability during standardisation produces above-normal spread in "
            "product protein content, with some batches landing outside the 10–14 % target band."
        ),
        "what_to_watch": [
            "flow_rate_lpm and shear_rpm: elevated variability in standardization_mix",
            "anomaly_flag = 1 rows during the standardisation step",
            "lab: protein_pct spread wider than NORMAL; result_flag may be WARN or FAIL",
        ],
        "key_events_expected": ["standardization_start"],
        "critical_steps": ["standardization_mix"],
    },

    ("QUARK", "RAW_MAT_VAR"): {
        "short_title": "Quark: Raw-milk protein variability",
        "narrative": (
            "The incoming milk blend has an off-baseline protein content, which propagates "
            "through fermentation rate and into the final product protein level."
        ),
        "what_to_watch": [
            "raw_protein_pct in run metadata: deviation from the 3.2 % baseline",
            "milk_tank_ids and blend_ratio: multi-tank blending compounds variation",
            "fermentation_time_hr: shorter with higher raw protein (richer culture substrate)",
            "lab: protein_pct tracking the raw material — compare against the NORMAL baseline",
        ],
        "key_events_expected": ["inoculation", "rennet_addition"],
        "critical_steps": ["early_fermentation", "late_fermentation"],
    },

    # ── High-Protein Pudding ──────────────────────────────────────────────────

    ("HIGH_PROTEIN_PUDDING", "NORMAL"): {
        "short_title": "Pudding: Clean heat-treatment baseline",
        "narrative": (
            "Mixing, heating, and holding all complete within setpoint tolerances. "
            "No fouling signal appears on the heat exchanger; the product reaches fill "
            "temperature on target."
        ),
        "what_to_watch": [
            "temperature_C in heating + holding: stable at the 88 °C setpoint",
            "deltaT_heat_exchanger: flat and low (~8–12 °C) — no deposit build-up",
            "fouling_index: near zero across all steps",
            "lab: fouling_index_end < 0.05 and result_flag = PASS",
        ],
        "key_events_expected": ["filling_start"],
        "critical_steps": ["heating", "holding"],
    },

    ("HIGH_PROTEIN_PUDDING", "BLOCK"): {
        "short_title": "Pudding: Partial line blockage",
        "narrative": (
            "A partial obstruction develops mid-run in one of the transfer lines, causing a "
            "sudden flow drop and a compensatory pressure spike lasting up to 25 minutes."
        ),
        "what_to_watch": [
            "flow_rate_lpm: abrupt step-down to ~45 % of normal during the blockage window",
            "pressure_bar: spike of ~0.9 bar above baseline coinciding with the flow drop",
            "anomaly_flag = 1 for the full duration of the blockage event",
            "downtime_minutes and yield_loss_pct elevated in run metadata",
        ],
        "key_events_expected": ["filling_start"],
        "critical_steps": ["heating", "cooling"],
    },

    ("HIGH_PROTEIN_PUDDING", "FOUL"): {
        "short_title": "Pudding: Progressive heat-exchanger fouling",
        "narrative": (
            "Protein and mineral deposits accumulate on heat-exchanger surfaces during heating "
            "and holding. The fouling signal builds progressively: outlet temperature drops, "
            "back-pressure rises, and deltaT across the exchanger climbs — triggering an extra "
            "CIP cycle."
        ),
        "what_to_watch": [
            "fouling_index: ramp from 0 to 0.25–0.55 across the heating + holding steps",
            "deltaT_heat_exchanger: climbing from the ~8 °C baseline toward 15–20 °C",
            "temperature_C: outlet drifting ~4 °C below setpoint as fouling intensifies",
            "pressure_bar: back-pressure rise of ~0.8 bar; extra_cleaning = 1 in run metadata",
        ],
        "key_events_expected": ["filling_start"],
        "critical_steps": ["heating", "holding"],
    },

    ("HIGH_PROTEIN_PUDDING", "DRIFT"): {
        "short_title": "Pudding: Gradual process drift (FOUL substitute)",
        "narrative": (
            "Sensor readings drift slowly over the course of the run — temperature creeps "
            "upward while flow rate declines — indicating a gradually degrading process "
            "condition. Included here as a substitute for a second FOUL example."
        ),
        "what_to_watch": [
            "temperature_C: slow upward trend visible after ~20 % of run elapsed",
            "flow_rate_lpm: gradual decline correlated with temperature rise",
            "anomaly_flag: sparse early in the run, denser toward end",
        ],
        "key_events_expected": ["filling_start"],
        "critical_steps": ["heating", "holding"],
    },

    # ── Cross-cutting ─────────────────────────────────────────────────────────

    ("ANY", "SENSOR_FAULT"): {
        "short_title": "Cross-product: Instrument spike anomalies",
        "narrative": (
            "Intermittent sensor faults produce isolated temperature spikes — either well above "
            "or below the expected process range — flagged by anomaly_flag = 1 in the timeseries."
        ),
        "what_to_watch": [
            "temperature_C: isolated ±12–14 °C spikes clearly inconsistent with step context",
            "anomaly_flag = 1: a small number of isolated rows (~1–3 per affected step)",
            "Other sensors unaffected — fault localised to the temperature instrument",
        ],
        "key_events_expected": [],
        "critical_steps": [],
    },

    ("ANY", "DRIFT"): {
        "short_title": "Cross-product: Gradual process drift",
        "narrative": (
            "Process conditions drift slowly across the run: temperature rises and flow rate "
            "declines in a time-correlated pattern typical of a gradually degrading process."
        ),
        "what_to_watch": [
            "temperature_C: slow upward trend becoming visible after ~20 % of run elapsed",
            "flow_rate_lpm: gradual decline correlated with temperature rise",
            "anomaly_flag: sparse early in the run, denser toward the end",
        ],
        "key_events_expected": [],
        "critical_steps": [],
    },
}


# =============================================================================
# DATA LOADING
# =============================================================================

def _load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    runs  = pd.read_csv(_DATA_DIR / "runs.csv")
    ts    = pd.read_csv(_DATA_DIR / "timeseries.csv")
    lab   = pd.read_csv(_DATA_DIR / "lab_results.csv")
    evts  = pd.read_csv(_DATA_DIR / "events.csv")
    return runs, ts, lab, evts


# =============================================================================
# SELECTION
# =============================================================================

def _pick_run(
    runs_df:   pd.DataFrame,
    product:   str | None,
    scenarios: list[str],
    used:      set[str],
) -> tuple[str | None, str, str]:
    """Try each scenario in order. Return (run_id, matched_scenario, why) or (None, '', '')."""
    for scenario in scenarios:
        mask = runs_df["scenario"] == scenario
        if product is not None:
            mask &= runs_df["product"] == product
        mask &= ~runs_df["run_id"].isin(used)

        candidates = runs_df[mask]
        if candidates.empty:
            continue

        # Prefer PRODUCTION scale: lower noise_multiplier gives cleaner demo signals
        prod = candidates[candidates["scale"] == "PRODUCTION"]
        pool = prod if not prod.empty else candidates
        row  = pool.iloc[0]

        n   = len(candidates)
        why = (
            f"{scenario} {row['product'].replace('HIGH_PROTEIN_PUDDING', 'PUDDING')}"
            f" — {'only match' if n == 1 else f'{n} available'},"
            f" scale={row['scale']}"
        )
        return str(row["run_id"]), scenario, why

    return None, "", ""


def _build_run_summary(
    run_id:  str,
    runs_df: pd.DataFrame,
    lab_df:  pd.DataFrame,
) -> dict:
    run = runs_df[runs_df["run_id"] == run_id].iloc[0]
    lab = lab_df[lab_df["run_id"] == run_id].iloc[0]

    summary: dict = {
        "total_duration_hr": round(float(run["total_duration_hr"]), 3),
        "result_flag":       str(lab["result_flag"]),
        "yield_loss_pct":    round(float(run["yield_loss_pct"]), 3),
        "scale":             str(run["scale"]),
    }
    # Quark-specific
    if pd.notna(run["gel_break_time_min"]):
        summary["gel_break_time_min"] = round(float(run["gel_break_time_min"]), 1)
    if pd.notna(lab["fermentation_time_hr"]):
        summary["fermentation_time_hr"] = round(float(lab["fermentation_time_hr"]), 2)
    if pd.notna(lab["final_pH"]):
        summary["final_pH"] = round(float(lab["final_pH"]), 3)
    # Pudding-specific
    if pd.notna(lab["fouling_index_end"]):
        summary["fouling_index_end"] = round(float(lab["fouling_index_end"]), 4)
    # Shared
    if pd.notna(run["downtime_minutes"]) and float(run["downtime_minutes"]) > 0:
        summary["downtime_minutes"] = round(float(run["downtime_minutes"]), 1)

    return summary


def _select_cases(
    runs_df: pd.DataFrame,
    lab_df:  pd.DataFrame,
) -> tuple[list[dict], list[str]]:
    """Select one run per story slot. Returns (cases, selection_warnings)."""
    used:     set[str]  = set()
    cases:    list[dict] = []
    warnings: list[str]  = []

    for story_id, product, scenarios in _STORY_PLAN:
        run_id, matched, why = _pick_run(runs_df, product, scenarios, used)

        if run_id is None:
            warnings.append(
                f"[WARN] No run found for story '{story_id}' "
                f"(tried: {', '.join(scenarios)}"
                + (f" for product={product}" if product else "")
                + ")"
            )
            continue

        used.add(run_id)
        row = runs_df[runs_df["run_id"] == run_id].iloc[0]
        prod_key = str(row["product"])

        # Resolve metadata: product-specific first, then ANY
        meta = (
            _CASE_META.get((prod_key, matched))
            or _CASE_META.get(("ANY", matched))
            or {}
        )

        cases.append({
            "story_id":            story_id,
            "run_id":              run_id,
            "product":             prod_key,
            "scale":               str(row["scale"]),
            "scenario":            matched,
            "why_selected":        why,
            "short_title":         meta.get("short_title", f"{matched} — {prod_key}"),
            "narrative":           meta.get("narrative", ""),
            "what_to_watch":       meta.get("what_to_watch", []),
            "key_events_expected": meta.get("key_events_expected", []),
            "_critical_steps":     meta.get("critical_steps", []),  # used for validation only
            "run_summary":         _build_run_summary(run_id, runs_df, lab_df),
        })

    return cases, warnings


# =============================================================================
# VALIDATION
# =============================================================================

def _validate_cases(
    cases:   list[dict],
    runs_df: pd.DataFrame,
    ts_df:   pd.DataFrame,
    lab_df:  pd.DataFrame,
    evts_df: pd.DataFrame,
) -> list[str]:
    """Check each selected run_id across all four files and verify critical steps."""
    ids_runs = set(runs_df["run_id"])
    ids_ts   = set(ts_df["run_id"])
    ids_lab  = set(lab_df["run_id"])
    ids_evts = set(evts_df["run_id"])

    warnings: list[str] = []

    for case in cases:
        rid = case["run_id"]

        for label, ids in [
            ("runs.csv",        ids_runs),
            ("timeseries.csv",  ids_ts),
            ("lab_results.csv", ids_lab),
            ("events.csv",      ids_evts),
        ]:
            if rid not in ids:
                warnings.append(f"[FAIL] {rid}: absent from {label}")

        if rid not in ids_ts:
            continue  # can't check steps if timeseries is missing

        steps_present = set(ts_df.loc[ts_df["run_id"] == rid, "step"].unique())
        for step in case["_critical_steps"]:
            if step not in steps_present:
                warnings.append(
                    f"[WARN] {rid} ({case['scenario']}): "
                    f"critical step '{step}' not found in timeseries"
                )

    return warnings


# =============================================================================
# CONSOLE OUTPUT
# =============================================================================

def _print_table(cases: list[dict], warnings: list[str]) -> None:
    col_run      = 12
    col_product  = 22
    col_scenario = 30

    header = (
        f"{'run_id':<{col_run}} "
        f"{'product':<{col_product}} "
        f"{'scenario':<{col_scenario}} "
        f"why selected"
    )
    sep = "-" * min(len(header), 100)

    print(f"\n-- Demo case selection ({len(cases)} runs) --")
    print(header)
    print(sep)

    for c in cases:
        prod_short = c["product"].replace("HIGH_PROTEIN_PUDDING", "PUDDING")
        print(
            f"{c['run_id']:<{col_run}} "
            f"{prod_short:<{col_product}} "
            f"{c['scenario']:<{col_scenario}} "
            f"{c['why_selected']}"
        )

    if warnings:
        print(f"\nWarnings ({len(warnings)}):")
        for w in warnings:
            print(f"  {w}")
    else:
        print("\n  All validation checks passed.")


# =============================================================================
# OPTIONAL: REGENERATE WITH BOOSTED FOUL PROBABILITY
# =============================================================================

def _regenerate_with_foul(n_runs: int, freq_min: float = 1.0, seed: int = 43) -> None:
    """
    Temporarily boost Pudding FOUL + DRIFT scenario probability in process_data._CFG,
    call generate_dataset, and overwrite the data_raw CSVs.

    Uses seed=43 (not the default 42) to produce a different draw from the same generator.
    Original probabilities are always restored, even if generation fails.
    """
    try:
        from . import process_data as _pd
        from .run_all import generate_dataset
    except ImportError:
        import process_data as _pd       # type: ignore[no-redef]
        from run_all import generate_dataset  # type: ignore[no-redef]

    pud  = _pd._CFG["scenarios_pudding"]
    orig = {s: pud[s]["probability"] for s in ("FOUL", "DRIFT") if s in pud}
    for s in orig:
        pud[s]["probability"] = min(0.40, orig[s] * 5.0)

    try:
        print(
            f"  Boosted Pudding FOUL→{pud.get('FOUL', {}).get('probability', '?'):.2f}, "
            f"DRIFT→{pud.get('DRIFT', {}).get('probability', '?'):.2f}; "
            f"regenerating {n_runs} runs (seed={seed}) ..."
        )
        dfs = generate_dataset(n_runs, seed, freq_min)
        for name, df in dfs.items():
            df.to_csv(_DATA_DIR / f"{name}.csv", index=False)

        n_foul = int(
            ((dfs["runs"]["product"] == "HIGH_PROTEIN_PUDDING") &
             (dfs["runs"]["scenario"] == "FOUL")).sum()
        )
        print(f"  Regenerated: {n_foul} Pudding FOUL run(s) in new dataset.")
    finally:
        for s, p in orig.items():
            pud[s]["probability"] = p


# =============================================================================
# ENTRY POINT
# =============================================================================

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Curate demo cases from the dairy PoC synthetic dataset"
    )
    ap.add_argument(
        "--demo_mode", action="store_true",
        help=(
            "If Pudding FOUL runs < 2, regenerate the dataset with boosted FOUL "
            "probability before curating. Default: curate from existing data only."
        ),
    )
    args = ap.parse_args()

    runs, ts, lab, evts = _load_data()
    print(
        f"Loaded: runs={len(runs)}  "
        f"timeseries={len(ts):,}  "
        f"lab={len(lab)}  "
        f"events={len(evts)}"
    )

    # Check FOUL availability ─────────────────────────────────────────────────
    n_foul = int(
        ((runs["product"] == "HIGH_PROTEIN_PUDDING") &
         (runs["scenario"] == "FOUL")).sum()
    )
    if n_foul < 2:
        if args.demo_mode:
            _regenerate_with_foul(n_runs=len(runs), freq_min=1.0)
            runs, ts, lab, evts = _load_data()
            n_foul_new = int(
                ((runs["product"] == "HIGH_PROTEIN_PUDDING") &
                 (runs["scenario"] == "FOUL")).sum()
            )
            if n_foul_new < 2:
                print(
                    f"  [WARN] Regeneration produced only {n_foul_new} FOUL run(s); "
                    "DRIFT will be used as substitute."
                )
        else:
            print(
                f"[WARN] Only {n_foul} Pudding FOUL run(s) found. "
                "DRIFT will substitute for the second FOUL slot. "
                "Re-run with --demo_mode to regenerate with boosted FOUL probability."
            )
    else:
        print(f"  {n_foul} Pudding FOUL run(s) available — no regeneration needed.")

    # Select and validate ─────────────────────────────────────────────────────
    cases, sel_warnings = _select_cases(runs, lab)
    val_warnings = _validate_cases(cases, runs, ts, lab, evts)
    all_warnings = sel_warnings + val_warnings

    _print_table(cases, all_warnings)

    # Strip internal validation field before writing ──────────────────────────
    stories = [
        {k: v for k, v in c.items() if not k.startswith("_")}
        for c in cases
    ]

    # Write JSON ──────────────────────────────────────────────────────────────
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _OUT_DIR / "demo_cases.json"

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_stats": {
            "n_runs":          len(runs),
            "timeseries_rows": len(ts),
            "lab_results":     len(lab),
            "events":          len(evts),
        },
        "n_cases":  len(stories),
        "warnings": all_warnings,
        "stories":  stories,
    }

    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)

    print(f"\nWritten: {out_path}")
    print(f"  {len(stories)} stories, {len(all_warnings)} warning(s)")


if __name__ == "__main__":
    main()
