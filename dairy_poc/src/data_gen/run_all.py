"""
CLI entry point — generate all four raw CSVs for the dairy PoC.

Usage (from dairy_poc/ directory):
    python -m src.data_gen.run_all --n_runs 200 --seed 42 --freq_min 1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Support both `python -m src.data_gen.run_all` and direct script execution
try:
    from .process_data import generate_run, _scenario_weights
    from .lab_data import generate_lab_result
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent))
    from process_data import generate_run, _scenario_weights   # type: ignore
    from lab_data import generate_lab_result                   # type: ignore

_OUT_DIR      = Path(__file__).parents[2] / "data_raw"
_BASE_DATE    = pd.Timestamp("2024-01-15 06:00:00")
_PRODUCTS     = ["QUARK", "HIGH_PROTEIN_PUDDING"]
_SCALES       = ["RD", "TECHNIKUM", "PRODUCTION"]
_SCALE_PROBS  = [0.20, 0.35, 0.45]


def _draw_scenario(product: str, rng: np.random.Generator) -> str:
    names, weights = _scenario_weights(product)
    return str(rng.choice(names, p=weights))


def generate_dataset(
    n_runs:   int   = 200,
    seed:     int   = 42,
    freq_min: float = 1.0,
) -> dict[str, pd.DataFrame]:
    """
    Generate all four tables and return them as a dict of DataFrames.
    Writes nothing to disk; the caller is responsible for saving.
    """
    rng = np.random.default_rng(seed)
    _OUT_DIR.mkdir(exist_ok=True)

    runs_rows:  list[dict]          = []
    ts_chunks:  list[pd.DataFrame]  = []
    lab_rows:   list[dict]          = []
    event_rows: list[dict]          = []

    # Stagger run start times per product to keep timestamps plausible
    quark_start   = _BASE_DATE
    pudding_start = _BASE_DATE
    evt_counter   = 1

    for i in range(n_runs):
        run_id   = f"RUN-{i + 1:04d}"
        product  = str(rng.choice(_PRODUCTS))          # 50 / 50 split
        scale    = str(rng.choice(_SCALES, p=_SCALE_PROBS))
        scenario = _draw_scenario(product, rng)

        if product == "QUARK":
            run_start     = quark_start
            quark_start  += pd.Timedelta(hours=16)    # space runs ~16 h apart
        else:
            run_start     = pudding_start
            pudding_start += pd.Timedelta(hours=5)

        run_meta, ts_df, events = generate_run(
            product=product, scenario=scenario, run_id=run_id,
            scale=scale, run_start=run_start, freq_min=freq_min, rng=rng,
        )
        lab_result = generate_lab_result(run_meta, rng)

        runs_rows.append(run_meta)
        ts_chunks.append(ts_df)
        lab_rows.append(lab_result)

        # Materialise events with absolute timestamps and unique IDs
        for evt in events:
            ts_evt = run_start + pd.Timedelta(minutes=float(evt["t_min"]))
            event_rows.append({
                "run_id":       run_id,
                "event_id":     f"EVT-{evt_counter:05d}",
                "timestamp":    ts_evt.isoformat(),
                "t_min":        round(float(evt["t_min"]), 3),
                "step":         evt["step"],
                "event_type":   evt["event_type"],
                "product":      product,
                "triggered_by": evt.get("triggered_by", "scheduled"),
                "duration_min": evt.get("duration_min", None),
                "notes":        None,
            })
            evt_counter += 1

    return {
        "runs":        pd.DataFrame(runs_rows),
        "timeseries":  pd.concat(ts_chunks, ignore_index=True),
        "lab_results": pd.DataFrame(lab_rows),
        "events":      pd.DataFrame(event_rows),
    }


def _print_step_table(run_id: str, ts_df: pd.DataFrame, ev_df: pd.DataFrame) -> None:
    """Print a compact step timeline for one run."""
    sub = ts_df[ts_df["run_id"] == run_id].copy()
    # Preserve step order by first appearance
    step_order = sub.groupby("step")["t_min"].min().sort_values().index.tolist()
    agg = (
        sub.groupby("step")["t_min"]
           .agg(start="min", end="max", rows="count")
           .loc[step_order]
    )
    agg["dur_min"] = (agg["end"] - agg["start"]).round(1)
    print(agg[["start", "end", "dur_min", "rows"]].to_string())

    evts = ev_df[ev_df["run_id"] == run_id][["event_type", "t_min", "step"]]
    if not evts.empty:
        print("  key events:")
        print(evts.to_string(index=False, col_space=28))


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate dairy PoC synthetic dataset")
    ap.add_argument("--n_runs",   type=int,   default=200)
    ap.add_argument("--seed",     type=int,   default=42)
    ap.add_argument("--freq_min", type=float, default=1.0)
    args = ap.parse_args()

    print(f"Generating {args.n_runs} runs  (seed={args.seed}, freq={args.freq_min} min) …\n")
    dfs = generate_dataset(args.n_runs, args.seed, args.freq_min)

    # Write CSVs
    for name, df in dfs.items():
        path = _OUT_DIR / f"{name}.csv"
        df.to_csv(path, index=False)

    # -- Row counts ------------------------------------------------------------
    print("-- Row counts --------------------------------------------------")
    for name, df in dfs.items():
        print(f"  {name + ':':<18} {len(df):>8,} rows  x  {len(df.columns)} cols")

    # -- Head preview (3 rows each) -------------------------------------------
    pd.set_option("display.max_columns", 8)
    pd.set_option("display.width", 110)
    for name, df in dfs.items():
        print(f"\n-- {name}.csv  head(3) --")
        print(df.head(3).to_string(max_colwidth=22))

    runs_df = dfs["runs"]
    ts_df   = dfs["timeseries"]
    ev_df   = dfs["events"]

    # -- Example Quark run timeline -------------------------------------------
    quark_runs = runs_df[runs_df["product"] == "QUARK"]
    if not quark_runs.empty:
        q_row = quark_runs.iloc[0]
        print(f"\n-- Quark example run: {q_row['run_id']}"
              f"  scale={q_row['scale']}  scenario={q_row['scenario']}"
              f"  gel_break={q_row['gel_break_time_min']} min"
              f"  sep_setpoint={q_row['separator_speed_setpoint_rpm']} rpm --")
        _print_step_table(q_row["run_id"], ts_df, ev_df)

    # -- Example Pudding run timeline -----------------------------------------
    pud_runs = runs_df[runs_df["product"] == "HIGH_PROTEIN_PUDDING"]
    if not pud_runs.empty:
        p_row = pud_runs.iloc[0]
        print(f"\n-- Pudding example run: {p_row['run_id']}"
              f"  scale={p_row['scale']}  scenario={p_row['scenario']} --")
        _print_step_table(p_row["run_id"], ts_df, ev_df)

    # -- Scenario distribution -------------------------------------------------
    print("\n-- Scenario distribution ----------------------------------------")
    print(
        runs_df.groupby(["product", "scenario"])
               .size()
               .rename("count")
               .to_string()
    )
    print("\nDone -- files written to data_raw/")


if __name__ == "__main__":
    main()
