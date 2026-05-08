"""
Run-level lab / QC result generator.

One row per run, driven by run_meta (product, scenario, gel_break_time_min,
raw_protein_pct, etc.).  Consistent with the timeseries scenario effects.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import yaml
from pathlib import Path

_CFG: dict = yaml.safe_load(
    (Path(__file__).parents[2] / "configs" / "scenarios.yaml").read_text(encoding="utf-8")
)


# ── Quark lab result ──────────────────────────────────────────────────────────

def _quark_lab(run_meta: dict, rng: np.random.Generator) -> dict:
    scenario       = run_meta["scenario"]
    raw_prot       = float(run_meta.get("raw_protein_pct") or 3.2)
    gel_break_time = float(run_meta.get("gel_break_time_min") or 10.0)
    gel_excess     = max(0.0, (gel_break_time - 5.0) / 5.0)

    # ── Base values (NORMAL / no-fault) ──────────────────────────────────────
    protein_pct      = rng.normal(11.8, 0.40)
    total_solids_pct = rng.normal(20.5, 0.50)
    viscosity_value  = rng.normal(9000, 600.0)
    d50_um           = rng.normal(28.0, 3.5)   # curd particle D50 after centrifugation

    # ── Fermentation outcome ──────────────────────────────────────────────────
    if scenario == "STALL_FERMENT":
        final_pH_offline = rng.normal(5.5, 0.05)
        ferm_hr          = 99.0                 # sentinel: run did not complete
    elif scenario == "OVER_ACID":
        final_pH_offline = float(rng.uniform(4.10, 4.35))
        ferm_hr          = rng.normal(11.5, 0.5)
    elif scenario == "SLOW_FERMENT":
        final_pH_offline = rng.normal(4.63, 0.06)
        ferm_hr          = rng.normal(15.5, 0.8)
    else:
        # Higher raw_protein_pct → marginally faster fermentation
        rate_adj         = 1.0 + 0.15 * (raw_prot - 3.2)
        final_pH_offline = rng.normal(4.52, 0.04)
        ferm_hr          = rng.normal(13.0 / rate_adj, 0.5)

    # ── Scenario-specific adjustments ────────────────────────────────────────

    if scenario == "SEPARATION_ISSUE":
        # Under-concentrated; gel-break spread compounds the protein miss
        protein_pct      = rng.normal(9.5, 0.5 + 0.08 * gel_excess)
        total_solids_pct -= rng.uniform(1.0, 3.0)
        d50_um            = rng.normal(44.0, 6.0)   # larger particles: whey not removed

    elif scenario == "STANDARDIZATION_OFFTARGET":
        # Wide spread on protein, mean stays close to target
        protein_pct = rng.normal(11.5, 1.2 + 0.08 * gel_excess)

    elif scenario == "OVER_ACID":
        viscosity_value = rng.normal(6500, 800.0)   # over-acid curd is softer
        d50_um          = rng.normal(18.0, 2.5)     # finer particles from over-syneresis

    elif scenario == "STALL_FERMENT":
        # Scrapped run: gel did not form; composition is unreliable
        protein_pct      = rng.normal(6.5, 1.2)
        total_solids_pct = rng.normal(12.0, 1.5)
        viscosity_value  = rng.normal(1500, 400.0)
        d50_um           = rng.normal(52.0, 9.0)    # large, undeveloped curd fragments

    elif scenario == "SLOW_FERMENT":
        d50_um = rng.normal(34.0, 4.0)   # under-developed gel → larger particles

    elif scenario == "RAW_MAT_VAR":
        # Protein content tracks raw material; not a fault per se
        protein_pct += (raw_prot - 3.2) * 1.5
        d50_um      -= (raw_prot - 3.2) * 2.0   # higher protein → firmer gel → smaller D50

    # Longer gel break → additional spread on protein; mechanical breakdown reduces D50
    protein_pct += rng.normal(0.0, 0.08 * gel_excess)
    d50_um      -= rng.normal(0.0, 1.0 * gel_excess)

    return {
        "protein_pct":          round(float(protein_pct),          3),
        "total_solids_pct":     round(float(total_solids_pct),      3),
        "viscosity_value":      round(float(viscosity_value),       1),
        "final_pH_offline":     round(float(final_pH_offline),      3),
        "d50_um":               round(float(max(2.0, d50_um)),      2),
        "fermentation_time_hr": round(float(ferm_hr),               2),
    }


# ── Pudding lab result ────────────────────────────────────────────────────────

def _pudding_lab(run_meta: dict, rng: np.random.Generator) -> dict:
    scenario = run_meta["scenario"]
    raw_prot = float(run_meta.get("raw_protein_pct") or 3.2)

    protein_pct      = rng.normal(12.5, 0.6)
    total_solids_pct = rng.normal(24.0, 0.8)
    viscosity_value  = rng.normal(4800, 400.0)
    final_pH_offline = rng.normal(6.65, 0.05)   # heat-set product; near-neutral
    d50_um           = rng.normal(14.0, 1.5)    # starch granules + protein aggregates

    if scenario == "FOUL":
        viscosity_value  = rng.normal(4200, 500.0)   # slight heat damage
        final_pH_offline = rng.normal(6.52, 0.06)    # Maillard → marginally lower pH
        d50_um           = rng.normal(18.5, 2.5)     # over-gelatinised starch → larger

    elif scenario == "BLOCK":
        protein_pct = rng.normal(11.8, 0.9)          # less controlled fill
        d50_um      = rng.normal(16.0, 3.5)          # thermal non-uniformity → wider D50

    elif scenario == "RAW_MAT_VAR":
        protein_pct     += (raw_prot - 3.2) * 1.2
        viscosity_value += (raw_prot - 3.2) * 80.0

    return {
        "protein_pct":          round(float(protein_pct),      3),
        "total_solids_pct":     round(float(total_solids_pct), 3),
        "viscosity_value":      round(float(viscosity_value),  1),
        "final_pH_offline":     round(float(final_pH_offline), 3),
        "d50_um":               round(float(max(2.0, d50_um)), 2),
        "fermentation_time_hr": None,
    }


# ── Spec-based QC flag ────────────────────────────────────────────────────────

def _spec_flag(row: dict, product: str) -> str:
    spec    = _CFG["products"][product]["spec_limits"]
    fails   = 0
    warns   = 0
    _CHECKS = ["protein_pct", "total_solids_pct", "viscosity_value",
               "final_pH_offline", "fermentation_time_hr", "d50_um"]

    for col in _CHECKS:
        if col not in spec:
            continue
        val = row.get(col)
        if val is None:
            continue
        if val == 99.0:        # STALL_FERMENT sentinel
            fails += 1
            continue
        lo, hi = spec[col]["lo"], spec[col]["hi"]
        span   = hi - lo
        if val < lo - 0.10 * span or val > hi + 0.10 * span:
            fails += 1
        elif val < lo or val > hi:
            warns += 1

    if fails:
        return "FAIL"
    if warns:
        return "WARN"
    return "PASS"


# ── Public entry point ────────────────────────────────────────────────────────

def generate_lab_result(run_meta: dict, rng: np.random.Generator) -> dict:
    """Return a single lab-result dict (one row of lab_results.csv)."""
    product  = run_meta["product"]
    run_id   = run_meta["run_id"]
    run_end  = pd.Timestamp(run_meta["run_end_ts"])

    fields = _quark_lab(run_meta, rng) if product == "QUARK" else _pudding_lab(run_meta, rng)

    result = {
        "run_id":      run_id,
        "product":     product,
        "sample_time": run_end.isoformat(),
        **fields,
    }
    result["result_flag"] = _spec_flag(result, product)
    return result
