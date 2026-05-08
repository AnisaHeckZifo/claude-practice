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
    fat_pct          = rng.normal(0.25, 0.05)
    viscosity_cP     = rng.normal(9000, 600.0)
    texture_score    = rng.normal(74.0, 5.0)
    microbial_cfu    = max(0.0, rng.normal(18.0, 7.0))
    whey_prot_loss   = max(0.0, rng.normal(0.07, 0.02))

    # ── Fermentation outcome ──────────────────────────────────────────────────
    if scenario == "STALL_FERMENT":
        final_pH = rng.normal(5.5, 0.05)
        ferm_hr  = 99.0                         # sentinel: run did not complete
    elif scenario == "OVER_ACID":
        final_pH = float(rng.uniform(4.10, 4.35))
        ferm_hr  = rng.normal(11.5, 0.5)
    elif scenario == "SLOW_FERMENT":
        final_pH = rng.normal(4.63, 0.06)
        ferm_hr  = rng.normal(15.5, 0.8)
    else:
        # Higher raw_protein_pct → marginally faster fermentation
        rate_adj = 1.0 + 0.15 * (raw_prot - 3.2)
        final_pH = rng.normal(4.52, 0.04)
        ferm_hr  = rng.normal(13.0 / rate_adj, 0.5)

    # ── Scenario-specific adjustments ────────────────────────────────────────

    if scenario == "SEPARATION_ISSUE":
        # Under-concentrated; gel-break spread compounds the protein miss
        protein_pct    = rng.normal(9.5, 0.5 + 0.08 * gel_excess)
        whey_prot_loss = float(rng.uniform(0.15, 0.40))
        total_solids_pct -= rng.uniform(1.0, 3.0)

    elif scenario == "STANDARDIZATION_OFFTARGET":
        # Wide spread on protein, mean stays close to target
        protein_pct = rng.normal(11.5, 1.2 + 0.08 * gel_excess)

    elif scenario == "OVER_ACID":
        viscosity_cP  = rng.normal(6500, 800.0)  # over-acid curd is softer
        texture_score = rng.normal(44.0, 7.0)

    elif scenario == "STALL_FERMENT":
        # Scrapped run: composition is unreliable
        protein_pct      = rng.normal(6.5, 1.2)  # not concentrated
        total_solids_pct = rng.normal(12.0, 1.5)
        viscosity_cP     = rng.normal(1500, 400.0)
        texture_score    = rng.normal(20.0, 8.0)

    elif scenario == "RAW_MAT_VAR":
        # Protein content tracks raw material; not a fault per se
        protein_pct += (raw_prot - 3.2) * 1.5

    # Gel break adds spread to protein_pct via standardization variance
    protein_pct += rng.normal(0.0, 0.08 * gel_excess)

    return {
        "protein_pct":            round(float(protein_pct),      3),
        "fat_pct":                round(float(fat_pct),          4),
        "total_solids_pct":       round(float(total_solids_pct), 3),
        "viscosity_cP":           round(float(viscosity_cP),     1),
        "texture_score":          round(float(np.clip(texture_score, 0.0, 100.0)), 2),
        "microbial_count_cfu":    round(float(microbial_cfu),    2),
        "final_pH":               round(float(final_pH),         3),
        "fermentation_time_hr":   round(float(ferm_hr),          2),
        "whey_protein_loss_proxy":round(float(whey_prot_loss),   4),
        "fouling_index_end":      None,
    }


# ── Pudding lab result ────────────────────────────────────────────────────────

def _pudding_lab(run_meta: dict, rng: np.random.Generator) -> dict:
    scenario = run_meta["scenario"]
    raw_prot = float(run_meta.get("raw_protein_pct") or 3.2)

    protein_pct      = rng.normal(12.5, 0.6)
    total_solids_pct = rng.normal(24.0, 0.8)
    fat_pct          = rng.normal(2.5,  0.2)
    viscosity_cP     = rng.normal(4800, 400.0)
    texture_score    = rng.normal(69.0, 5.0)
    microbial_cfu    = max(0.0, rng.normal(15.0, 6.0))
    fouling_idx_end  = float(rng.uniform(0.0, 0.05))   # near-zero for clean runs

    if scenario == "FOUL":
        viscosity_cP    = rng.normal(4200, 500.0)     # slight heat damage
        fouling_idx_end = float(rng.uniform(0.25, 0.55))

    elif scenario == "BLOCK":
        protein_pct     = rng.normal(11.8, 0.9)       # less controlled fill
        fouling_idx_end = float(rng.uniform(0.05, 0.20))

    elif scenario == "RAW_MAT_VAR":
        protein_pct  += (raw_prot - 3.2) * 1.2
        viscosity_cP += (raw_prot - 3.2) * 80.0

    return {
        "protein_pct":            round(float(protein_pct),      3),
        "fat_pct":                round(float(fat_pct),          4),
        "total_solids_pct":       round(float(total_solids_pct), 3),
        "viscosity_cP":           round(float(viscosity_cP),     1),
        "texture_score":          round(float(np.clip(texture_score, 0.0, 100.0)), 2),
        "microbial_count_cfu":    round(float(microbial_cfu),    2),
        "final_pH":               None,
        "fermentation_time_hr":   None,
        "whey_protein_loss_proxy":None,
        "fouling_index_end":      round(float(fouling_idx_end),  4),
    }


# ── Spec-based QC flag ────────────────────────────────────────────────────────

def _spec_flag(row: dict, product: str) -> str:
    spec    = _CFG["products"][product]["spec_limits"]
    fails   = 0
    warns   = 0
    _CHECKS = ["protein_pct", "total_solids_pct", "viscosity_cP",
               "final_pH", "fermentation_time_hr"]

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
