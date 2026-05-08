"""
YAML-driven process time-series generator for Quark and High-Protein Pudding.
Reads configs/scenarios.yaml as the source of truth for step durations,
sensor ranges, and scenario injection parameters.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import yaml
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
_CFG_PATH = Path(__file__).parents[2] / "configs" / "scenarios.yaml"
_CFG: dict = yaml.safe_load(_CFG_PATH.read_text(encoding="utf-8"))


# ── Low-level helpers ─────────────────────────────────────────────────────────

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def _step_order(product: str) -> list[str]:
    """Return step names sorted by their YAML order field."""
    steps = _CFG["products"][product]["process_steps"]
    return sorted(steps.keys(), key=lambda s: steps[s]["order"])


def _draw_step_durations(product: str, rng: np.random.Generator) -> dict[str, float]:
    """Sample a duration (minutes) for every step in the product."""
    result: dict[str, float] = {}
    for name, cfg in _CFG["products"][product]["process_steps"].items():
        if cfg.get("optional") and rng.random() > 0.80:
            continue  # Pudding filling_packaging skipped ~20 % of runs
        lo, hi = cfg["duration_min"]
        result[name] = float(rng.uniform(lo, hi))
    return result


def _noise(nominal: float | None, sd: float | None,
           n: int, rng: np.random.Generator,
           mult: float = 1.0) -> np.ndarray:
    """Return a sensor array: nominal + Gaussian noise scaled by mult."""
    if nominal is None or sd is None:
        return np.full(n, np.nan)
    return nominal + rng.normal(0.0, sd * mult, n)


# ── Milk tank blending ────────────────────────────────────────────────────────

_TANK_POOL = [f"T{i:02d}" for i in range(1, 31)]


def draw_blend(rng: np.random.Generator) -> dict:
    """Draw milk-tank blend metadata for one run."""
    n_tanks = int(rng.choice([1, 2], p=[0.30, 0.70]))
    tanks   = list(rng.choice(_TANK_POOL, size=n_tanks, replace=False))

    if n_tanks == 1:
        ids, ratio     = tanks[0], "1.0"
        prot_offset    = rng.normal(0.0, 0.12)
    else:
        r              = round(float(rng.uniform(0.30, 0.70)), 2)
        ids            = f"{tanks[0]}+{tanks[1]}"
        ratio          = f"{r}/{round(1.0 - r, 2)}"
        prot_offset    = rng.normal(0.0, 0.18)   # blending adds extra variation

    return {
        "milk_tank_ids":   ids,
        "blend_ratio":     ratio,
        "raw_protein_pct": round(float(np.clip(3.2 + prot_offset, 2.8, 3.8)), 3),
        "raw_solids_pct":  round(float(np.clip(
            12.2 + prot_offset * 0.5 + rng.normal(0.0, 0.25), 11.0, 13.5)), 3),
    }


# ── Quark fermentation pH trajectory ─────────────────────────────────────────

def _quark_ph_curve(
    n_early: int, n_late: int,
    scenario: str, raw_protein_pct: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build pH arrays for early_fermentation and late_fermentation.

    Early: mild linear drop 6.6 → 6.2 (rennet coagulation, minimal acid yet).
    Late:  sigmoid drop from 6.2 to target pH driven by lactic-acid bacteria.

    raw_protein_pct modulates rate slightly: higher protein → marginally faster
    acidification, consistent with a richer substrate for the culture.
    """
    ph_early = np.linspace(6.6, 6.2, max(1, n_early)) + rng.normal(0.0, 0.015, n_early)

    # Protein-driven rate adjustment (±15 % for ±1 pp from 3.2 baseline)
    prot_adj = 1.0 + 0.15 * (raw_protein_pct - 3.2)

    stall_onset = (
        _CFG.get("scenarios_quark", {})
            .get("STALL_FERMENT", {})
            .get("stall_onset_pct", 0.35)
    )

    if scenario == "SLOW_FERMENT":
        rate, target_ph = 0.55 * prot_adj, float(rng.uniform(4.55, 4.72))
    elif scenario == "STALL_FERMENT":
        rate, target_ph = 1.0, 4.45          # handled below
    elif scenario == "OVER_ACID":
        rate, target_ph = 1.35 * prot_adj, float(rng.uniform(4.10, 4.32))
    else:
        rate, target_ph = 1.0 * prot_adj, float(rng.uniform(4.42, 4.56))

    noise_l = rng.normal(0.0, 0.022, n_late)

    if scenario == "STALL_FERMENT":
        stall_idx = int(n_late * stall_onset)
        drop = np.linspace(6.2, 5.5, max(1, stall_idx))
        flat = np.full(n_late - stall_idx, 5.5)
        ph_late = np.concatenate([drop, flat]) + noise_l
    else:
        t      = np.linspace(0.0, 1.0, n_late)
        k      = 10.0 * rate
        centre = float(np.clip(0.40 / rate, 0.20, 0.75))
        ph_late = 6.2 - (6.2 - target_ph) * _sigmoid(k * (t - centre))
        ph_late += noise_l
        ph_late  = np.clip(ph_late, target_ph - 0.12, 6.35)

    return ph_early, ph_late


# ── Centrifuge speed profile ──────────────────────────────────────────────────

def _centrifuge_profile(
    n: int,
    setpoint: float,
    ramp_cfg: list,           # [min_ramp_min, max_ramp_min] from YAML
    shutdown_mult: float,     # noise multiplier during deceleration
    freq_min: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Centrifuge bowl speed:
    1. Sigmoidal startup ramp from 0 → setpoint over drawn_ramp_min.
    2. Steady state with small Gaussian noise (instrument jitter).
    3. Shutdown transient: last ~half-ramp_min gets elevated noise (vibration).
    """
    ramp_dur  = float(rng.uniform(ramp_cfg[0], ramp_cfg[1]))
    ramp_n    = max(2, int(ramp_dur / freq_min))
    stop_n    = max(2, int(ramp_dur * 0.5 / freq_min))
    base_sd   = 30.0

    speed = np.full(n, setpoint, dtype=float)

    # Startup: sigmoid from 0 to setpoint
    t_ramp       = np.linspace(-5.5, 3.5, ramp_n)
    speed[:ramp_n] = setpoint * _sigmoid(t_ramp)

    # Steady-state noise
    speed += rng.normal(0.0, base_sd, n)

    # Shutdown transient: last stop_n rows
    if stop_n < n:
        extra = rng.normal(0.0, base_sd * (shutdown_mult - 1.0), stop_n)
        speed[-stop_n:] += extra

    return np.clip(speed, 0.0, setpoint * 1.08)


# ── Scenario probability table ────────────────────────────────────────────────

def _scenario_weights(product: str) -> tuple[list[str], list[float]]:
    """Return (scenario_names, normalised_probs) for a given product."""
    common_pool   = _CFG["scenarios_common"]
    specific_key  = "scenarios_quark" if product == "QUARK" else "scenarios_pudding"
    specific_pool = _CFG.get(specific_key, {})

    names, weights = [], []
    for pool in (common_pool, specific_pool):
        for name, cfg in pool.items():
            names.append(name)
            weights.append(float(cfg["probability"]))

    total = sum(weights)
    return names, [w / total for w in weights]


# ── Quark timeseries builder ──────────────────────────────────────────────────

def _quark_ts(
    scenario: str,
    step_durs: dict[str, float],
    noise_mult: float,
    rng: np.random.Generator,
    freq_min: float,
    sep_setpoint: float,
    gel_break_time: float,
    raw_protein_pct: float,
    run_start: pd.Timestamp,
) -> tuple[pd.DataFrame, list[dict]]:
    """Build Quark per-minute timeseries and event list for one run."""

    steps_cfg = _CFG["products"]["QUARK"]["process_steps"]
    rows: list[dict] = []
    events: list[dict] = []
    t_elapsed = 0.0

    # Pre-compute full fermentation pH curve
    n_early = max(1, int(step_durs.get("early_fermentation", 180) / freq_min))
    n_late  = max(1, int(step_durs.get("late_fermentation",  600) / freq_min))
    ph_early, ph_late = _quark_ph_curve(n_early, n_late, scenario, raw_protein_pct, rng)

    # Rennet event lands 5–20 % of the way into early_fermentation
    rennet_frac = float(rng.uniform(0.05, 0.20))

    # Gel break → extra variance on separation_deltaP
    # Per YAML: +12 % noise SD per 5-min unit above baseline of 5 min
    gel_excess       = max(0.0, (gel_break_time - 5.0) / 5.0)
    sep_dp_noise_mult = 1.0 + 0.12 * gel_excess

    total_run_dur = sum(step_durs.values())

    for step_name in _step_order("QUARK"):
        if step_name not in step_durs:
            continue

        dur = step_durs[step_name]
        n   = max(1, int(dur / freq_min))
        sc  = steps_cfg[step_name]["sensors"]

        def s(col: str) -> np.ndarray:
            """Sample sensor array from YAML config with noise scaling."""
            cfg = sc.get(col, {})
            return _noise(cfg.get("nominal"), cfg.get("noise_sd"), n, rng, noise_mult)

        # Initialise all columns; product-specific ones default to NaN
        temp  = s("temperature_C")
        pres  = s("pressure_bar")
        flow  = s("flow_rate_lpm")
        shear = s("shear_rpm")       # null config → NaN during separation
        ph    = s("pH")
        cond  = s("conductivity_mS")
        sep_spd = np.full(n, np.nan)
        sep_dp  = np.full(n, np.nan)
        fou     = np.zeros(n)
        anom    = np.zeros(n, dtype=int)

        # ── step-specific overrides ───────────────────────────────────────────

        if step_name == "early_fermentation":
            ph = ph_early[:n].copy()
            # Rennet addition: point event within this step
            r_idx = int(n * rennet_frac)
            events.append({
                "event_type":   "rennet_addition",
                "t_min":        t_elapsed + r_idx * freq_min,
                "step":         step_name,
                "triggered_by": "scheduled",
            })

        elif step_name == "late_fermentation":
            ph = ph_late[:n].copy()
            # Fermentation anomalies are flagged across the whole late step
            if scenario in ("SLOW_FERMENT", "STALL_FERMENT", "OVER_ACID"):
                anom[:] = 1

        elif step_name == "separation":
            # shear_rpm stays NaN — centrifuge uses separator_speed_rpm
            sp_cfg = sc["separator_speed_rpm"]
            sep_spd = _centrifuge_profile(
                n, sep_setpoint,
                sp_cfg["startup_ramp_min"],
                sp_cfg["shutdown_noise_sd_multiplier"],
                freq_min, rng,
            )
            # separation_deltaP: nominal noise inflated by gel_break_time
            dp_cfg = sc["separation_deltaP"]
            base_sd = (dp_cfg.get("noise_sd") or 0.03) * noise_mult * sep_dp_noise_mult
            sep_dp  = dp_cfg["nominal"] + rng.normal(0.0, base_sd, n)

            if scenario == "SEPARATION_ISSUE":
                # deltaP drifts up; flow drops proportionally across the step
                ramp    = np.linspace(0.0, 0.5, n)
                sep_dp += ramp
                flow   -= ramp * 25.0
                anom[:] = 1

        elif step_name == "standardization_mix":
            if scenario == "STANDARDIZATION_OFFTARGET":
                # Dosing pump variability: extra noise on flow and shear only
                flow  += rng.normal(0.0, (sc["flow_rate_lpm"].get("noise_sd") or 3.0) * 2.0, n)
                shear += rng.normal(0.0, (sc["shear_rpm"].get("noise_sd")      or 20.0) * 1.5, n)
                anom[:] = 1

        # ── cross-cutting scenario effects ────────────────────────────────────

        if scenario == "DRIFT":
            run_pct = t_elapsed / total_run_dur
            if run_pct > 0.20:
                f = min((run_pct - 0.20) / 0.80, 1.0)
                temp += f * 2.5 * (sc.get("temperature_C", {}).get("noise_sd") or 0.3) * noise_mult
                flow -= f * 2.5 * (sc.get("flow_rate_lpm", {}).get("noise_sd") or 2.0) * noise_mult

        if scenario == "SENSOR_FAULT" and n > 0:
            n_spk  = int(rng.uniform(0, 4))
            idxs   = rng.integers(0, n, size=n_spk)
            temp[idxs] += rng.choice([-14.0, 14.0], size=n_spk)
            if n_spk:
                anom[idxs] = 1

        if scenario == "RAW_MAT_VAR":
            # Protein-driven baseline temperature offset (mild; process-insight only)
            temp += (raw_protein_pct - 3.2) * 0.30

        # ── step-start point events ───────────────────────────────────────────
        _STEP_EVENTS = {
            "inoculation_mixing": "inoculation",
            "gel_break_mixing":   "gel_break_start",
            "separation":         "separation_start",
            "standardization_mix":"standardization_start",
            "filling_packaging":  "filling_start",
        }
        if step_name in _STEP_EVENTS:
            events.append({
                "event_type":   _STEP_EVENTS[step_name],
                "t_min":        t_elapsed,
                "step":         step_name,
                "triggered_by": "scheduled",
            })

        # Clip flow: physically impossible to be negative
        flow = np.clip(flow, 0.0, None)

        # ── append rows ───────────────────────────────────────────────────────
        for i in range(n):
            rows.append({
                "t_min":                 round(t_elapsed + i * freq_min, 3),
                "timestamp":             run_start + pd.Timedelta(minutes=t_elapsed + i * freq_min),
                "step":                  step_name,
                "temperature_C":         round(float(temp[i]),  3),
                "pressure_bar":          round(float(pres[i]),  4),
                "flow_rate_lpm":         round(float(flow[i]),  2),
                "shear_rpm":             round(float(shear[i]), 1) if not np.isnan(shear[i]) else None,
                "separator_speed_rpm":   round(float(sep_spd[i]), 0) if not np.isnan(sep_spd[i]) else None,
                "pH":                    round(float(ph[i]),    4),
                "conductivity_mS":       round(float(cond[i]),  3) if not np.isnan(cond[i]) else None,
                "separation_deltaP":     round(float(sep_dp[i]),4) if not np.isnan(sep_dp[i]) else None,
                "viscosity_proxy":       None,
                "deltaT_heat_exchanger": None,
                "fouling_index":         round(float(fou[i]),   4),
                "anomaly_flag":          int(anom[i]),
            })

        t_elapsed += dur

    return pd.DataFrame(rows), events


# ── Pudding timeseries builder ────────────────────────────────────────────────

def _pudding_ts(
    scenario: str,
    step_durs: dict[str, float],
    noise_mult: float,
    rng: np.random.Generator,
    freq_min: float,
    raw_protein_pct: float,
    run_start: pd.Timestamp,
) -> tuple[pd.DataFrame, list[dict]]:
    """Build Pudding per-minute timeseries and event list for one run."""

    steps_cfg = _CFG["products"]["HIGH_PROTEIN_PUDDING"]["process_steps"]
    rows: list[dict] = []
    events: list[dict] = []
    t_elapsed = 0.0

    # FOUL: ramp is tracked over the combined heating+holding window
    foul_steps    = {"heating", "holding"}
    foul_dur_total = sum(step_durs.get(s, 0.0) for s in foul_steps if s in step_durs)
    foul_elapsed   = 0.0

    # BLOCK: pick one step for the blockage event
    block_step    = str(rng.choice(["heating", "cooling"]))
    block_onset   = 0.45
    block_dur_min = float(rng.uniform(5.0, 25.0))

    total_run_dur = sum(step_durs.values())

    for step_name in _step_order("HIGH_PROTEIN_PUDDING"):
        if step_name not in step_durs:
            continue

        dur = step_durs[step_name]
        n   = max(1, int(dur / freq_min))
        sc  = steps_cfg[step_name]["sensors"]

        def s(col: str) -> np.ndarray:
            cfg = sc.get(col, {})
            return _noise(cfg.get("nominal"), cfg.get("noise_sd"), n, rng, noise_mult)

        temp  = s("temperature_C")
        pres  = s("pressure_bar")
        flow  = s("flow_rate_lpm")
        shear = s("shear_rpm")
        ph    = s("pH")
        visc  = s("viscosity_proxy")
        dT_hx = s("deltaT_heat_exchanger")
        fou   = np.zeros(n)
        anom  = np.zeros(n, dtype=int)

        # ── FOUL: progressive deposit build-up on heat exchanger ──────────────
        if scenario == "FOUL" and step_name in foul_steps and foul_dur_total > 0:
            onset_t = foul_dur_total * 0.25   # fouling visible after 25 % of HX time
            foul_t  = np.array([foul_elapsed + i * freq_min for i in range(n)])
            ramp    = np.clip((foul_t - onset_t) / max(foul_dur_total - onset_t, 1.0), 0.0, 1.0)

            temp  -= ramp * 4.0              # outlet temperature drops
            pres  += ramp * 0.8              # back-pressure rises
            flow  -= ramp * 15.0             # flow restricted
            if not np.all(np.isnan(dT_hx)):
                dT_hx += ramp * 8.0          # HX temperature differential grows
            fou = ramp.copy()
            anom[ramp > 0.05] = 1
            foul_elapsed += dur

        # ── BLOCK: sudden flow restriction ───────────────────────────────────
        if scenario == "BLOCK" and step_name == block_step:
            onset_idx = int(n * block_onset)
            block_n   = max(1, int(block_dur_min / freq_min))
            end_idx   = min(n, onset_idx + block_n)
            flow[onset_idx:end_idx] *= 0.45
            pres[onset_idx:end_idx] += 0.90
            anom[onset_idx:end_idx]  = 1

        # ── DRIFT ─────────────────────────────────────────────────────────────
        if scenario == "DRIFT":
            run_pct = t_elapsed / total_run_dur
            if run_pct > 0.20:
                f = min((run_pct - 0.20) / 0.80, 1.0)
                temp += f * 2.5 * (sc.get("temperature_C", {}).get("noise_sd") or 0.6) * noise_mult
                flow -= f * 2.5 * (sc.get("flow_rate_lpm", {}).get("noise_sd") or 4.0) * noise_mult

        # ── SENSOR_FAULT: spikes on temperature ───────────────────────────────
        if scenario == "SENSOR_FAULT" and n > 0:
            n_spk = int(rng.uniform(0, 4))
            idxs  = rng.integers(0, n, size=n_spk)
            temp[idxs] += rng.choice([-12.0, 12.0], size=n_spk)
            if n_spk:
                anom[idxs] = 1

        # ── RAW_MAT_VAR: protein-driven viscosity baseline ────────────────────
        if scenario == "RAW_MAT_VAR" and not np.all(np.isnan(visc)):
            visc += (raw_protein_pct - 3.2) * 80.0

        # ── filling_start event ───────────────────────────────────────────────
        if step_name == "filling_packaging":
            events.append({
                "event_type":   "filling_start",
                "t_min":        t_elapsed,
                "step":         step_name,
                "triggered_by": "scheduled",
            })

        # Clip flow: physically impossible to be negative
        flow = np.clip(flow, 0.0, None)

        for i in range(n):
            rows.append({
                "t_min":                 round(t_elapsed + i * freq_min, 3),
                "timestamp":             run_start + pd.Timedelta(minutes=t_elapsed + i * freq_min),
                "step":                  step_name,
                "temperature_C":         round(float(temp[i]),  3),
                "pressure_bar":          round(float(pres[i]),  4),
                "flow_rate_lpm":         round(float(flow[i]),  2),
                "shear_rpm":             round(float(shear[i]), 1) if not np.isnan(shear[i]) else None,
                "separator_speed_rpm":   None,
                "pH":                    round(float(ph[i]),    4),
                "conductivity_mS":       None,
                "separation_deltaP":     None,
                "viscosity_proxy":       round(float(visc[i]),  2) if not np.isnan(visc[i]) else None,
                "deltaT_heat_exchanger": round(float(dT_hx[i]), 3) if not np.isnan(dT_hx[i]) else None,
                "fouling_index":         round(float(fou[i]),   4),
                "anomaly_flag":          int(anom[i]),
            })

        t_elapsed += dur

    return pd.DataFrame(rows), events


# ── Top-level run generator ───────────────────────────────────────────────────

def generate_run(
    product:   str,
    scenario:  str,
    run_id:    str,
    scale:     str,
    run_start: pd.Timestamp,
    freq_min:  float,
    rng:       np.random.Generator,
) -> tuple[dict, pd.DataFrame, list[dict]]:
    """
    Generate one synthetic run.

    Returns
    -------
    run_meta : dict
        One-row payload for runs.csv.
    ts_df : pd.DataFrame
        Minute-cadence timeseries for timeseries.csv.
    events : list[dict]
        Point-event records for events.csv.

    Lab results are generated separately in lab_data.generate_lab_result().
    """
    prod_cfg   = _CFG["products"][product]
    scale_cfg  = _CFG["scales"][scale]
    noise_mult = float(scale_cfg["noise_multiplier"])

    step_durs  = _draw_step_durations(product, rng)
    total_min  = sum(step_durs.values())
    run_end    = run_start + pd.Timedelta(minutes=total_min)

    blend = draw_blend(rng)

    # Quark-only run-level fields
    sep_setpoint   = None
    gel_break_time = None
    if product == "QUARK":
        sep_setpoint   = float(rng.choice([5500, 6000, 6500, 7000, 7500]))
        gel_break_time = step_durs.get("gel_break_mixing", 10.0)

    # Outcome defaults — overridden per scenario below
    quality_dev  = 0
    yield_loss   = float(rng.uniform(0.0, 0.8))   # baseline tiny loss
    downtime_min = 0.0
    fouling_grade = 0
    extra_clean  = 0

    if scenario == "FOUL":
        fouling_grade = int(rng.choice([1, 2, 3], p=[0.30, 0.50, 0.20]))
        extra_clean   = 1
        downtime_min  = float(rng.uniform(20.0, 90.0))
        yield_loss    = float(rng.uniform(1.0,  5.0))
        quality_dev   = int(rng.choice([0, 1], p=[0.5, 0.5]))
    elif scenario == "BLOCK":
        downtime_min  = float(rng.uniform(15.0, 60.0))
        quality_dev   = 1
        yield_loss    = float(rng.uniform(2.0,  8.0))
        fouling_grade = int(rng.choice([0, 1], p=[0.60, 0.40]))
        extra_clean   = int(rng.choice([0, 1], p=[0.50, 0.50]))
    elif scenario == "SEPARATION_ISSUE":
        yield_loss    = float(rng.uniform(5.0, 15.0))
        quality_dev   = 1
        downtime_min  = float(rng.uniform(20.0, 60.0))
        fouling_grade = int(rng.choice([0, 1], p=[0.60, 0.40]))
    elif scenario == "STANDARDIZATION_OFFTARGET":
        yield_loss    = float(rng.uniform(0.0, 5.0))
        quality_dev   = int(rng.choice([0, 1], p=[0.50, 0.50]))
    elif scenario == "STALL_FERMENT":
        yield_loss    = float(rng.uniform(80.0, 100.0))
        quality_dev   = 1
        downtime_min  = float(rng.uniform(60.0, 240.0))
    elif scenario == "OVER_ACID":
        yield_loss    = float(rng.uniform(5.0, 20.0))
        quality_dev   = 1
    elif scenario == "SLOW_FERMENT":
        downtime_min  = float(rng.uniform(30.0, 120.0))
        quality_dev   = int(rng.choice([0, 1], p=[0.50, 0.50]))
    elif scenario == "RAW_MAT_VAR":
        quality_dev   = int(rng.choice([0, 1], p=[0.50, 0.50]))
        yield_loss    = float(rng.uniform(0.0, 4.0))

    # Gel-break-driven spread on yield_loss (Quark only)
    if product == "QUARK" and gel_break_time is not None:
        gel_excess  = max(0.0, (gel_break_time - 5.0) / 5.0)
        yield_loss += float(rng.normal(0.0, 0.5 * gel_excess))
        yield_loss  = max(0.0, yield_loss)

    batch_lo, batch_hi = scale_cfg["batch_size_L"]

    run_meta = {
        "run_id":                      run_id,
        "product":                     product,
        "scale":                       scale,
        "scenario":                    scenario,
        "run_date":                    run_start.date().isoformat(),
        "run_start_ts":                run_start.isoformat(),
        "run_end_ts":                  run_end.isoformat(),
        "total_duration_hr":           round(total_min / 60.0, 3),
        "batch_size_L":                round(float(rng.uniform(batch_lo, batch_hi)), 1),
        "separator_speed_setpoint_rpm":sep_setpoint,
        "gel_break_time_min":          round(gel_break_time, 1) if gel_break_time is not None else None,
        **blend,
        "quality_deviation":           quality_dev,
        "yield_loss_pct":              round(yield_loss, 3),
        "downtime_minutes":            round(downtime_min, 1),
        "fouling_grade":               fouling_grade,
        "extra_cleaning":              extra_clean,
    }

    if product == "QUARK":
        ts_df, events = _quark_ts(
            scenario, step_durs, noise_mult, rng, freq_min,
            sep_setpoint, gel_break_time,
            float(blend["raw_protein_pct"]), run_start,
        )
    else:
        ts_df, events = _pudding_ts(
            scenario, step_durs, noise_mult, rng, freq_min,
            float(blend["raw_protein_pct"]), run_start,
        )

    ts_df.insert(0, "run_id",   run_id)
    ts_df.insert(1, "product",  product)

    return run_meta, ts_df, events
