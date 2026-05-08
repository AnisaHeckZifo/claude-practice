"""
Sanity-checker and validation-plot generator for the dairy PoC synthetic dataset.

Run from dairy_poc/ directory:
    .venv/Scripts/python.exe -m src.data_gen.validate

Reads:  data_raw/{runs,timeseries,lab_results,events}.csv
Writes: data_raw/validation_plots/*.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive; must be set before pyplot import
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Patch

_ROOT     = Path(__file__).parents[2]
_DATA_DIR = _ROOT / "data_raw"
_PLOT_DIR = _DATA_DIR / "validation_plots"

# ── Step color palette ─────────────────────────────────────────────────────────
_STEP_COLORS: dict[str, str] = {
    "inoculation_mixing":   "#ddeeff",
    "early_fermentation":   "#b8d4f8",
    "late_fermentation":    "#88bbf0",
    "gel_break_mixing":     "#ffe8cc",
    "separation":           "#ffd4aa",
    "standardization_mix":  "#ffcc88",
    "cooling":              "#ccf0dd",
    "filling_packaging":    "#e8f5cc",
    "mixing":               "#fff0d8",
    "heating":              "#ffd5b0",
    "holding":              "#ffb898",
}

# ── Check registry ─────────────────────────────────────────────────────────────
_checks: list[tuple[str, str, str]] = []   # (status, name, detail)


def _chk(name: str, passed: bool, detail: str = "") -> None:
    _checks.append(("PASS" if passed else "FAIL", name, detail))


# =============================================================================
# DATA LOADING
# =============================================================================

def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    runs  = pd.read_csv(_DATA_DIR / "runs.csv",        parse_dates=["run_start_ts", "run_end_ts"])
    ts    = pd.read_csv(_DATA_DIR / "timeseries.csv",  parse_dates=["timestamp"])
    lab   = pd.read_csv(_DATA_DIR / "lab_results.csv", parse_dates=["sample_time"])
    evts  = pd.read_csv(_DATA_DIR / "events.csv",      parse_dates=["timestamp"])
    return runs, ts, lab, evts


# =============================================================================
# PLOT HELPERS
# =============================================================================

def _step_spans(ts_run: pd.DataFrame) -> list[tuple[str, float, float]]:
    """Return [(step_name, t_start, t_end)] sorted by first appearance."""
    grp   = ts_run.groupby("step")["t_min"].agg(start="min", end="max")
    order = ts_run.groupby("step")["t_min"].min().sort_values().index.tolist()
    return [(s, float(grp.loc[s, "start"]), float(grp.loc[s, "end"]))
            for s in order if s in grp.index]


def _shade_steps(ax: plt.Axes, ts_run: pd.DataFrame,
                 alpha: float = 0.20) -> list[Patch]:
    """Shade each step in the axis background; return legend patches."""
    patches: list[Patch] = []
    seen: set[str] = set()
    for step, t0, t1 in _step_spans(ts_run):
        color = _STEP_COLORS.get(step, "#eeeeee")
        ax.axvspan(t0, t1, alpha=alpha, color=color, zorder=0)
        if step not in seen:
            patches.append(Patch(facecolor=color, alpha=0.7, label=step))
            seen.add(step)
    return patches


def _mark_events(ax: plt.Axes, ev_run: pd.DataFrame) -> None:
    """Draw vertical dashed lines with short rotated labels for each event."""
    for _, row in ev_run.sort_values("t_min").iterrows():
        ax.axvline(row["t_min"], color="#333333", linestyle="--",
                   linewidth=0.55, alpha=0.50, zorder=2)
        ax.text(
            row["t_min"], 0.97,
            str(row["event_type"])[:12],
            rotation=90, fontsize=5, va="top", ha="right", alpha=0.65,
            transform=ax.get_xaxis_transform(), zorder=3,
        )


def _pick_run(runs: pd.DataFrame, product: str, scenario: str) -> str | None:
    """Return first run_id matching product + scenario, or None."""
    sub = runs[(runs["product"] == product) & (runs["scenario"] == scenario)]
    return str(sub["run_id"].iloc[0]) if not sub.empty else None


# =============================================================================
# A  SANITY CHECKS
# =============================================================================

def check_run_id_consistency(
    runs: pd.DataFrame,
    ts: pd.DataFrame,
    lab: pd.DataFrame,
    evts: pd.DataFrame,
) -> None:
    """Every run_id in runs.csv must appear in timeseries and lab_results."""
    run_ids = set(runs["run_id"])
    missing_ts  = run_ids - set(ts["run_id"])
    missing_lab = run_ids - set(lab["run_id"])
    _chk("All run_ids present in timeseries",
         len(missing_ts) == 0,
         f"{len(missing_ts)} missing" if missing_ts else "")
    _chk("All run_ids present in lab_results",
         len(missing_lab) == 0,
         f"{len(missing_lab)} missing" if missing_lab else "")


def check_value_ranges(ts: pd.DataFrame) -> None:
    """Basic plausibility gates on sensor columns."""
    ph     = ts["pH"].dropna()
    # Exclude anomaly_flag=1 rows from temperature check: SENSOR_FAULT spikes are
    # intentional out-of-range readings and should not be counted as generator bugs.
    temp   = ts.loc[ts["anomaly_flag"] == 0, "temperature_C"].dropna()
    pres   = ts["pressure_bar"].dropna()
    flow   = ts["flow_rate_lpm"].dropna()

    bad_ph   = int(((ph   < 3.5)  | (ph   > 7.5)).sum())
    bad_temp = int(((temp < -5.0) | (temp > 120.0)).sum())
    neg_pres = int((pres  < 0.0).sum())
    neg_flow = int((flow  < 0.0).sum())

    _chk("pH in plausible range [3.5, 7.5]",                 bad_ph == 0,
         f"{bad_ph} out-of-range values")
    _chk("temperature_C in range [-5, 120] C (non-anomaly)", bad_temp == 0,
         f"{bad_temp} out-of-range values")
    _chk("pressure_bar >= 0",                                neg_pres == 0,
         f"{neg_pres} negative values")
    _chk("flow_rate_lpm >= 0",                               neg_flow == 0,
         f"{neg_flow} negative values")


def check_ph_trajectories(runs: pd.DataFrame, ts: pd.DataFrame) -> None:
    """A1: Quark fermentation pH should decrease for NORMAL/SLOW/OVER_ACID;
    STALL_FERMENT should plateau in the second half of late_fermentation."""
    q_runs = runs[runs["product"] == "QUARK"]

    # Decreasing scenarios: Pearson r(t_min, pH) in late_fermentation < -0.70
    for scenario in ("NORMAL", "SLOW_FERMENT", "OVER_ACID"):
        ids = q_runs.loc[q_runs["scenario"] == scenario, "run_id"]
        if ids.empty:
            _chk(f"pH decreasing in late_fermentation: {scenario}", True, "no runs -- skipped")
            continue

        corrs = []
        for rid in ids:
            late = ts[(ts["run_id"] == rid) & (ts["step"] == "late_fermentation")]
            if len(late) < 5:
                continue
            r = late["t_min"].corr(late["pH"])
            if pd.notna(r):
                corrs.append(r)

        if not corrs:
            _chk(f"pH decreasing: {scenario}", True, "no late_fermentation rows -- skipped")
            continue

        mean_r = float(np.mean(corrs))
        _chk(
            f"pH decreasing in late_fermentation: {scenario} (r < -0.70)",
            mean_r < -0.70,
            f"mean Pearson r = {mean_r:.3f} across {len(corrs)} runs",
        )

    # STALL_FERMENT: second half of late_fermentation should be nearly flat (std < 0.12)
    stall_ids = q_runs.loc[q_runs["scenario"] == "STALL_FERMENT", "run_id"]
    if stall_ids.empty:
        _chk("pH plateau: STALL_FERMENT", True, "no runs -- skipped")
    else:
        stds = []
        for rid in stall_ids:
            late = ts[(ts["run_id"] == rid) & (ts["step"] == "late_fermentation")]
            if len(late) < 10:
                continue
            second_half = late.iloc[len(late) // 2 :]
            stds.append(float(second_half["pH"].std()))
        if stds:
            mean_std = float(np.mean(stds))
            _chk(
                "pH plateau in 2nd half of late_fermentation: STALL_FERMENT (std < 0.12)",
                mean_std < 0.12,
                f"mean pH std = {mean_std:.4f} across {len(stds)} runs",
            )


def check_step_ordering(runs: pd.DataFrame, ts: pd.DataFrame) -> None:
    """A2: Steps appear in the correct product-defined order."""
    _EXPECTED: dict[str, list[str]] = {
        "QUARK": [
            "inoculation_mixing", "early_fermentation", "late_fermentation",
            "gel_break_mixing", "separation", "standardization_mix",
            "cooling", "filling_packaging",
        ],
        "HIGH_PROTEIN_PUDDING": [
            "mixing", "heating", "holding", "cooling", "filling_packaging",
        ],
    }
    wrong = 0
    total = len(runs)

    for _, row in runs.iterrows():
        expected = _EXPECTED.get(row["product"], [])
        run_ts   = ts[ts["run_id"] == row["run_id"]]
        actual   = (
            run_ts.groupby("step")["t_min"].min()
                  .sort_values()
                  .index.tolist()
        )
        # Allow optional steps to be absent; filter expected to what appeared
        filtered = [s for s in expected if s in actual]
        if actual != filtered:
            wrong += 1

    _chk(
        "Step order consistent with product definition",
        wrong == 0,
        f"{wrong}/{total} runs have unexpected step order",
    )


def check_step_durations(runs: pd.DataFrame, ts: pd.DataFrame) -> None:
    """A2b: Step durations should be within YAML [min, max] bounds (with 5 % buffer)."""
    # Hardcoded from YAML to keep validate.py self-contained
    _BOUNDS: dict[str, tuple[float, float]] = {
        "inoculation_mixing":  (10,  30),
        "early_fermentation":  (120, 240),
        "late_fermentation":   (480, 720),
        "gel_break_mixing":    (5,   25),
        "separation":          (30,  90),
        "standardization_mix": (15,  45),
        "cooling":             (30,  60),
        "filling_packaging":   (30,  90),
        "mixing":              (20,  40),
        "heating":             (20,  40),
        "holding":             (10,  30),
    }
    buf = 0.05
    violations = 0
    checked    = 0

    step_durs = (
        ts.groupby(["run_id", "step"])["t_min"]
          .agg(start="min", end="max")
          .assign(dur=lambda d: d["end"] - d["start"] + 1)  # +1: row count = span + 1 tick
          .reset_index()
    )
    for _, r in step_durs.iterrows():
        step = r["step"]
        if step not in _BOUNDS:
            continue
        lo, hi = _BOUNDS[step]
        dur    = r["dur"]
        checked += 1
        if dur < lo * (1 - buf) or dur > hi * (1 + buf):
            violations += 1

    _chk(
        "Step durations within YAML bounds (5 % buffer)",
        violations == 0,
        f"{violations}/{checked} step durations outside bounds",
    )


def check_scenario_outcomes(runs: pd.DataFrame) -> None:
    """A3: Outcome labels should correlate with scenario severity."""
    pud = runs[runs["product"] == "HIGH_PROTEIN_PUDDING"]
    qrk = runs[runs["product"] == "QUARK"]

    # FOUL should have higher fouling_grade, more extra_cleaning, more downtime
    foul   = pud[pud["scenario"] == "FOUL"]
    pnorm  = pud[pud["scenario"] == "NORMAL"]
    if len(foul) >= 2 and len(pnorm) >= 2:
        for col, label in [
            ("fouling_grade",   "fouling_grade mean"),
            ("yield_loss_pct",  "yield_loss_pct mean"),
            ("extra_cleaning",  "extra_cleaning rate"),
            ("downtime_minutes","downtime_minutes mean"),
        ]:
            foul_val  = foul[col].mean()
            norm_val  = pnorm[col].mean()
            _chk(
                f"Pudding FOUL > NORMAL: {col}",
                foul_val > norm_val,
                f"FOUL={foul_val:.2f}  NORMAL={norm_val:.2f}",
            )
    else:
        _chk("Pudding FOUL vs NORMAL outcomes", True,
             f"insufficient runs (FOUL={len(foul)}, NORMAL={len(pnorm)}) -- skipped")

    # SEPARATION_ISSUE: higher yield_loss vs NORMAL Quark
    sep_iss = qrk[qrk["scenario"] == "SEPARATION_ISSUE"]
    qnorm   = qrk[qrk["scenario"] == "NORMAL"]
    if len(sep_iss) >= 2 and len(qnorm) >= 2:
        si_yl = sep_iss["yield_loss_pct"].mean()
        qn_yl = qnorm["yield_loss_pct"].mean()
        _chk(
            "Quark SEPARATION_ISSUE yield_loss > NORMAL",
            si_yl > qn_yl,
            f"SEPARATION_ISSUE={si_yl:.2f}%  NORMAL={qn_yl:.2f}%",
        )
    else:
        _chk("Quark SEPARATION_ISSUE vs NORMAL", True,
             f"insufficient runs (SEP={len(sep_iss)}, NORMAL={len(qnorm)}) -- skipped")

    # STALL_FERMENT: highest yield_loss among Quark scenarios
    if len(qrk) > 0:
        stall = qrk[qrk["scenario"] == "STALL_FERMENT"]
        if not stall.empty and not qnorm.empty:
            stall_yl = stall["yield_loss_pct"].mean()
            _chk(
                "Quark STALL_FERMENT has highest yield_loss (> 50 %)",
                stall_yl > 50.0,
                f"STALL mean yield_loss = {stall_yl:.1f}%",
            )


def check_sensor_faults(runs: pd.DataFrame, ts: pd.DataFrame) -> None:
    """A4: SENSOR_FAULT runs should have a higher anomaly_flag rate than NORMAL."""
    sf_ids = runs.loc[runs["scenario"] == "SENSOR_FAULT", "run_id"]
    nm_ids = runs.loc[runs["scenario"] == "NORMAL",       "run_id"]

    if sf_ids.empty or nm_ids.empty:
        _chk("SENSOR_FAULT anomaly_flag rate > NORMAL", True, "no runs -- skipped")
        return

    sf_rate = float(ts[ts["run_id"].isin(sf_ids)]["anomaly_flag"].mean())
    nm_rate = float(ts[ts["run_id"].isin(nm_ids)]["anomaly_flag"].mean())
    _chk(
        "SENSOR_FAULT anomaly_flag rate > NORMAL",
        sf_rate > nm_rate,
        f"SENSOR_FAULT={sf_rate:.5f}  NORMAL={nm_rate:.5f}",
    )


def check_centrifuge_ramp(runs: pd.DataFrame, ts: pd.DataFrame) -> None:
    """Quark centrifuge: mean speed in first 20 % of separation < last 50 %."""
    q_ids   = runs.loc[runs["product"] == "QUARK", "run_id"]
    sep_ts  = ts[(ts["run_id"].isin(q_ids)) & (ts["step"] == "separation")]
    sep_ts  = sep_ts[sep_ts["separator_speed_rpm"].notna()]

    if sep_ts.empty:
        _chk("Centrifuge startup ramp (speed increases)", True,
             "no separation rows with separator_speed_rpm -- skipped")
        return

    ok = 0
    total = 0
    for rid, grp in sep_ts.groupby("run_id"):
        grp = grp.sort_values("t_min")
        n   = len(grp)
        if n < 10:
            continue
        first20 = float(grp.iloc[:max(1, n // 5)]["separator_speed_rpm"].mean())
        last50  = float(grp.iloc[n // 2 :]["separator_speed_rpm"].mean())
        if last50 > first20:
            ok += 1
        total += 1

    if total > 0:
        pct = ok / total
        _chk(
            "Centrifuge startup ramp (first-20% < last-50%, >= 80 % of runs)",
            pct >= 0.80,
            f"{ok}/{total} runs show expected ramp ({pct:.0%})",
        )


def check_pudding_foul_signals(runs: pd.DataFrame, ts: pd.DataFrame) -> None:
    """Pudding FOUL should show higher mean pressure and deltaT vs NORMAL during heating/holding."""
    foul_ids  = runs.loc[(runs["product"] == "HIGH_PROTEIN_PUDDING") &
                         (runs["scenario"] == "FOUL"),   "run_id"]
    norm_ids  = runs.loc[(runs["product"] == "HIGH_PROTEIN_PUDDING") &
                         (runs["scenario"] == "NORMAL"), "run_id"]

    if foul_ids.empty or norm_ids.empty:
        _chk("Pudding FOUL signal vs NORMAL (pressure/deltaT)", True,
             f"insufficient runs (FOUL={len(foul_ids)}, NORMAL={len(norm_ids)}) -- skipped")
        return

    heat_steps = {"heating", "holding"}
    foul_ts = ts[(ts["run_id"].isin(foul_ids))  & ts["step"].isin(heat_steps)]
    norm_ts = ts[(ts["run_id"].isin(norm_ids)) & ts["step"].isin(heat_steps)]

    if foul_ts.empty or norm_ts.empty:
        _chk("Pudding FOUL signal vs NORMAL", True, "no heating/holding rows -- skipped")
        return

    foul_pres = float(foul_ts["pressure_bar"].mean())
    norm_pres = float(norm_ts["pressure_bar"].mean())
    _chk(
        "Pudding FOUL mean pressure > NORMAL during heating/holding",
        foul_pres > norm_pres,
        f"FOUL={foul_pres:.4f} bar  NORMAL={norm_pres:.4f} bar",
    )

    dT_col = "deltaT_heat_exchanger"
    if foul_ts[dT_col].notna().any() and norm_ts[dT_col].notna().any():
        foul_dT = float(foul_ts[dT_col].mean())
        norm_dT = float(norm_ts[dT_col].mean())
        _chk(
            "Pudding FOUL mean deltaT_HX > NORMAL",
            foul_dT > norm_dT,
            f"FOUL={foul_dT:.3f} C  NORMAL={norm_dT:.3f} C",
        )


# =============================================================================
# B  PLOTS
# =============================================================================

def plot_quark_ph_curves(
    runs: pd.DataFrame, ts: pd.DataFrame, evts: pd.DataFrame
) -> Path | None:
    """B1: One panel each for NORMAL, OVER_ACID, STALL_FERMENT Quark pH curves."""
    scenarios = ["NORMAL", "OVER_ACID", "STALL_FERMENT"]
    run_ids   = [_pick_run(runs, "QUARK", s) for s in scenarios]

    if all(r is None for r in run_ids):
        return None

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    fig.suptitle("Quark fermentation pH curves (step-shaded, events marked)",
                 fontsize=12, fontweight="bold")

    ferm_steps = {
        "inoculation_mixing", "early_fermentation",
        "late_fermentation", "gel_break_mixing",
    }

    for ax, scenario, rid in zip(axes, scenarios, run_ids):
        ax.set_title(scenario, fontsize=10)
        ax.set_xlabel("Elapsed time (min)")
        ax.set_ylabel("pH")

        if rid is None:
            ax.text(0.5, 0.5, "No run available", ha="center", va="center",
                    transform=ax.transAxes, color="gray", fontsize=9)
            continue

        run_ts   = ts[ts["run_id"] == rid].sort_values("t_min")
        ferm_ts  = run_ts[run_ts["step"].isin(ferm_steps)]
        run_evts = evts[(evts["run_id"] == rid) &
                        evts["step"].isin(ferm_steps)]

        patches = _shade_steps(ax, ferm_ts)
        ax.plot(ferm_ts["t_min"], ferm_ts["pH"],
                color="#1a4f8a", linewidth=0.9, zorder=3)
        _mark_events(ax, run_evts)

        ax.set_ylim(3.8, 7.1)
        ax.yaxis.set_major_locator(mticker.MultipleLocator(0.5))
        ax.grid(axis="y", linestyle=":", alpha=0.45)

    # Step legend on last panel
    if run_ids[-1] is not None:
        last_ts = ts[ts["run_id"] == run_ids[-1]]
        last_ts = last_ts[last_ts["step"].isin(ferm_steps)]
        patches = _shade_steps(axes[-1], last_ts)
        seen: dict[str, Patch] = {}
        for p in patches:
            if p.get_label() not in seen:
                seen[p.get_label()] = p
        axes[-1].legend(handles=list(seen.values()), fontsize=6,
                        loc="upper right", framealpha=0.85)

    plt.tight_layout()
    out = _PLOT_DIR / "01_quark_ph_curves.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_pudding_fouling(
    runs: pd.DataFrame, ts: pd.DataFrame, evts: pd.DataFrame
) -> Path | None:
    """B2: NORMAL vs FOUL Pudding — pressure and deltaT_HX during process."""
    rid_norm = _pick_run(runs, "HIGH_PROTEIN_PUDDING", "NORMAL")
    rid_foul = _pick_run(runs, "HIGH_PROTEIN_PUDDING", "FOUL")

    if rid_norm is None and rid_foul is None:
        return None

    show_steps = {"mixing", "heating", "holding", "cooling"}
    fig, axes  = plt.subplots(2, 2, figsize=(14, 7))
    fig.suptitle("Pudding fouling signature: NORMAL vs FOUL",
                 fontsize=12, fontweight="bold")

    for col_idx, (rid, label) in enumerate([(rid_norm, "NORMAL"), (rid_foul, "FOUL")]):
        ax_pres, ax_dT = axes[0, col_idx], axes[1, col_idx]
        ax_pres.set_title(label, fontsize=10)

        if rid is None:
            for ax in (ax_pres, ax_dT):
                ax.text(0.5, 0.5, "No run available", ha="center", va="center",
                        transform=ax.transAxes, color="gray")
            continue

        run_ts  = ts[ts["run_id"] == rid].sort_values("t_min")
        sub_ts  = run_ts[run_ts["step"].isin(show_steps)]
        run_evts = evts[evts["run_id"] == rid]

        patches_pres = _shade_steps(ax_pres, sub_ts)
        _shade_steps(ax_dT, sub_ts)

        ax_pres.plot(sub_ts["t_min"], sub_ts["pressure_bar"],
                     color="#cc3300", linewidth=0.9, zorder=3)
        ax_pres.set_ylabel("pressure (bar)")
        ax_pres.grid(axis="y", linestyle=":", alpha=0.45)
        _mark_events(ax_pres, run_evts[run_evts["step"].isin(show_steps)])

        dT_vals = sub_ts["deltaT_heat_exchanger"]
        if dT_vals.notna().any():
            ax_dT.plot(sub_ts["t_min"], dT_vals,
                       color="#005599", linewidth=0.9, zorder=3)
            ax_dT.set_ylabel("deltaT heat exchanger (C)")
        else:
            ax_dT.text(0.5, 0.5, "deltaT not instrumented\n(mixing step only)",
                       ha="center", va="center", transform=ax_dT.transAxes,
                       color="gray", fontsize=8)

        ax_dT.set_xlabel("Elapsed time (min)")
        ax_dT.grid(axis="y", linestyle=":", alpha=0.45)

    plt.tight_layout()
    out = _PLOT_DIR / "02_pudding_fouling_signature.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_distributions(runs: pd.DataFrame, lab: pd.DataFrame) -> Path:
    """B3: Fouling grade by scenario, yield_loss boxplots, protein distributions, scatter."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Run-level outcome and lab distributions", fontsize=12, fontweight="bold")

    # top-left: fouling_grade stacked bar by scenario (Pudding only)
    ax = axes[0, 0]
    pud = runs[runs["product"] == "HIGH_PROTEIN_PUDDING"]
    fg  = pud.groupby(["scenario", "fouling_grade"]).size().unstack(fill_value=0)
    if not fg.empty:
        fg.plot(kind="bar", stacked=True, ax=ax,
                colormap="Blues", edgecolor="white", width=0.65)
        ax.set_title("Fouling grade by scenario (Pudding)")
        ax.set_xlabel("")
        ax.set_ylabel("Run count")
        ax.tick_params(axis="x", rotation=30, labelsize=8)
        ax.legend(title="Grade", fontsize=7, loc="upper right")
    ax.grid(axis="y", linestyle=":", alpha=0.45)

    # top-right: yield_loss_pct boxplot for top-8 scenarios by count
    ax = axes[0, 1]
    top8   = runs["scenario"].value_counts().head(8).index.tolist()
    sub    = runs[runs["scenario"].isin(top8)]
    order  = (sub.groupby("scenario")["yield_loss_pct"]
                  .median()
                  .sort_values(ascending=False)
                  .index.tolist())
    data   = [sub.loc[sub["scenario"] == s, "yield_loss_pct"].dropna().values
              for s in order]
    bp = ax.boxplot(data, labels=order, patch_artist=True,
                    medianprops=dict(color="black", linewidth=1.4))
    cmap = plt.colormaps["tab10"]
    for patch, i in zip(bp["boxes"], range(len(order))):
        patch.set_facecolor(cmap(i / 10))
        patch.set_alpha(0.60)
    ax.set_title("Yield loss % by scenario (top 8 by frequency)")
    ax.set_ylabel("yield_loss_pct (%)")
    ax.tick_params(axis="x", rotation=35, labelsize=7)
    ax.grid(axis="y", linestyle=":", alpha=0.45)

    # bottom-left: protein_pct histogram by product
    ax = axes[1, 0]
    merged = runs[["run_id", "product"]].merge(lab[["run_id", "protein_pct"]], on="run_id")
    for product, color, lbl in [
        ("QUARK",                "#2060aa", "Quark"),
        ("HIGH_PROTEIN_PUDDING", "#cc6600", "HP Pudding"),
    ]:
        vals = merged.loc[merged["product"] == product, "protein_pct"].dropna()
        if not vals.empty:
            ax.hist(vals, bins=20, alpha=0.55, color=color,
                    label=lbl, edgecolor="white")
    ax.set_title("Lab protein % distribution by product")
    ax.set_xlabel("protein_pct (%)")
    ax.set_ylabel("Run count")
    ax.legend(fontsize=8)
    ax.grid(axis="y", linestyle=":", alpha=0.45)

    # bottom-right: protein_pct vs yield_loss_pct scatter, coloured by product
    ax = axes[1, 1]
    merged2 = runs[["run_id", "product", "scenario", "yield_loss_pct"]].merge(
        lab[["run_id", "protein_pct"]], on="run_id"
    )
    for product, marker, color, lbl in [
        ("QUARK",                "+", "#2060aa", "Quark"),
        ("HIGH_PROTEIN_PUDDING", "x", "#cc6600", "HP Pudding"),
    ]:
        sub = merged2[merged2["product"] == product]
        ax.scatter(sub["protein_pct"], sub["yield_loss_pct"],
                   marker=marker, color=color, alpha=0.55, s=30, label=lbl)
    ax.set_title("Lab protein % vs run yield loss %")
    ax.set_xlabel("protein_pct (%)")
    ax.set_ylabel("yield_loss_pct (%)")
    ax.legend(fontsize=8)
    ax.grid(linestyle=":", alpha=0.40)

    plt.tight_layout()
    out = _PLOT_DIR / "03_distributions.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_missingness(runs: pd.DataFrame, ts: pd.DataFrame) -> Path:
    """B4: % missing per sensor column by scenario — SENSOR_FAULT and
    product-specific nulls (separator_speed_rpm for Pudding) should stand out."""
    sensor_cols = [c for c in [
        "temperature_C", "pressure_bar", "flow_rate_lpm", "shear_rpm",
        "separator_speed_rpm", "pH", "conductivity_mS", "separation_deltaP",
        "viscosity_proxy", "deltaT_heat_exchanger",
    ] if c in ts.columns]

    ts_with_scen = ts.merge(runs[["run_id", "scenario"]], on="run_id")
    scenarios    = sorted(ts_with_scen["scenario"].unique())

    miss: dict[str, dict[str, float]] = {}
    for scen in scenarios:
        sub = ts_with_scen[ts_with_scen["scenario"] == scen]
        miss[scen] = {col: 100.0 * sub[col].isna().mean() for col in sensor_cols}

    miss_df = pd.DataFrame(miss).T   # rows = scenarios, cols = sensors

    fig, ax = plt.subplots(figsize=(13, max(5, len(scenarios) * 0.55)))
    im = ax.imshow(miss_df.values, aspect="auto", cmap="YlOrRd", vmin=0, vmax=100)

    ax.set_xticks(range(len(sensor_cols)))
    ax.set_xticklabels(sensor_cols, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(len(scenarios)))
    ax.set_yticklabels(scenarios, fontsize=8)
    ax.set_title("% missing per sensor by scenario  "
                 "(product-specific nulls are expected, not faults)",
                 fontsize=10)
    plt.colorbar(im, ax=ax, label="% missing")

    # Annotate non-trivial cells
    for i, scen in enumerate(scenarios):
        for j, col in enumerate(sensor_cols):
            v = miss_df.loc[scen, col]
            if v > 3.0:
                ax.text(j, i, f"{v:.0f}%",
                        ha="center", va="center", fontsize=5.5,
                        color="black" if v < 55 else "white")

    plt.tight_layout()
    out = _PLOT_DIR / "04_missingness.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    _PLOT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data from data_raw/ ...")
    try:
        runs, ts, lab, evts = load_data()
    except FileNotFoundError as exc:
        print(f"  ERROR: {exc}")
        print("  Run the generator first:  python -m src.data_gen.run_all")
        sys.exit(1)

    print(f"  runs={len(runs):,}  timeseries={len(ts):,}  "
          f"lab={len(lab):,}  events={len(evts):,}\n")

    # ── A: Sanity checks ──────────────────────────────────────────────────────
    print("=== SANITY CHECKS ===")
    check_run_id_consistency(runs, ts, lab, evts)
    check_value_ranges(ts)
    check_ph_trajectories(runs, ts)
    check_step_ordering(runs, ts)
    check_step_durations(runs, ts)
    check_scenario_outcomes(runs)
    check_sensor_faults(runs, ts)
    check_centrifuge_ramp(runs, ts)
    check_pudding_foul_signals(runs, ts)

    fails = [c for c in _checks if c[0] == "FAIL"]
    for status, name, detail in _checks:
        tag  = "[PASS]" if status == "PASS" else "[FAIL]"
        line = f"  {tag} {name}"
        if detail:
            line += f"\n         -> {detail}"
        print(line)

    # ── B: Plots ──────────────────────────────────────────────────────────────
    print("\n=== GENERATING PLOTS ===")
    plot_fns = [
        ("Quark pH curves",       lambda: plot_quark_ph_curves(runs, ts, evts)),
        ("Pudding fouling",       lambda: plot_pudding_fouling(runs, ts, evts)),
        ("Distributions",         lambda: plot_distributions(runs, lab)),
        ("Missingness heatmap",   lambda: plot_missingness(runs, ts)),
    ]
    saved: list[Path] = []
    for label, fn in plot_fns:
        try:
            path = fn()
            if path:
                saved.append(path)
                print(f"  saved: {path.name}")
            else:
                print(f"  [SKIP] {label}: no data")
        except Exception as exc:
            print(f"  [ERROR] {label}: {exc}")

    # ── Validation summary ────────────────────────────────────────────────────
    print("\n=== VALIDATION SUMMARY ===")
    n_pass = len(_checks) - len(fails)
    print(f"  Checks: {len(_checks)}   PASS: {n_pass}   FAIL: {len(fails)}")

    if fails:
        print("  RED FLAGS (suggested fixes):")
        _FIXES: dict[str, str] = {
            "pH decreasing":
                "Check _quark_ph_curve() sigmoid parameters; rate may be too slow.",
            "pH plateau":
                "Check STALL_FERMENT stall_onset_pct and rate_multiplier=0.0.",
            "Step order":
                "Check _step_order() sorting and _draw_step_durations() for missing steps.",
            "Step durations":
                "Verify YAML duration_min bounds match _BOUNDS in check_step_durations().",
            "FOUL > NORMAL":
                "Check FOUL scenario outcome overrides in generate_run().",
            "SEPARATION_ISSUE":
                "Check SEPARATION_ISSUE yield_loss range in generate_run().",
            "SENSOR_FAULT":
                "Verify anomaly_flag is set on spike rows in _quark_ts/_pudding_ts.",
            "Centrifuge":
                "Check _centrifuge_profile() sigmoid parameters and ramp duration.",
            "pressure_bar >= 0":
                "BLOCK/SEPARATION_ISSUE may push flow/pressure negative; add np.clip.",
            "flow_rate_lpm >= 0":
                "Add np.clip(flow, 0, None) at end of _quark_ts and _pudding_ts.",
        }
        for _, name, detail in fails:
            fix = next((v for k, v in _FIXES.items() if k.lower() in name.lower()), "Review generator logic.")
            print(f"    * {name}")
            print(f"      Detail:  {detail}")
            print(f"      Fix:     {fix}")
    else:
        print("  All checks passed -- dataset looks healthy.")

    print(f"\n  Plots saved to: {_PLOT_DIR}")
    for p in saved:
        print(f"    {p}")


if __name__ == "__main__":
    main()
