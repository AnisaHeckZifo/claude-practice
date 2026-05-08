# Dairy Manufacturing PoC — Process + Lab Integration

Synthetic data platform for exploring **fouling detection**, **anomaly scoring**, and
**process–lab data fusion** in a dairy UHT / quark / high-protein pudding manufacturing context.

---

## Project layout

```
dairy_poc/
├── configs/
│   └── scenarios.yaml          # scenario params + QC spec limits
├── data_raw/                   # generated CSVs (git-ignored in production)
├── data_processed/             # merged / feature-engineered Parquet files
├── src/
│   ├── data_gen/
│   │   ├── process_data.py     # 1-second sensor time-series generator
│   │   ├── lab_data.py         # 45-min cadence QC result generator
│   │   └── run_all.py          # generate all scenarios in one shot
│   └── utils/
│       ├── align.py            # temporal merge: lab onto process timestamps
│       ├── anomaly.py          # z-score + Isolation Forest flagging
│       └── io.py               # CSV → Parquet I/O helpers
├── notebooks/                  # EDA notebooks (optional)
├── app/                        # Streamlit dashboard (next phase)
├── requirements.txt
└── README.md
```

---

## Scenarios

| Code     | Description |
|----------|-------------|
| `NORMAL` | Clean baseline run |
| `FOUL`   | Gradual heat-exchanger fouling — outlet temp drifts ↓, pressure ↑, protein % degrades |
| `SPIKE`  | Random transient sensor spikes (instrument noise / steam hammer) |
| `BLOCK`  | Sudden 55% flow drop at mid-run (partial blockage / valve fault) |

Scenario parameters and QC spec limits live in [configs/scenarios.yaml](configs/scenarios.yaml).

---

## How to generate data

### 1. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Generate all raw CSVs (all four scenarios)

```bash
cd src/data_gen
python run_all.py
```

This writes 8 files to `data_raw/`:

```
process_normal.csv   lab_normal.csv
process_foul.csv     lab_foul.csv
process_spike.csv    lab_spike.csv
process_block.csv    lab_block.csv
```

Each `process_*.csv` contains ~28,800 rows (8 h × 3,600 s/h) at 1-second resolution.
Each `lab_*.csv` contains ~11 rows (one sample every 45 min over 8 h).

### 3. Generate a single scenario (optional)

```python
from src.data_gen.process_data import generate_process_run
from src.data_gen.lab_data import generate_lab_run

proc = generate_process_run(scenario="FOUL", duration_minutes=240)
lab  = generate_lab_run(scenario="FOUL", duration_hours=4)
```

---

## Data dictionary

### Process sensors (`process_*.csv`)

| Column           | Unit   | Notes |
|------------------|--------|-------|
| `timestamp`      | —      | 1-second UTC timestamps |
| `inlet_temp_c`   | °C     | Feed temperature |
| `outlet_temp_c`  | °C     | Product temperature post-heater; drops under fouling |
| `flow_rate_lpm`  | L/min  | Volumetric flow; drops under BLOCK scenario |
| `pressure_bar`   | bar    | Line pressure; rises under fouling |
| `fouling_index`  | 0–1    | Synthetic proxy (0 = clean, 1 = fully fouled) |
| `cip_active`     | bool   | True during CIP cycle |
| `scenario`       | str    | Label for supervised training |

### Lab QC results (`lab_*.csv`)

| Column                | Unit    | Notes |
|-----------------------|---------|-------|
| `sample_time`         | —       | Timestamp sample was pulled |
| `fat_pct`             | %       | Fat content |
| `protein_pct`         | %       | Key KPI for high-protein products; degrades under FOUL |
| `total_solids_pct`    | %       | Total solids |
| `ph`                  | —       | pH; drifts under fouling |
| `viscosity_mpa_s`     | mPa·s   | Sensitive to heat damage and fouling |
| `microbial_count_cfu` | CFU/mL  | Aerobic plate count |
| `result_flag`         | str     | PASS / WARN / FAIL vs. spec limits in scenarios.yaml |
| `scenario`            | str     | Label |

---

## Next steps

1. **EDA notebook** — load process + lab CSVs, visualise signal distributions per scenario,
   identify leading indicators of fouling onset.

2. **Alignment layer** — run `src/utils/align.py` to `merge_process_lab()` and create
   a unified dataset at process cadence with lab values forward-filled.

3. **Anomaly scoring** — apply `zscore_flag` and `isolation_flag` from `src/utils/anomaly.py`
   to the merged dataset; compare flagging precision by scenario.

4. **Feature engineering** — derive rolling delta-T (outlet − inlet), pressure gradient,
   viscosity slope — the inputs the dashboard will use.

5. **Streamlit dashboard** (`app/`) — time-series viewer with scenario selector,
   anomaly overlay, and live KPI gauges for protein %, pH, fouling index.
