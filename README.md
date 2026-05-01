# SRA PySeismoSoil — EQL with Auto-Tuning

A Python script for **1-D Equivalent-Linear (EQL) site response analysis** built on top of [PySeismoSoil](https://github.com/jsh9/PySeismoSoil). The script automatically tunes soil damping and stiffness parameters so the simulated acceleration at a target depth matches a reference recording (e.g. a centrifuge sensor or a downhole accelerometer) as closely as possible.

> ⚠️ **Disclaimer**: This is research code provided "as is" for academic use. Results have not been independently verified for engineering design. The author accepts no liability for any use of this code.

## What it does

Given a Vs profile, G/Gmax & damping curves, and an input motion, the script:

1. Runs an untuned EQL simulation as a baseline.
2. (Optional) Searches across candidate Vs profiles built from basaltic-sand quantile regressions to find the best match.
3. Auto-tunes the damping per soil material (bisection on D_min, then sweep of the damping curve scale) to match the reference sensor's peak acceleration.
4. Plots G/Gmax curves, damping curves, time histories, transfer functions, and depth profiles — untuned vs tuned.
5. Writes a plain-text tuning report.

## Requirements

- Python ≥ 3.8
- `numpy`, `pandas`, `matplotlib`, `PySeismoSoil`

```bash
pip install numpy pandas matplotlib PySeismoSoil
```

## How to use

### 1. Set your paths and filenames

Open `SRA_PySeismoSoil_tune__v01.py` and edit **Section 1**. By default everything is resolved relative to the script's location:

```python
BASE_DIR   = Path(__file__).resolve().parent
MOTION_DIR = BASE_DIR
SRA_DIR    = BASE_DIR
SAVE_DIR   = BASE_DIR / "EQL_output"

VS_PROFILE_NAME = "Vs_profile_EQL.txt"
CURVES_NAME     = "Daming_and_GGmax.csv"
FNAME           = "ACC.lvm"
```

Drop your three input files next to the script and you're ready to run. To use absolute paths, override `MOTION_DIR` / `SRA_DIR` / `SAVE_DIR` directly.

### 2. Check the input file formats

| File | Format |
|---|---|
| **Vs profile** (`.txt`) | 5 columns, space-delimited, no header: `thickness[m]  Vs[m/s]  D_min[%]  rho[kg/m3]  Material`. Last row is the half-space (thickness = 0, Material = 0). |
| **G/Gmax & D curves** (`.csv`) | Header row required, 3 columns: `strain_percentage`, `D_percentage`, `G_Gmax`. Strain and D in **percent**, not decimal. |
| **Input motion** (`.lvm`) | Tab-delimited LabView file with 23 header rows. Must contain `X_Value` (time) plus one column per channel. Set `BASE_INPUT_COL` (drives the simulation) and `SENSOR_NAME` (reference) in Section 1. |

### 3. Run it

```bash
python SRA_PySeismoSoil_tune__v01.py
```

A run takes a few minutes (longer if `ENABLE_SAND_SEARCH = True`). Plots open one after another, console prints the GOF metrics, and a `.txt` tuning report is written to `SAVE_DIR`.

## Outputs

- **Plots** (on screen): G/Gmax curves, damping curves, time history at target depth, transfer function, peak depth profiles — all showing untuned vs tuned side by side.
- **`<motion_name>__tuning_report__EQL__<datetime>.txt`** — run metadata, tuned parameters per material, GOF metrics (RMSE, peak ratio, cross-correlation, frequency shift), tuned soil profile, strain regime per layer, and the tuned G/Gmax & D curves.
- **`<motion_name>__sand_winner__<label>.txt`** — winning Vs profile (only when the sand search is enabled).

## How the tuning works

PySeismoSoil computes total damping at each strain as `D_total = D_min (from Vs profile) + D_curve (from CSV)`. Both terms affect the EQL response. The script tunes three knobs **per soil material**:

- **`d_min_offset`** — shifts D_min up or down to control peak amplitude
- **`d_scale`** — multiplies the damping curve to control overall energy dissipation
- **`strain_shift`** — shifts the strain axis (manual; only needed when the transfer-function peak frequency is wrong)

For each material, a bisection search finds the `d_min_offset` that brings the peak acceleration ratio (sim/reference) into the target window (default 0.95–1.05). Then `d_scale` is swept over a small set of values and the one giving the lowest RMSE is kept. The optional sand search wraps this whole tuning routine in an outer loop over candidate Vs profiles, picks the best one by a weighted combination of peak ratio, RMSE, and TF-frequency shift, and applies physical-consistency guardrails.

## Troubleshooting

- **`FileNotFoundError: ... not found`** — the script's startup check tells you exactly which file and which `_DIR` variable to fix in Section 1. This is almost always a path or filename typo.
- **`KeyError: 'X_Value'` or similar** — your `.lvm` file uses different channel names. Open it in a text editor, find the actual column names, and update `BASE_INPUT_COL` and `SENSOR_NAME` in Section 1. The `header=22` in `pd.read_table` assumes 23 header rows; change it if your file has a different header length.
- **`Strain must be strictly increasing`** — the strain column in your curves CSV has duplicate or out-of-order values. Sort by strain ascending and remove duplicates.
- **Peak ratio doesn't reach 0.95–1.05 after 20 iterations** — widen `TUNE_D_MIN_BOUNDS` in Section 1 (e.g. `(-5.0, 15.0)`), or raise `TUNE_MAX_ITER`. If the issue is a TF frequency mismatch rather than amplitude, set `STRAIN_SHIFTS_MANUAL = {1: 0.7, 2: 1.0}` (values < 1.0 shift curves left).
- **Sand search "NO CANDIDATE PASSED THE PHYSICAL GUARDRAILS"** — your reference data may genuinely require an inverted layering, or the GOF weights are too aggressive on amplitude. Either set `ENABLE_PHYSICAL_GUARDRAILS = False`, raise `D_TOTAL_CAP_PCT`, or relax `GOF_WEIGHTS`.
- **Sand search takes too long** — lower `TOP_N_REFINE` (e.g. 3 instead of 5), lower `SCREEN_BISECTION_ITER`, or set `ENABLE_SAND_SEARCH = False` to skip the search entirely.

## Citation

If you use this code in published work, please cite:

- **PySeismoSoil library** — Shi, J. (Caltech), https://github.com/jsh9/PySeismoSoil
- **Liquefaction behavior of Icelandic basaltic sand under monotonic and cyclic direct simple shear loadings** — [doi:10.1007/s10706-025-03496-2](https://doi.org/10.1007/s10706-025-03496-2)
- **Database of measured shear wave velocity profiles for Icelandic soil sites** — [doi:10.1201/9781003431749-627](https://doi.org/10.1201/9781003431749-627)

## License

This project is licensed under the MIT License — see the `LICENSE` file for details.

Copyright © 2026 Javad Fattahi

## Author

Javad Fattahi — sjf7@hi.is
