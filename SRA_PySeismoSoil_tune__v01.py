"""
================================================================================
Site Response Analysis (SRA) using PySeismoSoil  -  EQL only
================================================================================
Author       : Javad Fattahi
Email        : sjf7@hi.is
Version      : v01 (public)

What this code does
-------------------
1-D Equivalent-Linear (EQL) site response analysis using the PySeismoSoil
library (Caltech). The code automatically tunes the soil parameters so that
the simulated acceleration at TARGET_DEPTH matches a reference (e.g. a
centrifuge, borehole) as closely as possible.

Required Python packages (install via pip):
    numpy, pandas, matplotlib, PySeismoSoil

How tuning works
----------------
PySeismoSoil computes total damping internally as:
    D_total(strain) = D_min [Vs profile, col 3]
                    + D_curve(strain) [external CSV]
EQL is sensitive to BOTH terms. The code adjusts both per material:
    1. d_min_offset  -- shifts D_min up or down [%]
    2. d_scale       -- multiplies the D_curve [-]
    3. strain_shift  -- shifts the strain axis (manual, for TF freq fix)

For each material the code runs a bisection search on d_min_offset until
the peak amplitude at TARGET_DEPTH matches the reference (peak ratio in
[0.95, 1.05]), then sweeps d_scale to minimise RMSE.

Vs-profile search
-------------------------
On top of the per-material tuning, the code optionally searches across
several Vs profiles built from quantile regressions Vs(z) = a * z^b for
basaltic sands (https://doi.org/10.1007/s10706-025-03496-2). Quantiles independently per
material gives Vs profile combinations. A hard cap D_TOTAL_CAP_PCT keeps total damping physically
realistic for clean basaltic sand.

Two-pass speed design:
    Pass 0  Vs-ordering prefilter (zero EQL cost, drops ~half the candidates)
    Pass 1  Coarse bisection (8 iter, no d_scale sweep) on survivors
    Pass 2  Full tuning on the top-N finalists from Pass 1
Result: faster than running full tuning on every candidate.

Physical-consistency guardrails
-------------------------------
The search optimiser is purely numerical - it can in principle pick
combinations that violate basic soil mechanics. The guardrails enforce:
    Mat2 (denser) avg Vs   > Mat1 (looser) avg Vs
    Mat2 (denser) D_min    < Mat1 (looser) D_min
The chosen winner is the highest-GOF candidate that passes both rules.
If no candidate passes, the code falls back to the highest-GOF candidate
overall and flags the result as physically inconsistent.

Plots and figures (per run)
---------------------------
    Plot A  G/Gmax curves per material - untuned vs tuned
    Plot B  D curves per material      - untuned vs tuned (with D_total)
    Plot C  Time history at TARGET_DEPTH - reference | untuned | tuned
    Plot D  Transfer function          - reference | untuned | tuned
    Section 6.5 also prints a 5-panel diagnostic depth profile.

Files written to SAVE_DIR
-------------------------
    {FNAME stem}__tuning_report__EQL__{date_time}.txt
        Plain-text report with run metadata, tuning levers, GOF metrics,
        soil profile after tuning, strain regime per layer, and the tuned
        G/Gmax & D curves used by EQL.
    {FNAME stem}__sand_winner__{label}.txt
        Winning Vs profile when ENABLE_SAND_SEARCH = True.

Profile file format (default: Vs_profile_EQL.txt)
-------------------------------------------------
    5 columns, space-delimited, no header:
        Thickness [m] | Vs [m/s] | D_min [%] | rho [kg/m3] | Material
    Last row = half-space (thickness = 0).
    Material 0 = half-space (never tuned).
    Material numbers are read automatically from column 5.

External curves CSV format (default: GGmax_D_curves.csv)
--------------------------------------------------------
    Header row required. 3 columns:
        strain_percentage [%] | D_percentage [%] | G_Gmax [-]
    Strain in percent, NOT decimal. D in percent.

Input motion file (default: LabView .lvm format)
------------------------------------------------
    Tab-delimited with 23 header rows. Must contain a time column "X_Value"
    and one acceleration column per channel. The base column is named "A??"
    by default; the comparison sensor name is set in SENSOR_NAME.
================================================================================
"""

# ── Standard library ─────────────────────────────────────────────────────────
import os
from pathlib import Path
from datetime import datetime
from itertools import product
from numpy.fft import rfft, rfftfreq

# ── Third-party ──────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ── PySeismoSoil ─────────────────────────────────────────────────────────────
from PySeismoSoil.class_Vs_profile import Vs_Profile
from PySeismoSoil.class_ground_motion import Ground_Motion
from PySeismoSoil.class_simulation import Equiv_Linear_Simulation
from PySeismoSoil.class_curves import Multiple_GGmax_Damping_Curves


# ==============================================================================
#  INLINED HELPER 1 - Extract acceleration time history at a target depth
# ==============================================================================

def func_extract_sublayers_v1(res, target_z):
    """
    Pull the acceleration time history at the depth closest to `target_z`
    out of a PySeismoSoil Simulation_Results object.

    Parameters
    ----------
    res      : Simulation_Results  must have been run with every_layer=True
    target_z : float               extraction depth [m, prototype scale]

    Returns
    -------
    t              : np.ndarray  time vector [s]
    acc_subLayer   : np.ndarray  acceleration at the closest layer [m/s^2]
    target_z       : float       the target depth that was requested [m]
    """
    if res is None:
        raise ValueError(
            "Simulation result 'res' is None (run the simulation first)."
        )

    # depth array and accel matrix (all layer boundaries)
    z = res.rediscretized_profile.get_depth_array()
    A = res.time_history_accel

    # ensure A is (n_time, n_depth)
    if A.shape[0] == z.size:
        A = A.T

    i = int(np.argmin(np.abs(z - target_z)))
    acc_subLayer = A[:, i]

    # time vector (use the same time base as the output surface motion)
    t = res.accel_on_surface.time

    return t, acc_subLayer, target_z


# ==============================================================================
#  SECTION 1 - USER CONTROL PANEL
#  All settings you will ever need to change are here.
# ==============================================================================

# Boundary condition:
#   'rigid'   -> borehole motion at base of profile 
#   'elastic' -> rock-outcrop free-surface motion
BOUNDARY_CONDITION = "rigid"

# Display figures?
SHOW_FIGURES = True

# Scale factor N. Set N_SCALE = 1 if your input
# motion is already at prototype scale (i.e. not centrifuge data).
N_SCALE = 1

# Gravitational acceleration [m/s^2]
G_GRAVITY = 9.80665

# ── Input/Output directories ─────────────────────────────────────────────────
# Three directories are needed:
#   MOTION_DIR  - where the LabView .lvm motion file lives
#   SRA_DIR     - where the Vs profile + G/Gmax & D curves CSV live
#   SAVE_DIR    - where the tuning .txt report and Sand winner profile land

BASE_DIR  = Path(__file__).resolve().parent
MOTION_DIR = str(BASE_DIR)
SRA_DIR    = str(BASE_DIR)
SAVE_DIR   = str(BASE_DIR / "EQL_output")

# ── Input filenames ─────────────────────────────────────────────────────────
# VS_PROFILE_NAME and CURVES_NAME are looked up inside SRA_DIR.
# FNAME is looked up inside MOTION_DIR.
VS_PROFILE_NAME = "Vs_profile_EQL.txt"
CURVES_NAME     = "Daming_and_GGmax.csv"
FNAME           = "ACC.lvm"

# Time-window slice on the input motion file (samples). Leave SLICE_END = None
# to use the entire record after SLICE_START.
SLICE_START = 25_000
SLICE_END   = None


# Reference sensor column name in the input motion file (comparison target).
# This is the channel the simulation will be tuned to match.
SENSOR_NAME = "A??"

# Base motion column name in the input motion file (drives the simulation).
BASE_INPUT_COL = "A??"

# Depth to extract simulation results [m].
# The reference sensor (SENSOR_NAME) is assumed to record this depth.
TARGET_DEPTH = 3

# ── Automatic tuning settings ────────────────────────────────────────────────
# Target peak_ratio range for bisection convergence
TUNE_RATIO_TARGET   = (0.95, 1.05)

# Search bounds for d_min_offset [%] (lo, hi)
TUNE_D_MIN_BOUNDS   = (-2.0, 8.0)

# Maximum bisection iterations per material
TUNE_MAX_ITER       = 20

# After d_min converges, sweep d_scale to minimise RMSE?
TUNE_D_SCALE        = True

# d_scale values to try (applied after d_min_offset is found)
TUNE_D_SCALE_VALUES = [0.5, 0.7, 1.0, 1.2, 1.5, 2.0]

# Manual strain_shift per material.
# Only change if the transfer function PEAK FREQUENCY is shifted.
# Values < 1.0 shift curves LEFT (earlier softening -> lower natural frequency).
# Format: {material_id: shift_value}  e.g. {1: 0.7, 2: 1.0}
# Leave as empty dict {} to use 1.0 (neutral) for all materials.
STRAIN_SHIFTS_MANUAL = {}


# ── Vs profile search + damping cap ─────────────────────────────────
# Searches over all quantile profiles Vs(z) independently
# per material plus the original file profile 
# candidates. For each, runs the per-material tuning under a hard cap
# on D_total. Picks the candidate with the lowest combined GOF score.
ENABLE_SAND_SEARCH      = True   # master switch
ENABLE_PHYSICAL_GUARDRAILS = True   # reject physically inconsistent winners:
                                    #   Mat2 Vs must be > Mat1 Vs
                                    #   Mat2 D_min must be < Mat1 D_min
D_TOTAL_CAP_PCT = 3.0               # hard cap on D_total at every strain [%]

# ── Two-pass speed settings ─────────────────────────────────────────────────
# The Sand search runs all candidates. With full tuning per candidate
# Two-pass approach is faster.
SCREEN_BISECTION_ITER = 8           # bisection iterations during fast screen
TOP_N_REFINE          = 5           # number of finalists to fully tune
PREFILTER_VS_ORDERING = True        # skip Vs-ordering-violating candidates
                                    # before any EQL is run (zero EQL cost)

# Reporting at the end of the search
LEADERBOARD_TOP_K  = 5              # candidates shown in summary tables
NEAR_TIE_THRESHOLD = 0.05           # near-tie flag threshold
GOF_WEIGHTS        = (1.0, 1.0, 0.05)
                                    # (w_ratio, w_rmse, w_freq)

# Sand quantile regression coefficients (Vs in m/s, z in m).
#   Vs(z) = a * z^b
# q is the quantile level (0.5 = median, 0.023 = -2 SD, 0.977 = +2 SD).
basaltic_QUANTILES = [
    # (label,        q,      a,       b)
    ("Q69_2pct",    1,  2,  3),   #<==  it should be in this format, replace ?? with values
    ("Q97_2pct",    1,  2,  3),
]


# ==============================================================================
#  SECTION 2 - PATH RESOLUTION
#
#  Combines the MOTION_DIR / SRA_DIR / SAVE_DIR / filenames set in Section 1
#  into the full paths used by the rest of the script. Edit the directories
#  in Section 1, NOT here.
# ==============================================================================

VS_PROFILE_PATH   = os.path.join(SRA_DIR, VS_PROFILE_NAME)
CURVES_PATH       = os.path.join(SRA_DIR, CURVES_NAME)
                     
INPUT_MOTION_PATH = os.path.join(MOTION_DIR, FNAME)

os.makedirs(SAVE_DIR, exist_ok=True)


for _label, _path, _dir_var in [
    ("Vs profile",   VS_PROFILE_PATH,   "SRA_DIR"),
    ("Curves CSV",   CURVES_PATH,       "SRA_DIR"),
    ("Input motion", INPUT_MOTION_PATH, "MOTION_DIR"),
]:
    if _path is not None and not os.path.isfile(_path):
        raise FileNotFoundError(
            f"\n  {_label} file not found:\n    {_path}\n"
            f"  -> Edit {_dir_var} (and the filename) in Section 1 of the script."
        )


# ==============================================================================
#  SECTION 3 - DATA LOADING & PRE-PROCESSING
# ==============================================================================

print(f"\n[INFO] Loading: {INPUT_MOTION_PATH}")
df = pd.read_table(INPUT_MOTION_PATH, header=22)


time_proto = df["X_Value"] * N_SCALE
dt         = float(time_proto.iloc[1] - time_proto.iloc[0])
print(f"[INFO] Prototype time step dt = {dt:.6f} s")


_slice_end  = SLICE_END if SLICE_END is not None else len(df)
_time_slice = time_proto.iloc[SLICE_START:_slice_end]
time_zero   = (_time_slice - _time_slice.iloc[0]).to_numpy(dtype=float)


acc_base_raw = df[BASE_INPUT_COL].iloc[SLICE_START:_slice_end].to_numpy(dtype=float)
acc_base_ms2 = acc_base_raw * G_GRAVITY / N_SCALE


ACC_base = np.column_stack((time_zero, acc_base_ms2))


acc_centrifuge = (
    df[SENSOR_NAME].iloc[SLICE_START:_slice_end].to_numpy(dtype=float)
    * G_GRAVITY / N_SCALE
)

print(f"[INFO] Base acc    : {acc_base_ms2.min():.3f} - {acc_base_ms2.max():.3f} m/s^2")
print(f"[INFO] Sensor {SENSOR_NAME}  : "
      f"{acc_centrifuge.min():.3f} - {acc_centrifuge.max():.3f} m/s^2")


# ==============================================================================
#  SECTION 4 - PySeismoSoil INPUT OBJECTS
# ==============================================================================

print(f"\n[INFO] Reading Vs profile: {VS_PROFILE_PATH}")
# Column order in file: Thk[m], Vs[m/s], D[%], rho[kg/m^3], MaterialNo
vs_profile   = Vs_Profile(VS_PROFILE_PATH,
                          damping_unit="%",
                          density_unit="kg/m^3")
input_motion = Ground_Motion(ACC_base, unit="m/s/s")

vs_profile.summary()
if SHOW_FIGURES:
    input_motion.plot()


# ==============================================================================
#  SECTION 5 - HELPER FUNCTIONS, CURVE LOADING & MATERIAL DETECTION
#
#  Key design principle:
#    run_eql() is the SINGLE function used to build curves and run EQL.
#    All runs (untuned baseline in Section 6, each bisection iteration in
#    Section 7.5, and the final tuned run) call run_eql() with different
#    parameter dictionaries. This guarantees a fair comparison because
#    both untuned and tuned results are built from identical code.
# ==============================================================================

def compute_tf(acc_out, acc_in, dt_val, smooth=7):
    """
    Transfer function H(f) = FFT(output) / FFT(input).

    Returns
    -------
    freqs : array  [Hz]
    amp   : array  smoothed |H(f)|
    """
    n     = min(len(acc_out), len(acc_in))
    freqs = rfftfreq(n, d=dt_val)
    H     = rfft(acc_out[:n]) / (rfft(acc_in[:n]) + 1e-12)
    amp   = np.abs(H)
    amp   = np.convolve(amp, np.ones(smooth) / smooth, mode="same")
    return freqs, amp


def gof_metrics(acc_sim, acc_ref):
    """
    Goodness-of-fit metrics.

    Returns
    -------
    rmse, peak_sim, peak_ref, ratio
    """
    n        = min(len(acc_sim), len(acc_ref))
    rmse     = float(np.sqrt(np.mean((acc_sim[:n] - acc_ref[:n]) ** 2)))
    peak_sim = float(np.max(np.abs(acc_sim[:n])))
    peak_ref = float(np.max(np.abs(acc_ref[:n])))
    ratio    = peak_sim / peak_ref if peak_ref > 0 else np.nan
    return rmse, peak_sim, peak_ref, ratio


def run_eql(profile_arr, strain_base, Gg_base, D_base,
            mat_ids, mat_col,
            d_min_offsets, d_scales, strain_shifts,
            input_motion, boundary, target_depth,
            verbose=False, show_fig=False,
            d_total_cap=None):
    """
    Build a tuned Vs profile + curve matrix and run one EQL simulation.

    This is the canonical EQL runner. ALWAYS use this function - never call
    Equiv_Linear_Simulation directly elsewhere in this script.

    Parameters
    ----------
    profile_arr    : np.ndarray  raw Vs profile (from np.loadtxt) - NOT modified
    strain_base    : array       base strain values from external CSV [%]
    Gg_base        : array       base G/Gmax values [-]
    D_base         : array       base D values [%]
    mat_ids        : array       material IDs to process (e.g. [1, 2])
    mat_col        : array       material number column from profile_arr
    d_min_offsets  : dict        {mat_id: offset [%]}  added to D_min
    d_scales       : dict        {mat_id: D_curve scale [-]}
    strain_shifts  : dict        {mat_id: strain axis multiplier [-]}
    input_motion   : Ground_Motion object
    boundary       : str         'rigid' or 'elastic'
    target_depth   : float       extraction depth [m]
    verbose        : bool        print EQL convergence
    show_fig       : bool        show PySeismoSoil result figure
    d_total_cap    : float|None  hard cap on D_total at every strain [%].
                                  None = use a relaxed cap of 30 %.

    Returns
    -------
    time_out, acc_out, depth_out, sim_res, curve_arrs
    """
    # ── Build tuned profile ───────────────────────────────────────────────
    # Pass a copy: Vs_Profile modifies its input in-place (D % -> decimal)
    profile_copy = profile_arr.copy()
    for mid in mat_ids:
        mask   = mat_col == mid
        offset = d_min_offsets.get(mid, 0.0)
        d_new  = np.clip(profile_arr[mask, 2] + offset, 0.1, 15.0)
        profile_copy[mask, 2] = d_new

    vs_prof = Vs_Profile(profile_copy,
                         damping_unit='%',
                         density_unit='kg/m^3')

    # ── Build curve matrix (one block per material, stacked side by side) ──
    curve_blocks = []
    curve_arrs   = {}

    for mid in mat_ids:
        ss      = strain_shifts.get(mid, 1.0)
        ds      = d_scales.get(mid, 1.0)
        offset  = d_min_offsets.get(mid, 0.0)
        d_min_t = float(np.clip(profile_arr[mat_col == mid, 2][0] + offset,
                                0.1, 15.0))

        strain_m    = strain_base * ss
        D_curve_raw = D_base * ds
        if d_total_cap is not None:
            curve_cap_pct = max(0.0, float(d_total_cap) - d_min_t)
            D_m = np.clip(D_curve_raw, 0.0, curve_cap_pct)
        else:
            D_m = np.clip(D_curve_raw, 0.0, 30.0)
        Gg_m = Gg_base.copy()

        assert np.all(np.diff(strain_m) > 0), (
            f"Material {mid}: strain not strictly increasing "
            f"after strain_shift={ss}."
        )

        # PySeismoSoil 4-column layout: [strain%, G/Gmax, strain%, D%]
        curve_blocks.append(np.column_stack([strain_m, Gg_m, strain_m, D_m]))

        curve_arrs[mid] = {
            'strain'  : strain_m,
            'Gg'      : Gg_m,
            'D_curve' : D_m,
            'D_total' : D_m + d_min_t,
            'd_min'   : d_min_t,
        }

    curve_matrix = np.hstack(curve_blocks)
    n_mat_local  = len(mat_ids)
    assert curve_matrix.shape[1] == 4 * n_mat_local, (
        f"Curve matrix shape error: expected {4*n_mat_local} cols, "
        f"got {curve_matrix.shape[1]}."
    )

    curves_obj = Multiple_GGmax_Damping_Curves(data=curve_matrix)

    # ── Run EQL ───────────────────────────────────────────────────────────
    sim = Equiv_Linear_Simulation(vs_prof, input_motion,
                                  curves_obj, boundary=boundary)
    sim_res = sim.run(verbose=verbose, show_fig=show_fig)

    # ── Extract result at target depth ────────────────────────────────────
    t_out, a_out, d_out = func_extract_sublayers_v1(sim_res, target_depth)
    return t_out, a_out, d_out, sim_res, curve_arrs


# ── Load external CSV base curves ─────────────────────────────────────────
if CURVES_PATH is None:
    raise RuntimeError(
        "This release requires external G/Gmax & D curves "
        "The empirical / HH-Darendeli branch is not included."
    )

print("\n[INFO] Loading external G/Gmax & D curves ...")
df_curves   = pd.read_csv(CURVES_PATH)

# Confirmed column order: strain_percentage, D_percentage, G_Gmax
strain_base = df_curves["strain_percentage"].to_numpy(float)   # [%]
D_base      = df_curves["D_percentage"].to_numpy(float)        # [%]
Gg_base     = df_curves["G_Gmax"].to_numpy(float)              # [-]

idx                          = np.argsort(strain_base)
strain_base, D_base, Gg_base = strain_base[idx], D_base[idx], Gg_base[idx]
assert np.all(np.diff(strain_base) > 0), "Strain must be strictly increasing."

op_mask = (strain_base >= 0.001) & (strain_base <= 0.1)
print(f"  Strain range : {strain_base.min():.4f} - {strain_base.max():.1f}  %")
print(f"  G/Gmax range : {Gg_base.min():.4f} - {Gg_base.max():.4f}  [-]")
print(f"  D range      : {D_base.min():.3f}  - {D_base.max():.3f}   %")


# ── Load Vs profile raw array & detect material structure ─────────────────
profile_original = np.loadtxt(VS_PROFILE_PATH)

thk_col      = profile_original[:, 0]
vs_col       = profile_original[:, 1]
d_col        = profile_original[:, 2]
rho_col      = profile_original[:, 3]
mat_col      = profile_original[:, 4].astype(int)
depth_tops_p = np.concatenate(([0.0], np.cumsum(thk_col[:-1])))
depth_bots_p = np.cumsum(thk_col)

mat_ids = np.unique(mat_col[mat_col > 0])   # e.g. array([1, 2])
n_mat   = len(mat_ids)

print(f"\n[INFO] Materials detected from Vs profile: {mat_ids.tolist()}")
for mid in mat_ids:
    mask = mat_col == mid
    rows = np.where(mask)[0]
    print(f"  Material {mid}:  "
          f"depth {depth_tops_p[rows[0]]:.2f}-{depth_bots_p[rows[-1]]:.2f} m  "
          f"Vs {vs_col[mask].min():.0f}-{vs_col[mask].max():.0f} m/s  "
          f"D_min={d_col[mask][0]:.3f} %")
hs = mat_col == 0
print(f"  Half-space (0): Vs={vs_col[hs][0]:.0f} m/s  (NOT tuned)")

# Fill any missing strain_shifts with 1.0 (neutral)
for mid in mat_ids:
    if mid not in STRAIN_SHIFTS_MANUAL:
        STRAIN_SHIFTS_MANUAL[mid] = 1.0


# ==============================================================================
#  SECTION 6 - RUN UNTUNED EQL SIMULATION
#  Calls run_eql() with all neutral levers (offset=0, scale=1, shift=1).
#  This is the true untuned baseline - same code path as tuned run.
# ==============================================================================

print("\n[INFO] Running EQL simulation (untuned baseline) ...")

neutral_offsets = {mid: 0.0 for mid in mat_ids}
neutral_scales  = {mid: 1.0 for mid in mat_ids}
neutral_shifts  = {mid: 1.0 for mid in mat_ids}

(time_untuned, acc_untuned, depth_untuned,
 sim_results_1, curve_arrs_untuned) = run_eql(
    profile_original, strain_base, Gg_base, D_base,
    mat_ids, mat_col,
    d_min_offsets=neutral_offsets,
    d_scales=neutral_scales,
    strain_shifts=neutral_shifts,
    input_motion=input_motion,
    boundary=BOUNDARY_CONDITION,
    target_depth=TARGET_DEPTH,
    verbose=True,
    show_fig=SHOW_FIGURES,
)
print(f"[INFO] Untuned extracted depth: {depth_untuned:.3f} m")


# ==============================================================================
#  SECTION 6.5 - SIMULATION DIAGNOSTICS
#  Depth-profile plots from the untuned simulation.
# ==============================================================================

mss  = sim_results_1.max_strain_stress
mavd = sim_results_1.max_a_v_d
prof = sim_results_1.rediscretized_profile.vs_profile

depth_mid_diag = mss[:,  0]
depth_bnd_diag = mavd[:, 0]
max_strain     = mss[:, 1] * 100.0    # decimal fraction -> %
max_stress     = mss[:, 2] / 1000.0   # Pa -> kPa
max_acc_diag   = mavd[:, 1]           # m/s^2

# D from rediscretised profile is stored as decimal fraction internally
thk_r       = prof[:, 0]
n_finite_r  = int(np.sum(thk_r > 0))
D_diag      = prof[:n_finite_r, 2]    # decimal fraction
Vs_diag     = prof[:n_finite_r, 1]    # m/s
depths_r    = np.concatenate(([0.0], np.cumsum(thk_r[:n_finite_r])))
depth_mid_r = (depths_r[:-1] + depths_r[1:]) / 2.0

def strain_regime(s):
    if   s > 0.1:  return "NONLINEAR  <<<<"
    elif s > 0.01: return "Transitional <"
    else:          return "Linear"

print(f"\n[DIAGNOSTICS] gamma_max : {max_strain.min():.4f} - "
      f"{max_strain.max():.4f}  %")
print(f"[DIAGNOSTICS] D_min     : {D_diag.min()*100:.4f} - "
      f"{D_diag.max()*100:.4f}  %")

if SHOW_FIGURES:
    fig, axes = plt.subplots(1, 5, figsize=(20, 7), sharey=True)
    fig.suptitle(
        f"Untuned EQL Diagnostics  |  {FNAME}  |  boundary={BOUNDARY_CONDITION}",
        fontsize=12,
    )

    ax = axes[0]
    ax.plot(max_strain, depth_mid_diag, "o-", color="darkorange", lw=1.2, ms=4)
    ax.axvspan(0.0,  0.01, alpha=0.10, color="green", label="Linear")
    ax.axvspan(0.01, 0.1,  alpha=0.10, color="gold",  label="Transit.")
    ax.axvspan(0.1,  100., alpha=0.10, color="red",   label="Nonlinear")
    ax.set_xlabel("gamma_max (%)"); ax.set_ylabel("Depth (m)")
    ax.set_xscale("log"); ax.invert_yaxis()
    ax.grid(True, which="both", ls="--", alpha=0.4)
    ax.set_title("Peak Strain"); ax.legend(fontsize=7, frameon=False)
    for i, (s, d) in enumerate(zip(max_strain, depth_mid_diag)):
        if i % 4 == 0:
            ax.annotate(f"{s:.4f}%", xy=(s, d), xytext=(4, 0),
                        textcoords="offset points", fontsize=6.5, color="grey")

    ax = axes[1]
    ax.plot(max_stress, depth_mid_diag, "D-", color="purple", lw=1.2, ms=4)
    ax.set_xlabel("tau_max (kPa)"); ax.invert_yaxis()
    ax.grid(True, ls="--", alpha=0.4); ax.set_title("Peak Shear Stress")

    ax = axes[2]
    ax.plot(max_acc_diag, depth_bnd_diag, "s-", color="crimson", lw=1.2, ms=4)
    ax.set_xlabel("Max accel. (m/s^2)"); ax.invert_yaxis()
    ax.grid(True, ls="--", alpha=0.4); ax.set_title("Peak Acceleration")

    ax = axes[3]
    # ── D_min vs D_total per layer ─────────────────────────────────────────
    D_min_pct = D_diag * 100.0
    ax.plot(D_min_pct, depth_mid_r, "^-", color="steelblue", lw=1.2, ms=4,
            label=r"$D_{\min}$ (Vs file)")

    # Effective strain per layer (gamma_eff = 0.65 * gamma_max, in %)
    EFF_STRAIN_FACTOR = 0.65
    gamma_eff = EFF_STRAIN_FACTOR * max_strain
    D_curve_pct  = np.interp(gamma_eff, strain_base, D_base)
    D_min_at_mid = np.interp(depth_mid_diag, depth_mid_r, D_min_pct)
    D_total_pct  = D_min_at_mid + D_curve_pct

    ax.plot(D_total_pct, depth_mid_diag, "o-", color="crimson", lw=1.2, ms=4,
            label=(r"$D_{\mathrm{total}}=D_{\min}+"
                   r"D_{\mathrm{curve}}(\gamma_{\mathrm{eff}})$"))

    for i, (dt_v, d_v, g_v) in enumerate(zip(D_total_pct,
                                             depth_mid_diag,
                                             gamma_eff)):
        if i % 4 == 0:
            ax.annotate(rf"$\gamma_{{\mathrm{{eff}}}}$={g_v:.3f}%",
                        xy=(dt_v, d_v), xytext=(4, 0),
                        textcoords="offset points",
                        fontsize=6.5, color="grey")

    ax.set_xlabel("Damping D (%)"); ax.invert_yaxis()
    ax.grid(True, ls="--", alpha=0.4)
    ax.set_title("Damping Profile\n"
                 r"($D_{\min}$ vs $D_{\mathrm{total}}$)")
    ax.legend(fontsize=7, frameon=False, loc="best")

    ax = axes[4]
    ax.plot(Vs_diag, depth_mid_r, "v-", color="teal", lw=1.2, ms=4)
    ax.set_xlabel("Vs (m/s)"); ax.invert_yaxis()
    ax.grid(True, ls="--", alpha=0.4); ax.set_title("Vs Profile")

    plt.tight_layout(); plt.show()

# Layer table
print(f"\n  {'#':>3}  {'Depth(m)':>9}  {'gamma_max(%)':>13}  "
      f"{'tau_max(kPa)':>13}  Regime")
print(f"  {'-'*65}")
for i, (d, s, tau) in enumerate(zip(depth_mid_diag, max_strain, max_stress)):
    print(f"  {i+1:>3}  {d:>9.3f}  {s:>13.5f}  {tau:>13.3f}  {strain_regime(s)}")


# ==============================================================================
#  SECTION 7 - INITIAL COMPARISON PLOT (untuned sim vs reference)
# ==============================================================================

if SHOW_FIGURES and time_untuned is not None:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(time_untuned, acc_untuned, color="red", lw=0.8,
            label="Simulation - untuned")
    ax.plot(time_zero, acc_centrifuge, color="blue", lw=1.5,
            ls="--", alpha=0.5, label=f"Reference ({SENSOR_NAME})")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Acceleration (m/s^2)")
    ax.set_title(f"Untuned simulation at ~{depth_untuned:.3f} m  |  {FNAME}")
    ax.grid(True, ls="--", alpha=0.4)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1),
              borderaxespad=0.1, frameon=False)
    plt.tight_layout(); plt.show()


# ==============================================================================
#  SECTION 7.4 - Vs PROFILE + DAMPING CAP SEARCH
#
#  Activates when ENABLE_SAND_SEARCH = True.
#
#  Algorithm
#  ---------
#  1. Build candidate Vs profiles by replacing the file Vs values with
#     Vs(z) = a*z^b at each layer mid-depth, using all quantile regressions
#     independently per material.
#  2. For each candidate, run per-material tuning (bisection on d_min_offset
#     plus d_scale sweep) with d_total_cap = D_TOTAL_CAP_PCT.
#  3. Score each result by:
#         score = w_ratio * |peak_ratio - 1|
#               + w_rmse  * RMSE
#               + w_freq  * |frequency_shift|
#  4. The candidate with the LOWEST score is selected. profile_original is
#     replaced with the winning Vs profile so Section 7.5 fine-tunes on top.
# ==============================================================================

_sand_active = ENABLE_SAND_SEARCH

if _sand_active:
    print(f"\n{'='*70}")
    print(f"  SECTION 7.4 - Vs SEARCH + DAMPING CAP")
    print(f"  D_TOTAL_CAP_PCT = {D_TOTAL_CAP_PCT:.2f} %")
    print(f"  GOF_WEIGHTS     = {GOF_WEIGHTS}  (w_ratio, w_rmse, w_freq)")
    print(f"{'='*70}\n")

    # ── Step 1: Build candidate Vs profile list ───────────────────────────
    def build_basaltic_profile(profile_file, mat_col_arr, mat_ids_arr,
                              quant_per_mat):
        """
        Return a Vs profile array with Vs values substituted by basaltic
        Vs(z)=a*z^b at each finite layer's mid-depth.
        """
        prof = profile_file.copy()
        thk  = prof[:, 0]
        n_finite = int(np.sum(thk > 0))

        bot = np.cumsum(thk[:n_finite])
        top = np.concatenate(([0.0], bot[:-1]))
        mid = (top + bot) / 2.0

        for i in range(n_finite):
            mid_id = int(mat_col_arr[i])
            if mid_id not in mat_ids_arr:
                continue   # half-space row, leave alone
            _label, _q, a, b = quant_per_mat[mid_id]
            prof[i, 1] = a * (mid[i] ** b)

        return prof

    quant_options = {mid: basaltic_QUANTILES for mid in mat_ids}

    candidates = []
    candidates.append(("Q_FILE", profile_original, None))

    quant_iters = [list(quant_options[mid]) for mid in mat_ids]
    for combo in product(*quant_iters):
        quant_per_mat = {mid: combo[i] for i, mid in enumerate(mat_ids)}
        label_parts = [f"M{mid}_{combo[i][0]}" for i, mid in enumerate(mat_ids)]
        cand_label = "__".join(label_parts)
        cand_profile = build_basaltic_profile(
            profile_original, mat_col, mat_ids, quant_per_mat
        )
        candidates.append((cand_label, cand_profile, quant_per_mat))

    print(f"[INFO] Built {len(candidates)} candidate Vs profiles "
          f"(1 original + {len(candidates)-1} combinations).\n")

    # ── Step 2: Helper to run a single candidate through tuning ───────────
    def tune_candidate(profile_cand, label, verbose=False,
                       max_iter=None, do_d_scale=None):
        """Per-material bisection + d_scale sweep on one candidate Vs profile."""
        if max_iter is None:
            max_iter = TUNE_MAX_ITER
        if do_d_scale is None:
            do_d_scale = TUNE_D_SCALE

        offsets = {mid: 0.0 for mid in mat_ids}
        scales  = {mid: 1.0 for mid in mat_ids}
        shifts  = {mid: STRAIN_SHIFTS_MANUAL.get(mid, 1.0) for mid in mat_ids}

        # ── Bisection per material on d_min_offset ────────────────────────
        for mid in mat_ids:
            lo, hi = TUNE_D_MIN_BOUNDS
            for _ in range(max_iter):
                mid_off = 0.5 * (lo + hi)
                offsets[mid] = mid_off
                _, acc_t, _, _, _ = run_eql(
                    profile_cand, strain_base, Gg_base, D_base,
                    mat_ids, mat_col, offsets, scales, shifts,
                    input_motion, BOUNDARY_CONDITION, TARGET_DEPTH,
                    verbose=False, show_fig=False,
                    d_total_cap=D_TOTAL_CAP_PCT,
                )
                _, _, _, ratio = gof_metrics(acc_t, acc_centrifuge)
                if TUNE_RATIO_TARGET[0] <= ratio <= TUNE_RATIO_TARGET[1]:
                    break
                if ratio > TUNE_RATIO_TARGET[1]:
                    lo = mid_off
                else:
                    hi = mid_off

        # ── d_scale sweep (best by RMSE) ──────────────────────────────────
        if do_d_scale:
            for mid in mat_ids:
                best_rmse = np.inf
                best_ds   = scales[mid]
                for ds_val in TUNE_D_SCALE_VALUES:
                    scales[mid] = ds_val
                    _, acc_t, _, _, _ = run_eql(
                        profile_cand, strain_base, Gg_base, D_base,
                        mat_ids, mat_col, offsets, scales, shifts,
                        input_motion, BOUNDARY_CONDITION, TARGET_DEPTH,
                        verbose=False, show_fig=False,
                        d_total_cap=D_TOTAL_CAP_PCT,
                    )
                    rmse, _, _, _ = gof_metrics(acc_t, acc_centrifuge)
                    if rmse < best_rmse:
                        best_rmse = rmse
                        best_ds   = ds_val
                scales[mid] = best_ds

        # ── Final run with chosen levers, compute full GOF ────────────────
        _, acc_final, _, _, _ = run_eql(
            profile_cand, strain_base, Gg_base, D_base,
            mat_ids, mat_col, offsets, scales, shifts,
            input_motion, BOUNDARY_CONDITION, TARGET_DEPTH,
            verbose=False, show_fig=False,
            d_total_cap=D_TOTAL_CAP_PCT,
        )
        rmse, peak_s, peak_c, ratio = gof_metrics(acc_final, acc_centrifuge)

        # TF frequency shift in 0.5-50 Hz band
        f_sim, TF_sim_amp = compute_tf(acc_final, acc_base_ms2, dt)
        f_ct,  TF_ct_amp  = compute_tf(acc_centrifuge, acc_base_ms2, dt)
        band = (f_sim >= 0.5) & (f_sim <= 50.0)
        f_peak_sim = f_sim[band][np.argmax(TF_sim_amp[band])] if band.any() else np.nan
        band_ct = (f_ct >= 0.5) & (f_ct <= 50.0)
        f_peak_ct  = f_ct[band_ct][np.argmax(TF_ct_amp[band_ct])] if band_ct.any() else np.nan
        freq_shift = f_peak_sim - f_peak_ct

        w_ratio, w_rmse, w_freq = GOF_WEIGHTS
        score = (
            w_ratio * abs(ratio - 1.0)
            + w_rmse  * rmse
            + w_freq  * abs(freq_shift)
        )

        vs_per_mat    = {}
        d_min_per_mat = {}
        for mid in mat_ids:
            mask = mat_col == mid
            vs_per_mat[mid] = float(np.mean(profile_cand[mask, 1]))
            d_min_per_mat[mid] = float(np.clip(
                profile_cand[mask, 2][0] + offsets[mid], 0.1, 15.0
            ))

        return {
            "label"          : label,
            "offsets"        : dict(offsets),
            "scales"         : dict(scales),
            "shifts"         : dict(shifts),
            "ratio"          : ratio,
            "rmse"           : rmse,
            "freq_shift"     : freq_shift,
            "score"          : score,
            "vs_per_mat"     : vs_per_mat,
            "d_min_per_mat"  : d_min_per_mat,
        }

    # ── Physical-consistency check ────────────────────────────────────────
    def check_physical_plausibility(res):
        """Return (passed, message). Skip silently when only 1 material exists."""
        if not ENABLE_PHYSICAL_GUARDRAILS:
            return True, "skipped"
        if len(mat_ids) < 2:
            return True, "single material"

        m_top = mat_ids[0]
        m_bot = mat_ids[-1]
        vs_top = res["vs_per_mat"][m_top]
        vs_bot = res["vs_per_mat"][m_bot]
        d_top  = res["d_min_per_mat"][m_top]
        d_bot  = res["d_min_per_mat"][m_bot]

        if vs_bot <= vs_top:
            return False, f"Vs_Mat{m_bot}({vs_bot:.0f})<=Vs_Mat{m_top}({vs_top:.0f})"
        if d_bot >= d_top:
            return False, f"D_Mat{m_bot}({d_bot:.2f})>=D_Mat{m_top}({d_top:.2f})"
        return True, "OK"

    # ── Step 3a: Pre-filter on Vs ordering (zero EQL cost) ────────────────
    prefiltered = []
    if PREFILTER_VS_ORDERING and len(mat_ids) >= 2:
        m_top, m_bot = mat_ids[0], mat_ids[-1]
        skipped_pf = 0
        for k, (label, prof_cand, qmeta) in enumerate(candidates, 1):
            vs_top = float(np.mean(prof_cand[mat_col == m_top, 1]))
            vs_bot = float(np.mean(prof_cand[mat_col == m_bot, 1]))
            if vs_bot > vs_top:
                prefiltered.append((k, label, prof_cand, qmeta))
            else:
                skipped_pf += 1
        print(f"[Pass 0] Vs-ordering prefilter: kept {len(prefiltered)}/"
              f"{len(candidates)}  (skipped {skipped_pf}).\n")
    else:
        prefiltered = [(k, lbl, p, q) for k, (lbl, p, q)
                       in enumerate(candidates, 1)]

    # ── Step 3b: PASS 1 - Fast screen ─────────────────────────────────────
    print(f"[Pass 1] Coarse screen "
          f"({SCREEN_BISECTION_ITER} bisection iter, no d_scale sweep) ...")
    print(f"  {'#':>4}  {'Candidate':<32}  {'ratio':>6}  {'RMSE':>7}  "
          f"{'df[Hz]':>8}  {'score':>8}  PHYS")
    print(f"  {'-'*88}")

    pass1_results = []
    for k, label, prof_cand, _ in prefiltered:
        try:
            res = tune_candidate(prof_cand, label,
                                 max_iter=SCREEN_BISECTION_ITER,
                                 do_d_scale=False)
            phys_ok, phys_msg = check_physical_plausibility(res)
            res["phys_ok"]  = phys_ok
            res["phys_msg"] = phys_msg
            pass1_results.append((k, label, prof_cand, res))
            phys_tag = "OK" if phys_ok else "FAIL"
            print(f"  {k:>4}  {label[:32]:<32}  {res['ratio']:>6.3f}  "
                  f"{res['rmse']:>7.4f}  {res['freq_shift']:>8.2f}  "
                  f"{res['score']:>8.4f}  {phys_tag}")
        except Exception as e:
            print(f"  {k:>4}  {label[:32]:<32}  FAILED: {str(e)[:30]}")

    # ── Step 3c: PASS 2 - Refine the top N candidates ─────────────────────
    pass1_sorted = sorted(pass1_results, key=lambda r: r[3]["score"])
    finalists = [r for r in pass1_sorted
                 if r[3].get("phys_ok", False)][:TOP_N_REFINE]
    if len(finalists) < TOP_N_REFINE:
        for r in pass1_sorted:
            if r not in finalists:
                finalists.append(r)
            if len(finalists) >= TOP_N_REFINE:
                break

    print(f"\n[Pass 2] Full tuning of top {len(finalists)} finalists "
          f"({TUNE_MAX_ITER} bisection iter + d_scale sweep) ...")
    print(f"  {'#':>4}  {'Candidate':<32}  {'ratio':>6}  {'RMSE':>7}  "
          f"{'df[Hz]':>8}  {'score':>8}  {'PHYS':<6}  Reason")
    print(f"  {'-'*98}")

    results = []
    for k, label, prof_cand, _ in finalists:
        try:
            res = tune_candidate(prof_cand, label,
                                 max_iter=TUNE_MAX_ITER,
                                 do_d_scale=TUNE_D_SCALE)
            phys_ok, phys_msg = check_physical_plausibility(res)
            res["phys_ok"]  = phys_ok
            res["phys_msg"] = phys_msg
            results.append((k, label, prof_cand, res))
            phys_tag = "OK" if phys_ok else "FAIL"
            print(f"  {k:>4}  {label[:32]:<32}  {res['ratio']:>6.3f}  "
                  f"{res['rmse']:>7.4f}  {res['freq_shift']:>8.2f}  "
                  f"{res['score']:>8.4f}  {phys_tag:<6}  {phys_msg}")
        except Exception as e:
            print(f"  {k:>4}  {label[:32]:<32}  FAILED: {str(e)[:30]}")

    if not results:
        raise RuntimeError("No candidates completed successfully.")

    # ── Step 4: Leaderboards & winner pick ────────────────────────────────
    def _print_leaderboard(title, ranked, top_k):
        print(f"\n  --- {title} (top {min(top_k, len(ranked))}) ---")
        print(f"    {'rank':>4}  {'#':>4}  {'Candidate':<32}  "
              f"{'ratio':>6}  {'RMSE':>7}  {'df[Hz]':>8}  "
              f"{'score':>8}  PHYS")
        print(f"    {'-'*88}")
        for rank, (k, lbl, _, res) in enumerate(ranked[:top_k], 1):
            ptag = "OK" if res.get("phys_ok") else "FAIL"
            print(f"    {rank:>4}  {k:>4}  {lbl[:32]:<32}  "
                  f"{res['ratio']:>6.3f}  {res['rmse']:>7.4f}  "
                  f"{res['freq_shift']:>8.2f}  {res['score']:>8.4f}  {ptag}")

    ranked_all     = sorted(results,           key=lambda r: r[3]["score"])
    ranked_phys_ok = [r for r in ranked_all if r[3].get("phys_ok", False)]

    print("\n" + "=" * 60)
    print("  SECTION 7.4 LEADERBOARDS")
    print("=" * 60)
    _print_leaderboard("BEST OVERALL (any PHYS)",
                       ranked_all, LEADERBOARD_TOP_K)
    if ranked_phys_ok:
        _print_leaderboard("BEST PHYSICALLY CONSISTENT (PHYS=OK only)",
                           ranked_phys_ok, LEADERBOARD_TOP_K)

    def _flag_near_ties(ranked, label):
        if len(ranked) < 2:
            return
        top_score = ranked[0][3]["score"]
        ties = [r for r in ranked[1:]
                if (r[3]["score"] - top_score) <= NEAR_TIE_THRESHOLD]
        if ties:
            print(f"\n  [NEAR-TIE in {label}]  "
                  f"{len(ties)} candidate(s) within "
                  f"+{NEAR_TIE_THRESHOLD:.3f} of best score "
                  f"({top_score:.4f}):")
            for r in ties:
                _, lbl, _, res = r
                gap = res["score"] - top_score
                print(f"      {lbl[:40]:<40}  score={res['score']:.4f}  "
                      f"(+{gap:.4f})")
            print("      -> Winner choice is sensitive to noise / weights.")

    _flag_near_ties(ranked_all,     "BEST OVERALL")
    if ranked_phys_ok:
        _flag_near_ties(ranked_phys_ok, "BEST PHYS=OK")

    if ranked_phys_ok:
        winner = ranked_phys_ok[0]
        skipped = [r for r in ranked_all
                   if r[3]["score"] < winner[3]["score"]
                   and not r[3].get("phys_ok", False)]
        if skipped:
            print(f"\n  [INFO] {len(skipped)} candidate(s) had a lower (better) "
                  f"score than the chosen winner but failed the physical "
                  f"guardrails:")
            for s in skipped[:LEADERBOARD_TOP_K]:
                _, lbl, _, sres = s
                print(f"      skip  {lbl[:40]:<40}  score={sres['score']:.4f}  "
                      f"({sres['phys_msg']})")
            if len(skipped) > LEADERBOARD_TOP_K:
                print(f"      ... and {len(skipped) - LEADERBOARD_TOP_K} more")
    else:
        winner = ranked_all[0]
        print("\n  " + "!" * 60)
        print("  [WARN] NO CANDIDATE PASSED THE PHYSICAL GUARDRAILS.")
        print("  The simulation will use the highest-GOF candidate, but its")
        print("  Vs / D_min ordering violates basic soil-mechanics expectations.")
        print(f"  Reason: {winner[3].get('phys_msg', '?')}")
        print("  Possible actions:")
        print("    1. Relax GOF_WEIGHTS to favour TF / RMSE over peak ratio.")
        print("    2. Increase D_TOTAL_CAP_PCT (current cap may be too tight).")
        print("    3. Set ENABLE_PHYSICAL_GUARDRAILS = False to silence the check.")
        print("    4. Inspect the leaderboards above to choose a candidate")
        print("       manually (e.g. one with slightly worse score but PHYS=OK).")
        print("  " + "!" * 60)

    winner_k, winner_label, winner_profile, winner_res = winner

    phys_tag = "OK" if winner_res.get("phys_ok") else "FAIL (best-of-bad)"
    print(f"\n  WINNER: #{winner_k} - {winner_label}    [PHYS: {phys_tag}]")
    print(f"    score = {winner_res['score']:.4f}")
    print(f"    ratio = {winner_res['ratio']:.3f}")
    print(f"    RMSE  = {winner_res['rmse']:.5f} m/s^2")
    print(f"    df    = {winner_res['freq_shift']:+.2f} Hz")
    if "vs_per_mat" in winner_res:
        for mid in mat_ids:
            print(f"    Mat{mid}: Vs_avg={winner_res['vs_per_mat'][mid]:6.1f} m/s   "
                  f"D_min={winner_res['d_min_per_mat'][mid]:5.3f} %")

    # Save winning profile to disk
    win_path = os.path.join(
        SAVE_DIR,
        f"{Path(FNAME).stem}__sand_winner__{winner_label}.txt"
    )
    np.savetxt(win_path, winner_profile,
               fmt=["%6.3f", "%7.2f", "%6.4f", "%7.1f", "%2d"],
               header="thk[m]   Vs[m/s]  D[%]  rho[kg/m3]  Mat",
               comments="# ")
    print(f"  Wrote winning Vs profile to:\n    {win_path}")

    # Replace profile_original so Section 7.5 fine-tunes on top of the winner
    profile_original_filebackup = profile_original.copy()
    profile_original = winner_profile

    # Refresh helper columns derived from profile_original
    thk_col      = profile_original[:, 0]
    vs_col       = profile_original[:, 1]
    d_col        = profile_original[:, 2]
    rho_col      = profile_original[:, 3]
    mat_col      = profile_original[:, 4].astype(int)
    depth_tops_p = np.concatenate(([0.0], np.cumsum(thk_col[:-1])))
    depth_bots_p = np.cumsum(thk_col)

    print("\n[INFO] profile_original replaced with Sand winner.")
    print("       Section 7.5 will fine-tune on top of this profile.\n")
else:
    print("\n[INFO] Section 7.4 skipped (ENABLE_SAND_SEARCH = False).")


# ==============================================================================
#  SECTION 7.5 - AUTOMATIC PER-MATERIAL TUNING
#
#  Algorithm
#  ---------
#  For each material (shallowest to deepest):
#    Step A - Bisection on d_min_offset:
#      Binary search between TUNE_D_MIN_BOUNDS until peak_ratio at
#      TARGET_DEPTH is within TUNE_RATIO_TARGET. Other materials keep their
#      previously found d_min_offset. d_scale=1.0 and strain_shift=1.0.
#    Step B - d_scale sweep (if TUNE_D_SCALE = True):
#      With d_min_offset fixed, try each TUNE_D_SCALE_VALUES and pick the
#      value that gives the lowest RMSE.
#
#  After all materials are tuned, produce four plots:
#    A: G/Gmax curves per material - untuned vs tuned
#    B: D curves per material      - untuned vs tuned (with D_total)
#    C: Time history                - reference | untuned | tuned + residual
#    D: Transfer function           - reference | untuned | tuned
# ==============================================================================

print(f"\n{'='*65}")
print(f"  AUTOMATIC PER-MATERIAL TUNING")
print(f"  Materials     : {mat_ids.tolist()}")
print(f"  Target ratio  : {TUNE_RATIO_TARGET[0]} - {TUNE_RATIO_TARGET[1]}")
print(f"  d_min bounds  : {TUNE_D_MIN_BOUNDS} %")
print(f"  Max iter      : {TUNE_MAX_ITER} per material")
print(f"  d_scale sweep : {TUNE_D_SCALE}")
print(f"  strain_shifts : {STRAIN_SHIFTS_MANUAL}")
print(f"{'='*65}\n")

best_d_min_offsets = {mid: 0.0 for mid in mat_ids}
best_d_scales      = {mid: 1.0 for mid in mat_ids}

for mat_target in mat_ids:
    rows_m = np.where(mat_col == mat_target)[0]
    print(f"\n{'-'*65}")
    print(f"  Tuning Material {mat_target}  "
          f"(depth {depth_tops_p[rows_m[0]]:.2f}-"
          f"{depth_bots_p[rows_m[-1]]:.2f} m  "
          f"Vs {vs_col[mat_col==mat_target].min():.0f}-"
          f"{vs_col[mat_col==mat_target].max():.0f} m/s)")
    print(f"{'-'*65}")

    # ── Step A: Bisection on d_min_offset ─────────────────────────────────
    lo, hi      = TUNE_D_MIN_BOUNDS
    best_offset = 0.0
    converged   = False

    print(f"\n  Step A - Bisection on d_min_offset")
    print(f"  {'Iter':>4}  {'d_min_offset[%]':>16}  "
          f"{'peak_ratio':>12}  {'RMSE':>10}  Status")
    print(f"  {'-'*58}")

    for iteration in range(TUNE_MAX_ITER):
        mid_val = (lo + hi) / 2.0

        trial_offsets = {**best_d_min_offsets, mat_target: mid_val}
        trial_scales  = {**best_d_scales}
        trial_shifts  = STRAIN_SHIFTS_MANUAL.copy()

        _, acc_iter, _, _, _ = run_eql(
            profile_original, strain_base, Gg_base, D_base,
            mat_ids, mat_col,
            d_min_offsets=trial_offsets,
            d_scales=trial_scales,
            strain_shifts=trial_shifts,
            input_motion=input_motion,
            boundary=BOUNDARY_CONDITION,
            target_depth=TARGET_DEPTH,
            verbose=False,
            show_fig=False,
        )

        rmse_i, _, _, ratio_i = gof_metrics(acc_iter, acc_centrifuge)

        in_range = TUNE_RATIO_TARGET[0] <= ratio_i <= TUNE_RATIO_TARGET[1]
        if in_range:
            status = "OK converged"
        elif ratio_i > TUNE_RATIO_TARGET[1]:
            status = "raise D (too high)"
        else:
            status = "lower D (too low)"

        print(f"  {iteration+1:>4}  {mid_val:>+16.4f}  "
              f"{ratio_i:>12.4f}  {rmse_i:>10.5f}  {status}")

        if in_range:
            best_offset = mid_val
            converged   = True
            break

        if ratio_i > TUNE_RATIO_TARGET[1]:
            lo = mid_val
        else:
            hi = mid_val

    if not converged:
        best_offset = (lo + hi) / 2.0
        print(f"  [WARN] Did not converge in {TUNE_MAX_ITER} iterations. "
              f"Using d_min_offset = {best_offset:+.4f} %")
    else:
        print(f"  OK Converged: d_min_offset = {best_offset:+.4f} %")

    best_d_min_offsets[mat_target] = best_offset

    # ── Step B: d_scale sweep ─────────────────────────────────────────────
    if TUNE_D_SCALE:
        print(f"\n  Step B - d_scale sweep  {TUNE_D_SCALE_VALUES}")
        print(f"  {'d_scale':>10}  {'peak_ratio':>12}  {'RMSE':>10}  Note")
        print(f"  {'-'*45}")

        best_rmse  = np.inf
        best_scale = 1.0
        sweep_offsets = best_d_min_offsets.copy()

        for ds_val in TUNE_D_SCALE_VALUES:
            sweep_scales = {**best_d_scales, mat_target: ds_val}
            sweep_shifts = STRAIN_SHIFTS_MANUAL.copy()

            _, acc_s, _, _, _ = run_eql(
                profile_original, strain_base, Gg_base, D_base,
                mat_ids, mat_col,
                d_min_offsets=sweep_offsets,
                d_scales=sweep_scales,
                strain_shifts=sweep_shifts,
                input_motion=input_motion,
                boundary=BOUNDARY_CONDITION,
                target_depth=TARGET_DEPTH,
                verbose=False,
                show_fig=False,
            )
            rmse_s, _, _, ratio_s = gof_metrics(acc_s, acc_centrifuge)
            note = "<-- best" if rmse_s < best_rmse else ""
            print(f"  {ds_val:>10.2f}  {ratio_s:>12.4f}  "
                  f"{rmse_s:>10.5f}  {note}")

            if rmse_s < best_rmse:
                best_rmse  = rmse_s
                best_scale = ds_val

        best_d_scales[mat_target] = best_scale
        print(f"  OK Best d_scale = {best_scale:.2f}  "
              f"(RMSE = {best_rmse:.5f} m/s^2)")

# ── Final tuned run with all materials' best parameters ──────────────────
print(f"\n{'='*65}")
print(f"  FINAL TUNED RUN  (all materials combined)")
print(f"{'='*65}")
for mid in mat_ids:
    d_o = float(profile_original[mat_col == mid, 2][0])
    d_t = float(np.clip(d_o + best_d_min_offsets[mid], 0.1, 15.0))
    print(f"  Material {mid}:  "
          f"D_min {d_o:.3f}->{d_t:.3f}%   "
          f"d_scale={best_d_scales[mid]:.2f}   "
          f"strain_shift={STRAIN_SHIFTS_MANUAL[mid]:.2f}")

(time_tuned, acc_tuned, depth_tuned,
 sim_tuned, curve_arrs_tuned) = run_eql(
    profile_original, strain_base, Gg_base, D_base,
    mat_ids, mat_col,
    d_min_offsets=best_d_min_offsets,
    d_scales=best_d_scales,
    strain_shifts=STRAIN_SHIFTS_MANUAL,
    input_motion=input_motion,
    boundary=BOUNDARY_CONDITION,
    target_depth=TARGET_DEPTH,
    verbose=False,
    show_fig=False,
)
print(f"[TUNING] Final extracted depth: {depth_tuned:.3f} m")

# ── Goodness-of-fit ──────────────────────────────────────────────────────
rmse_un,  peak_un,  peak_ct, ratio_un  = gof_metrics(acc_untuned,
                                                     acc_centrifuge)
rmse_tun, peak_tun, _,       ratio_tun = gof_metrics(acc_tuned,
                                                     acc_centrifuge)

print(f"\n{'='*65}")
print(f"  FINAL GOF  |  depth approx {depth_tuned:.2f} m  |  {SENSOR_NAME}")
print(f"{'='*65}")
print(f"  {'Metric':<28}  {'Untuned':>10}  {'Tuned':>10}  Target")
print(f"  {'-'*58}")
print(f"  {'RMSE (m/s^2)':<28}  {rmse_un:>10.5f}  {rmse_tun:>10.5f}  minimize")
print(f"  {'Peak sim (m/s^2)':<28}  {peak_un:>10.5f}  {peak_tun:>10.5f}  "
      f"{peak_ct:.5f}")
print(f"  {'Peak ratio':<28}  {ratio_un:>10.4f}  {ratio_tun:>10.4f}  1.0000")
print(f"{'='*65}")

# colour palette - one colour per material
mat_colors = ["steelblue", "darkorange", "green", "purple"]

# ─────────────────────────────────────────────────────────────────────────
#  PLOT A - G/Gmax curves per material: untuned vs tuned
# ─────────────────────────────────────────────────────────────────────────
if SHOW_FIGURES:
    fig_g, axes_g = plt.subplots(1, n_mat, figsize=(6 * n_mat, 5),
                                 sharey=True)
    if n_mat == 1:
        axes_g = [axes_g]
    fig_g.suptitle(
        f"G/Gmax Curves - Untuned vs Tuned  |  {FNAME}",
        fontsize=12,
    )

    for i, mid in enumerate(mat_ids):
        rows_m = np.where(mat_col == mid)[0]
        col    = mat_colors[i % len(mat_colors)]
        ca_u   = curve_arrs_untuned[mid]
        ca_t   = curve_arrs_tuned[mid]

        ax = axes_g[i]
        ax.semilogx(ca_u['strain'], ca_u['Gg'], color="grey",
                    lw=2.0, ls="--", label="Untuned")
        ax.semilogx(ca_t['strain'], ca_t['Gg'], color=col,
                    lw=2.0, label="Tuned")
        ax.axvspan(0.003, 0.03, alpha=0.10, color="red",
                   label="Operating range")
        ax.set_xlabel("Shear strain (%)")
        ax.set_ylabel("G/Gmax  [-]")
        ax.set_ylim(0, 1.05)
        ax.set_title(
            f"Material {mid}  |  "
            f"Vs {vs_col[mat_col==mid].min():.0f}-"
            f"{vs_col[mat_col==mid].max():.0f} m/s\n"
            f"depth {depth_tops_p[rows_m[0]]:.1f}-"
            f"{depth_bots_p[rows_m[-1]]:.1f} m  |  "
            f"strain_shift={STRAIN_SHIFTS_MANUAL[mid]:.2f}",
            fontsize=9,
        )
        ax.grid(True, which="both", ls="--", alpha=0.4)
        ax.legend(fontsize=8, frameon=False)

    plt.tight_layout()
    plt.show()

    # ─────────────────────────────────────────────────────────────────────
    #  PLOT B - D curves per material: untuned vs tuned (with D_total)
    # ─────────────────────────────────────────────────────────────────────
    fig_d, axes_d = plt.subplots(1, n_mat, figsize=(6 * n_mat, 5),
                                 sharey=False)
    if n_mat == 1:
        axes_d = [axes_d]
    fig_d.suptitle(
        f"Damping Curves - Untuned vs Tuned  |  {FNAME}\n"
        f"D_total = D_min + D_curve  (what EQL actually uses)",
        fontsize=11,
    )

    for i, mid in enumerate(mat_ids):
        rows_m = np.where(mat_col == mid)[0]
        col    = mat_colors[i % len(mat_colors)]
        ca_u   = curve_arrs_untuned[mid]
        ca_t   = curve_arrs_tuned[mid]

        ax = axes_d[i]

        ax.semilogx(ca_u['strain'], ca_u['D_curve'], color="grey",
                    lw=1.5, ls="--", label="D_curve untuned")
        ax.semilogx(ca_u['strain'], ca_u['D_total'], color="grey",
                    lw=2.5, ls="-.",
                    label=f"D_total untuned (D_min={ca_u['d_min']:.2f}%)")
        ax.axhline(ca_u['d_min'], color="grey", lw=1.0, ls=":",
                   alpha=0.7, label=f"D_min untuned={ca_u['d_min']:.2f}%")

        ax.semilogx(ca_t['strain'], ca_t['D_curve'], color=col,
                    lw=1.5,
                    label=f"D_curve tuned (scale={best_d_scales[mid]:.2f})")
        ax.semilogx(ca_t['strain'], ca_t['D_total'], color=col,
                    lw=2.5, ls="-.",
                    label=f"D_total tuned (D_min={ca_t['d_min']:.2f}%)")
        ax.axhline(ca_t['d_min'], color=col, lw=1.0, ls=":",
                   alpha=0.8, label=f"D_min tuned={ca_t['d_min']:.2f}%")

        ax.axvspan(0.003, 0.03, alpha=0.10, color="red",
                   label="Operating range")

        ax.set_xlabel("Shear strain (%)")
        ax.set_ylabel("Damping  [%]")
        ax.set_title(
            f"Material {mid}  |  "
            f"Vs {vs_col[mat_col==mid].min():.0f}-"
            f"{vs_col[mat_col==mid].max():.0f} m/s\n"
            f"depth {depth_tops_p[rows_m[0]]:.1f}-"
            f"{depth_bots_p[rows_m[-1]]:.1f} m  |  "
            f"d_min_offset={best_d_min_offsets[mid]:+.3f}%  "
            f"d_scale={best_d_scales[mid]:.2f}",
            fontsize=9,
        )
        ax.grid(True, which="both", ls="--", alpha=0.4)
        ax.legend(fontsize=7, frameon=False, loc="upper left")

    plt.tight_layout()
    plt.show()

    # ─────────────────────────────────────────────────────────────────────
    #  PLOT C - Time history: reference | untuned | tuned  +  residual
    # ─────────────────────────────────────────────────────────────────────
    tuning_label = "  ".join([
        f"Mat{mid}[Doff={best_d_min_offsets[mid]:+.2f}%  "
        f"Dsc={best_d_scales[mid]:.2f}]"
        for mid in mat_ids
    ])

    fig_t, axes_t = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig_t.suptitle(
        f"Time History Comparison  |  {FNAME}  |  depth approx {TARGET_DEPTH} m\n"
        f"{tuning_label}",
        fontsize=10,
    )

    ax = axes_t[0]
    ax.plot(time_zero, acc_centrifuge, color="blue", lw=1.5,
            ls="--", alpha=0.6, label=f"Reference ({SENSOR_NAME})")
    ax.plot(time_untuned, acc_untuned, color="grey", lw=0.9, alpha=0.9,
            label=f"Untuned  (ratio={ratio_un:.3f}  RMSE={rmse_un:.4f})")
    ax.plot(time_tuned,   acc_tuned,   color="red",  lw=1.0,
            label=f"Tuned    (ratio={ratio_tun:.3f}  RMSE={rmse_tun:.4f})")
    ax.set_ylabel("Acceleration (m/s^2)")
    ax.set_title("Time History")
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    ax.grid(True, ls="--", alpha=0.4)

    n_min    = min(len(time_tuned), len(time_zero),
                   len(acc_tuned),  len(acc_centrifuge))
    residual = acc_tuned[:n_min] - acc_centrifuge[:n_min]

    ax = axes_t[1]
    ax.plot(time_tuned[:n_min], residual, color="darkgreen", lw=0.8)
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.fill_between(time_tuned[:n_min], residual, alpha=0.25, color="green")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Residual (m/s^2)\nTuned - Reference")
    ax.set_title("Residual  (zero = perfect match)")
    ax.grid(True, ls="--", alpha=0.4)

    plt.tight_layout()
    plt.show()

# ─────────────────────────────────────────────────────────────────────────
#  PLOT D - Transfer function: reference | untuned | tuned
#  (TF computed unconditionally - needed by Section 9 report)
# ─────────────────────────────────────────────────────────────────────────
freqs_un,  TF_un  = compute_tf(acc_untuned,    acc_base_ms2, dt)
freqs_tun, TF_tun = compute_tf(acc_tuned,      acc_base_ms2, dt)
freqs_ct,  TF_ct  = compute_tf(acc_centrifuge, acc_base_ms2, dt)

if SHOW_FIGURES:
    fig_f, ax = plt.subplots(figsize=(10, 4))
    ax.semilogy(freqs_ct,  TF_ct,  color="blue",  lw=1.5, ls="--",
                alpha=0.7, label=f"Reference ({SENSOR_NAME})")
    ax.semilogy(freqs_un,  TF_un,  color="grey",  lw=0.9, alpha=0.9,
                label=f"Untuned  (ratio={ratio_un:.3f})")
    ax.semilogy(freqs_tun, TF_tun, color="red",   lw=1.2,
                label=f"Tuned    (ratio={ratio_tun:.3f})")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("|H(f)|  -  Amplification")
    ax.set_title(
        f"Transfer Function  |  {FNAME}  |  depth approx {depth_tuned:.2f} m"
    )
    ax.set_xlim(0, 60)
    ax.set_ylim(1e-2, 1e2)
    ax.grid(True, which="both", ls="--", alpha=0.4)
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.show()

# ── Summary ──────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  TUNING COMPLETE")
print("=" * 65)
for mid in mat_ids:
    d_o = float(profile_original[mat_col == mid, 2][0])
    d_t = float(np.clip(d_o + best_d_min_offsets[mid], 0.1, 15.0))
    print(f"  Mat {mid}:  D_min {d_o:.3f}->{d_t:.3f}%   "
          f"d_scale={best_d_scales[mid]:.2f}   "
          f"strain_shift={STRAIN_SHIFTS_MANUAL[mid]:.2f}")
print(f"  Untuned : ratio={ratio_un:.3f}  RMSE={rmse_un:.5f}")
print(f"  Tuned   : ratio={ratio_tun:.3f}  RMSE={rmse_tun:.5f}")
if abs(ratio_tun - 1.0) < 0.05:
    print("  Amplitude match: GOOD")
else:
    print("  Amplitude match: CHECK - ratio outside 0.95-1.05")
print("  If TF peak FREQUENCY is shifted, set STRAIN_SHIFTS_MANUAL")
print("  to values < 1.0 for the relevant material and re-run.")
print("=" * 65)


# ==============================================================================
#  SECTION 9 - TUNING REPORT  (written automatically to SAVE_DIR)
# ==============================================================================

date_stamp_rep = datetime.now().strftime("%Y-%m-%d_%H-%M")
report_name    = (f"{Path(FNAME).stem}"
                  f"__tuning_report__EQL__{date_stamp_rep}.txt")
report_path    = os.path.join(SAVE_DIR, report_name)

# Pull tuned simulation result arrays
mss_t        = sim_tuned.max_strain_stress
depth_mid_t  = mss_t[:, 0]
max_strain_t = mss_t[:, 1] * 100.0    # -> %
max_stress_t = mss_t[:, 2] / 1000.0   # -> kPa

# Cumulative depths for soil profile table
n_fin_p      = int(np.sum(thk_col > 0))
depth_tops_t = np.concatenate(([0.0], np.cumsum(thk_col[:n_fin_p])))
depth_mids_t = (depth_tops_t[:-1] + depth_tops_t[1:]) / 2.0
depth_bots_t = depth_tops_t[1:]

# Cross-correlation for residual characterisation
n_cc     = min(len(acc_tuned), len(acc_centrifuge))
a_s      = acc_tuned[:n_cc];   a_c = acc_centrifuge[:n_cc]
cc       = np.correlate(a_s - a_s.mean(), a_c - a_c.mean(), mode="full")
cc      /= (np.std(a_s) * np.std(a_c) * n_cc + 1e-12)
lags     = np.arange(-(n_cc - 1), n_cc)
lag_smp  = int(lags[int(np.argmax(cc))])
lag_sec  = lag_smp * dt
cc_val   = float(np.max(cc))
res_r    = a_s - a_c
res_mean = float(np.mean(res_r));  res_std = float(np.std(res_r))
if abs(lag_sec) > 2 * dt:
    res_char = f"phase lag ({lag_sec:+.4f} s = {lag_smp:+d} samples)"
elif abs(res_mean) > 0.1 * res_std:
    res_char = f"systematic bias (mean={res_mean:+.5f} m/s^2)"
else:
    res_char = "approximately random (mean approx 0)"

# TF metrics
fm = (freqs_tun >= 0.5) & (freqs_tun <= 50.0)
if fm.sum() > 0:
    tf_diff   = np.abs(TF_tun[fm] - TF_ct[fm])
    tf_rms    = float(np.sqrt(np.mean(tf_diff**2)))
    worst_f   = float(freqs_tun[fm][int(np.argmax(tf_diff))])
    f_ct      = float(freqs_ct[fm][int(np.argmax(TF_ct[fm]))])
    f_sim     = float(freqs_tun[fm][int(np.argmax(TF_tun[fm]))])
    freq_shft = f_sim - f_ct
else:
    tf_rms = worst_f = f_ct = f_sim = freq_shft = float("nan")

SEP = "=" * 80;  sep = "-" * 80

with open(report_path, "w", encoding="utf-8") as f:

    f.write(SEP + "\n")
    f.write("  SRA TUNING REPORT - PySeismoSoil  v01\n")
    f.write(SEP + "\n\n")

    f.write("  This file documents one tuned EQL site response analysis run.\n")
    f.write("  Sections below cover: run metadata, tuning parameters\n")
    f.write("  found per material, goodness-of-fit metrics, the tuned\n")
    f.write("  soil profile, strain regime per layer, and the tuned\n")
    f.write("  G/Gmax and D curves used by EQL.\n\n")
    f.write("  Units:\n")
    f.write("    depth, thickness         m (prototype scale)\n")
    f.write("    Vs                       m/s\n")
    f.write("    rho                      kg/m^3\n")
    f.write("    D_min, D_curve, D_total  %  (NOT decimal)\n")
    f.write("    strain                   %  (NOT decimal)\n")
    f.write("    acceleration             m/s^2 (prototype scale)\n")
    f.write("    stress                   kPa\n")
    f.write("    frequency                Hz\n\n")

    f.write(sep + "\n  BLOCK 1 - RUN IDENTIFICATION\n" + sep + "\n")
    f.write(f"  Date / time           : "
            f"{datetime.now().strftime('%Y-%m-%d  %H:%M')}\n")
    f.write(f"  Input motion file     : {FNAME}\n")
    f.write(f"  Time slice (samples)  : {SLICE_START} - {SLICE_END}\n")
    f.write(f"  Scale N    : {N_SCALE}  (prototype/model)\n")
    f.write(f"  Time step dt          : {dt:.6f}  s\n")
    f.write(f"  Simulation method     : EQL\n")
    f.write(f"  Boundary condition    : {BOUNDARY_CONDITION}  "
            f"(rigid = borehole input)\n")
    f.write(f"  Curves source         : {CURVES_PATH}\n")
    f.write(f"  Vs profile file       : {VS_PROFILE_PATH}\n")
    f.write(f"  Reference sensor      : {SENSOR_NAME}\n")
    f.write(f"  Target depth          : {TARGET_DEPTH:.2f}  m\n")
    f.write(f"  Actual depth          : {depth_tuned:.3f}  m  "
            f"(nearest available layer)\n")
    f.write(f"  Output directory      : {SAVE_DIR}\n")
    sand_used = bool(_sand_active)
    f.write(f"  Sand search used   : "
            f"{'yes' if sand_used else 'no'}\n")
    if sand_used:
        f.write(f"  D_total cap (PCT)     : {D_TOTAL_CAP_PCT:.2f}  %\n")
    f.write("\n")

    f.write(sep + "\n  BLOCK 2 - AUTOMATIC TUNING RESULTS PER MATERIAL\n"
            + sep + "\n")
    f.write(f"  Algorithm : bisection on d_min_offset, then d_scale sweep\n")
    f.write(f"  Target peak ratio : {TUNE_RATIO_TARGET}\n\n")
    f.write(f"  {'Mat':>4}  {'Depth[m]':>14}  {'Vs[m/s]':>10}  "
            f"{'D_min_orig[%]':>13}  {'d_min_offset[%]':>16}  "
            f"{'D_min_tuned[%]':>15}  {'d_scale':>8}  "
            f"{'strain_shift':>12}\n")
    f.write(f"  {'-'*105}\n")
    for mid in mat_ids:
        mask   = mat_col == mid
        rows_m = np.where(mask)[0]
        d_o    = float(profile_original[mask, 2][0])
        d_t    = float(np.clip(d_o + best_d_min_offsets[mid], 0.1, 15.0))
        dep    = (f"{depth_tops_p[rows_m[0]]:.2f}-"
                  f"{depth_bots_p[rows_m[-1]]:.2f}")
        vs_rng = (f"{vs_col[mask].min():.0f}-{vs_col[mask].max():.0f}")
        f.write(f"  {mid:>4}  {dep:>14}  {vs_rng:>10}  "
                f"{d_o:>13.4f}  {best_d_min_offsets[mid]:>+16.4f}  "
                f"{d_t:>15.4f}  {best_d_scales[mid]:>8.3f}  "
                f"{STRAIN_SHIFTS_MANUAL[mid]:>12.3f}\n")
    f.write("\n")

    f.write(sep + "\n  BLOCK 3 - GOODNESS-OF-FIT\n" + sep + "\n")
    f.write(f"  Comparison depth: {depth_tuned:.3f} m  |  "
            f"Sensor: {SENSOR_NAME}\n\n")
    f.write(f"  {'Metric':<40}  {'Untuned':>10}  {'Tuned':>10}  "
            f"Target / Units\n")
    f.write(f"  {'-'*75}\n")
    f.write(f"  {'RMSE':<40}  {rmse_un:>10.5f}  {rmse_tun:>10.5f}  "
            f"m/s^2  (minimize)\n")
    f.write(f"  {'Peak sim':<40}  {peak_un:>10.5f}  {peak_tun:>10.5f}  "
            f"m/s^2\n")
    f.write(f"  {'Peak reference':<40}  {'--':>10}  {peak_ct:>10.5f}  "
            f"m/s^2\n")
    f.write(f"  {'Peak ratio (sim/reference)':<40}  {ratio_un:>10.4f}  "
            f"{ratio_tun:>10.4f}  [-]  target=1.000\n")
    f.write(f"  {'Cross-correlation coefficient':<40}  {'--':>10}  "
            f"{cc_val:>10.4f}  [-]  (1.0=perfect)\n")
    f.write(f"  {'Cross-correlation lag':<40}  {'--':>10}  "
            f"{lag_sec:>+10.4f}  s\n")
    f.write(f"  {'Residual character':<40}  {res_char}\n")
    f.write(f"  {'TF RMS error (0.5-50 Hz)':<40}  {'--':>10}  "
            f"{tf_rms:>10.4f}  [-]\n")
    f.write(f"  {'Fundamental freq - reference':<40}  {'--':>10}  "
            f"{f_ct:>10.2f}  Hz\n")
    f.write(f"  {'Fundamental freq - tuned sim':<40}  {'--':>10}  "
            f"{f_sim:>10.2f}  Hz\n")
    f.write(f"  {'Frequency shift (sim - reference)':<40}  {'--':>10}  "
            f"{freq_shft:>+10.2f}  Hz\n\n")

    f.write(sep + "\n  BLOCK 4 - SOIL PROFILE AFTER TUNING\n" + sep + "\n")
    f.write("  D_min_tuned = D_min_orig + d_min_offset  "
            "(clipped to [0.1, 15] %)\n")
    f.write("  Vs, rho, thickness unchanged.\n\n")
    f.write(f"  {'#':>4}  {'Top[m]':>7}  {'Mid[m]':>7}  {'Bot[m]':>7}  "
            f"{'Thk[m]':>7}  {'Vs[m/s]':>8}  {'D_orig[%]':>10}  "
            f"{'D_tuned[%]':>11}  {'rho[kg/m3]':>11}  {'Mat':>4}\n")
    f.write(f"  {'-'*97}\n")
    for i in range(n_fin_p):
        mid_i = int(mat_col[i])
        d_o_i = float(d_col[i])
        d_t_i = float(np.clip(d_o_i + best_d_min_offsets.get(mid_i, 0.0),
                              0.1, 15.0))
        f.write(f"  {i+1:>4}  {depth_tops_t[i]:>7.3f}  "
                f"{depth_mids_t[i]:>7.3f}  {depth_bots_t[i]:>7.3f}  "
                f"{thk_col[i]:>7.3f}  {vs_col[i]:>8.1f}  {d_o_i:>10.4f}  "
                f"{d_t_i:>11.4f}  {rho_col[i]:>11.1f}  {mid_i:>4}\n")
    f.write("\n")

    f.write(sep + "\n  BLOCK 5 - STRAIN REGIME PER LAYER (TUNED)\n" + sep + "\n")
    f.write(f"  {'#':>4}  {'Mid[m]':>8}  {'gamma_max[%]':>13}  "
            f"{'tau_max[kPa]':>13}  Regime\n")
    f.write(f"  {'-'*65}\n")
    for i, (d, s, tau) in enumerate(zip(depth_mid_t,
                                         max_strain_t,
                                         max_stress_t)):
        f.write(f"  {i+1:>4}  {d:>8.3f}  {s:>13.5f}  "
                f"{tau:>13.3f}  {strain_regime(s)}\n")
    f.write("\n")

    f.write(sep + "\n  BLOCK 6 - TUNED G/Gmax AND D CURVES\n" + sep + "\n")
    f.write("  These are the actual curves PySeismoSoil used during EQL.\n")
    f.write("  D_total = D_min + D_curve  (this is what damps each EQL iteration).\n\n")

    for mid in mat_ids:
        ca = curve_arrs_tuned[mid]
        f.write(f"  Material {mid}  -  D_min={ca['d_min']:.3f}%  "
                f"d_scale={best_d_scales[mid]:.3f}  "
                f"strain_shift={STRAIN_SHIFTS_MANUAL[mid]:.3f}\n")
        f.write(f"  {'Strain[%]':>14}  {'G/Gmax[-]':>12}  "
                f"{'D_curve[%]':>12}  {'D_total[%]':>12}\n")
        f.write(f"  {'-'*56}\n")
        for s, g, d, dtot in zip(ca['strain'], ca['Gg'],
                                  ca['D_curve'], ca['D_total']):
            f.write(f"  {s:>14.6f}  {g:>12.6f}  "
                    f"{d:>12.4f}  {dtot:>12.4f}\n")
        f.write("\n")

    f.write(sep + "\n  BLOCK 7 - AUTOMATED OBSERVATIONS\n" + sep + "\n")
    f.write(f"  Peak ratio (tuned)    : {ratio_tun:.4f}  (target 1.000)\n")
    f.write(f"  RMSE (tuned)          : {rmse_tun:.5f}  m/s^2\n")
    f.write(f"  Cross-correlation     : {cc_val:.4f}\n")
    f.write(f"  Residual character    : {res_char}\n")
    f.write(f"  TF RMS error          : {tf_rms:.4f}\n")
    f.write(f"  Frequency shift       : {freq_shft:+.2f}  Hz\n\n")

    if abs(freq_shft) > 1.0 and not np.isnan(freq_shft):
        f.write(f"  [ACTION] TF frequency shifted {freq_shft:+.2f} Hz.\n")
        f.write(f"  Set STRAIN_SHIFTS_MANUAL for the relevant material\n")
        f.write(f"  to < 1.0 (shift left) or > 1.0 (shift right) and re-run.\n\n")
    else:
        f.write(f"  [OK] Frequency match is acceptable.\n\n")

    f.write("  [MANUAL] Visual inspection notes:\n")
    f.write("  Overall time history appearance  :\n")
    f.write("  Phase alignment                  :\n")
    f.write("  TF shape quality                 :\n")
    f.write("  Free notes                       :\n\n")
    f.write(SEP + "\n  END OF REPORT\n" + SEP + "\n")

print(f"\n[INFO] Tuning report written to:\n       {report_path}")


# ==============================================================================
#  SECTION 10 - TUNED SRA RESULTS: DISPLAY
#
#  Controlled by:
#    SHOW_FIGURES = True   ->  all plots displayed on screen
# ==============================================================================

print(f"\n{'='*65}")
print(f"  SECTION 10 - TUNED SRA RESULTS")
print(f"  SHOW_FIGURES    = {SHOW_FIGURES}")
print(f"{'='*65}")

tune_str = "  |  ".join([
    f"Mat{mid}: D_min"
    f"={float(np.clip(d_col[mat_col==mid][0] + best_d_min_offsets[mid], 0.1, 15)):.2f}%"
    f"  d_sc={best_d_scales[mid]:.2f}"
    for mid in mat_ids
])

# ─────────────────────────────────────────────────────────────────────────
#  10a - PySeismoSoil built-in 5-panel depth profile plot
# ─────────────────────────────────────────────────────────────────────────
print("\n[10a] Built-in depth-profile plot (tuned simulation) ...")
if SHOW_FIGURES:
    sim_tuned.plot(dpi=150, save_fig=False)

# ─────────────────────────────────────────────────────────────────────────
#  10b - Surface acceleration: tuned sim vs reference
# ─────────────────────────────────────────────────────────────────────────
print("\n[10b] Surface acceleration time history ...")
surf_motion = sim_tuned.accel_on_surface
surf_arr    = surf_motion.accel

if SHOW_FIGURES:
    fig_b, ax_b = plt.subplots(figsize=(11, 4))
    ax_b.plot(surf_arr[:, 0], surf_arr[:, 1],
              color="red", lw=0.8, label="Tuned sim - surface")
    ax_b.plot(time_zero, acc_centrifuge,
              color="blue", lw=1.5, ls="--", alpha=0.5,
              label=f"Reference ({SENSOR_NAME})")
    ax_b.set_xlabel("Time (s)")
    ax_b.set_ylabel("Acceleration (m/s^2)")
    ax_b.set_title(
        f"Surface Acceleration - Tuned EQL  |  {FNAME}\n{tune_str}"
    )
    ax_b.grid(True, ls="--", alpha=0.4)
    ax_b.legend(frameon=False, fontsize=9)
    plt.tight_layout()
    plt.show()

# ─────────────────────────────────────────────────────────────────────────
#  10c - Acceleration at TARGET_DEPTH: tuned sim vs reference
# ─────────────────────────────────────────────────────────────────────────
print(f"\n[10c] Acceleration at target depth {TARGET_DEPTH} m ...")
if SHOW_FIGURES:
    fig_c, axes_c = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    fig_c.suptitle(
        f"Tuned EQL  |  {FNAME}  |  depth approx {depth_tuned:.3f} m\n"
        f"{tune_str}\n"
        f"peak_ratio={ratio_tun:.3f}   RMSE={rmse_tun:.5f} m/s^2",
        fontsize=10,
    )

    ax = axes_c[0]
    ax.plot(time_zero, acc_centrifuge,
            color="blue", lw=1.5, ls="--", alpha=0.6,
            label=f"Reference ({SENSOR_NAME})")
    ax.plot(time_tuned, acc_tuned,
            color="red", lw=1.0,
            label=f"Tuned sim  "
                  f"(ratio={ratio_tun:.3f}  RMSE={rmse_tun:.5f})")
    ax.set_ylabel("Acceleration (m/s^2)")
    ax.set_title(f"Time History at depth approx {depth_tuned:.3f} m")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(True, ls="--", alpha=0.4)

    n_min_c    = min(len(time_tuned), len(time_zero),
                     len(acc_tuned),  len(acc_centrifuge))
    residual_c = acc_tuned[:n_min_c] - acc_centrifuge[:n_min_c]

    ax = axes_c[1]
    ax.plot(time_tuned[:n_min_c], residual_c, color="darkgreen", lw=0.8)
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.fill_between(time_tuned[:n_min_c], residual_c,
                    alpha=0.25, color="green")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Residual (m/s^2)\nTuned - Reference")
    ax.set_title("Residual  (zero = perfect match)")
    ax.grid(True, ls="--", alpha=0.4)

    plt.tight_layout()
    plt.show()

# ─────────────────────────────────────────────────────────────────────────
#  10d - Transfer function: tuned sim vs reference
# ─────────────────────────────────────────────────────────────────────────
print("\n[10d] Transfer function (tuned simulation) ...")
freqs_tun10, TF_tun10 = compute_tf(acc_tuned,      acc_base_ms2, dt)
freqs_ct10,  TF_ct10  = compute_tf(acc_centrifuge,  acc_base_ms2, dt)

if SHOW_FIGURES:
    fig_d10, ax_d10 = plt.subplots(figsize=(10, 4))
    ax_d10.semilogy(freqs_ct10,  TF_ct10,
                    color="blue", lw=1.5, ls="--", alpha=0.7,
                    label=f"Reference ({SENSOR_NAME})")
    ax_d10.semilogy(freqs_tun10, TF_tun10,
                    color="red",  lw=1.2,
                    label=f"Tuned sim  (ratio={ratio_tun:.3f})")
    ax_d10.set_xlabel("Frequency (Hz)")
    ax_d10.set_ylabel("|H(f)|  -  Amplification")
    ax_d10.set_title(
        f"Transfer Function - Tuned EQL  |  {FNAME}  "
        f"|  depth approx {depth_tuned:.2f} m\n{tune_str}"
    )
    ax_d10.set_xlim(0, 60)
    ax_d10.set_ylim(1e-2, 1e2)
    ax_d10.grid(True, which="both", ls="--", alpha=0.4)
    ax_d10.legend(frameon=False)
    plt.tight_layout()
    plt.show()

# ─────────────────────────────────────────────────────────────────────────
#  10e - Peak acceleration depth profile (tuned simulation)
# ─────────────────────────────────────────────────────────────────────────
print("\n[10e] Peak acceleration depth profile (tuned simulation) ...")

mavd_t     = sim_tuned.max_a_v_d
depth_t_e  = mavd_t[:, 0]
max_acc_t  = mavd_t[:, 1]
max_vel_t  = mavd_t[:, 2]
max_dsp_t  = mavd_t[:, 3] * 100.0     # -> cm

mss_t10        = sim_tuned.max_strain_stress
depth_t_mid    = mss_t10[:, 0]
max_strain_t10 = mss_t10[:, 1] * 100.0
max_stress_t10 = mss_t10[:, 2] / 1000.0

if SHOW_FIGURES:
    fig_e, axes_e = plt.subplots(1, 4, figsize=(16, 7), sharey=True)
    fig_e.suptitle(
        f"Tuned EQL - Depth Profiles  |  {FNAME}\n{tune_str}",
        fontsize=11,
    )

    ax = axes_e[0]
    ax.plot(max_acc_t, depth_t_e, "s-", color="crimson", lw=1.5, ms=5)
    ax.axhline(TARGET_DEPTH, color="blue", lw=1.0, ls=":",
               label=f"Target depth {TARGET_DEPTH} m")
    ax.set_xlabel("Peak accel. (m/s^2)")
    ax.set_ylabel("Depth (m)")
    ax.grid(True, ls="--", alpha=0.4)
    ax.set_title("Peak Acceleration")
    ax.legend(fontsize=7, frameon=False)

    ax = axes_e[1]
    ax.plot(max_strain_t10, depth_t_mid, "o-", color="darkorange",
            lw=1.5, ms=5)
    ax.axvspan(0.0,  0.01, alpha=0.10, color="green",  label="Linear")
    ax.axvspan(0.01, 0.1,  alpha=0.10, color="gold",   label="Transit.")
    ax.axvspan(0.1,  100., alpha=0.10, color="red",    label="Nonlinear")
    ax.axhline(TARGET_DEPTH, color="blue", lw=1.0, ls=":")
    ax.set_xlabel("Peak strain (%)")
    ax.set_xscale("log")
    ax.grid(True, which="both", ls="--", alpha=0.4)
    ax.set_title("Peak Shear Strain")
    ax.legend(fontsize=7, frameon=False)

    ax = axes_e[2]
    ax.plot(max_stress_t10, depth_t_mid, "D-", color="purple",
            lw=1.5, ms=5)
    ax.axhline(TARGET_DEPTH, color="blue", lw=1.0, ls=":")
    ax.set_xlabel("Peak stress (kPa)")
    ax.grid(True, ls="--", alpha=0.4)
    ax.set_title("Peak Shear Stress")

    ax = axes_e[3]
    ax.plot(max_dsp_t, depth_t_e, "v-", color="teal", lw=1.5, ms=5)
    ax.axhline(TARGET_DEPTH, color="blue", lw=1.0, ls=":",
               label=f"Target depth {TARGET_DEPTH} m")
    ax.set_xlabel("Peak displ. (cm)")
    ax.grid(True, ls="--", alpha=0.4)
    ax.set_title("Peak Displacement")
    ax.legend(fontsize=7, frameon=False)

    # Invert ONCE - sharey=True propagates the inversion to all panels.
    axes_e[0].invert_yaxis()

    plt.tight_layout()
    plt.show()

# ── Section 10 summary ──────────────────────────────────────────────────
print(f"\n{'='*65}")
print(f"  SECTION 10 COMPLETE")
print(f"{'='*65}")
print(f"  Tuned simulation:  ratio={ratio_tun:.3f}   RMSE={rmse_tun:.5f} m/s^2")
print(f"  Tuning parameters applied:")
for mid in mat_ids:
    d_o = float(profile_original[mat_col == mid, 2][0])
    d_t = float(np.clip(d_o + best_d_min_offsets[mid], 0.1, 15.0))
    print(f"    Material {mid}: D_min {d_o:.3f}%->{d_t:.3f}%  "
          f"d_scale={best_d_scales[mid]:.2f}  "
          f"strain_shift={STRAIN_SHIFTS_MANUAL[mid]:.2f}")
print(f"{'='*65}")