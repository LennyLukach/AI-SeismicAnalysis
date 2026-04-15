"""
Full seismic fault detection pipeline combining:
  - Feature extraction (NCC coherence, horizons, bandpass)
  - Fault detector (multi-attribute + simulated annealing)
  - Context-aware bounding boxes for CNN training
"""

import json
import re
import sys
import traceback
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from obspy import read
from scipy.signal import butter, filtfilt, find_peaks, hilbert, medfilt

sys.path.insert(0, str(Path(__file__).parent))
from fault_detection import detect_all_features, FAULT_CONFIG


#THESE ARE THE ONLY THINGS THAT YOU NEED TO MODFIY

# Set LINE to a single line name, or None to process every SGY file found
LINE = None

AIRGUN_DATA_ROOT = "../../airgun_data"

# Set to True to run multiple SA iterations per seed and average their probabilities
USE_AVERAGED_SA = False
# (If True) how many SA runs to average per seed
SA_RUNS = 5
# (If not using averaged SA) set random seed for reproducibility of single SA run
RANDOM_SEED = 50

LATERAL_CONTEXT_MULTIPLIER = 1.2
VERTICAL_CONTEXT_MULTIPLIER = 1.1
MIN_CONTEXT_PX = 40
MIN_BOX_WIDTH_PX = 80
MIN_BOX_HEIGHT_PX = 100

VELOCITY = 1500
N_HORIZONS = 3
SUPPRESS_SAMPLES = 80
MIN_HORIZON_SEP = 40
MAX_HORIZON_OFFSET = 800
MIN_SEAFLOOR_DEPTH_M = 600
COHERENCE_WINDOW = 5
NCC_FAULT_THRESH = 0.6
SMOOTH_KERNEL = 101
BANDPASS_LOW_HZ = 5.0
BANDPASS_HIGH_HZ = 200.0

SECTION_WIDTH_IN = 18
SECTION_HEIGHT_IN = 10
SECTION_DPI = 150

# Built dynamically at runtime from all SGY files found in AIRGUN_DATA_ROOT.
# Every 5th line in this sorted list goes to val, the rest to train.
# Populated in the entry point before any lines are processed.
ALL_LINES = []

# ══════════════════════════════════════════════════════════════════════════════
# DERIVED PATHS
# Each run gets its own timestamped folder under results/ containing
# annotations/, diagnostics/, images/, and the log file.
# ══════════════════════════════════════════════════════════════════════════════

RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

SCRIPT_DIR = Path(__file__).parent.parent
RUN_DIR = SCRIPT_DIR / "results" / f"run_{RUN_TIMESTAMP}"
IMAGES_DIR = RUN_DIR / "images"
ANNOT_DIR = RUN_DIR / "annotations"
DIAG_DIR = RUN_DIR / "diagnostics"
LOG_PATH = RUN_DIR / "run.log"

for d in [IMAGES_DIR, ANNOT_DIR, DIAG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

IMG_W = SECTION_WIDTH_IN * SECTION_DPI
IMG_H = SECTION_HEIGHT_IN * SECTION_DPI

# ══════════════════════════════════════════════════════════════════════════════
# LOGGER
# ══════════════════════════════════════════════════════════════════════════════

_log_file = open(LOG_PATH, "w", buffering=1)  # line-buffered so it flushes on crash


def log(msg="", level="INFO"):
    timestamp = datetime.now().strftime("%H:%M:%S")
    line = f"[{timestamp}] [{level:5s}] {msg}"
    print(line)
    _log_file.write(line + "\n")


def log_section(title):
    bar = "─" * 50
    log()
    log(bar)
    log(f"  {title}")
    log(bar)


def log_error(msg, exc=None):
    log(msg, level="ERROR")
    if exc:
        tb = traceback.format_exc()
        for line in tb.splitlines():
            log(f"  {line}", level="ERROR")


# ══════════════════════════════════════════════════════════════════════════════
# NAV LINE NAME TRANSLATION
# ══════════════════════════════════════════════════════════════════════════════


def translate_sgy_stem_to_nav_line(stem):
    """
    Translate a SGY filename stem to the corresponding NAV line name.
    Returns a list of candidates to try in priority order; the direct stem
    is always first so datasets that already match need no special casing.

    Conventions handled:

      F-2-87 / F-3-87  (NAV uses FXXX-N):
        f287stk01   -> F287-1
        f287stk14a  -> F287-14A
        f287stks1   -> F287-S1

      F-6-88  (NAV uses bare integers, with optional A suffix):
        f688stk04   -> 4
        f688stk04a  -> 4A

      F-10-88 / F-12-89  (NAV uses bare integers):
        f1088stk01  -> 1
        f1289stk03  -> 3

      W-3-69-BS  (NAV uses letter codes; SGY stem encodes SP count as suffix):
        G_17558     -> G        (strip trailing _NNNNN)
        G_1_408     -> G_1
        Q-EXT_45287 -> QEXT    (also normalise -EXT -> EXT)
        WA-024_1187 -> WA-024

      L-12-82-WG / L-8-81-WG  (NAV uses bare line number):
        202wg82stk  -> 202      (extract leading digits)
        100_1204    -> 100      (extract leading digits before _)

      T-21-10-AT:
        T2010.100.mig.1500 -> T2010.100.mig.1500  (direct match, no translation)
    """
    candidates = [stem]  # always try direct match first

    # ── F-series: fNNNNstkMM[a] or fNNNNstksMM ──────────────────────────────
    if re.match(r"^f\d+stk", stem, re.IGNORECASE):
        # F-2-87 / F-3-87 style: uppercase + STK->- + strip leading zeros
        # Handles trailing alpha suffix (14a -> 14A) and s-lines (stks1 -> S1)
        s = stem.upper()
        s = re.sub(r"STK", "-", s)
        s = re.sub(r"-0*(\d)", r"-\1", s)
        if s not in candidates:
            candidates.append(s)

        m = re.match(r"^f\d+stks?(\d+)(a?)$", stem, re.IGNORECASE)
        if m:
            num = str(int(m.group(1)))
            suffix = m.group(2).upper()
            bare = num + suffix
            if bare not in candidates:
                candidates.append(bare)
            if suffix and num not in candidates:
                candidates.append(num)

    m = re.match(r"^(.+?)_\d+$", stem)
    if m:
        base = m.group(1)
        base_norm = re.sub(r"-EXT$", "EXT", base, flags=re.IGNORECASE)
        if base_norm not in candidates:
            candidates.append(base_norm)
        if base != base_norm and base not in candidates:
            candidates.append(base)

    m = re.match(r"^(\d+)[_a-zA-Z]", stem)
    if m:
        num = m.group(1)
        if num not in candidates:
            candidates.append(num)

    return candidates


def resolve_nav_line(stem, nav, log_fn=None):
    """
    Find the NAV line name for a given SGY stem by trying direct match first,
    then dataset specific translations. Returns the matching subset of nav as
    a DataFrame. Raises RuntimeError if no match is found.
    """
    for candidate in translate_sgy_stem_to_nav_line(stem):
        subset = nav[nav["line"] == candidate]
        if not subset.empty:
            if candidate != stem and log_fn:
                log_fn(f"  NAV line matched via translation: '{stem}' -> '{candidate}'")
            return subset.copy()

    # Nothing matched
    available = sorted(nav["line"].unique().tolist())
    if log_fn:
        log_fn(
            f"  NAV match failed for '{stem}'. "
            f"Tried: {translate_sgy_stem_to_nav_line(stem)}. "
            f"Available: {available}",
            level="WARN",
        )
    raise RuntimeError(
        f"No NAV line matched '{stem}'. "
        f"Tried: {translate_sgy_stem_to_nav_line(stem)}."
    )


# ══════════════════════════════════════════════════════════════════════════════
# DISCOVERY
# ══════════════════════════════════════════════════════════════════════════════


def discover_all_lines(airgun_data_root):
    """
    Walk airgun_data_root. Each immediate subfolder is one airgun dataset.
    Expects:   <dataset>/Data/*.sgy
               <dataset>/Navigation/*.csv
    Returns list of (line_name, sgy_path, nav_path, dataset_name).
    """
    root = Path(airgun_data_root)
    entries = []

    for dataset_dir in sorted(root.iterdir()):
        if not dataset_dir.is_dir():
            continue
        data_dir = dataset_dir / "Data"
        nav_dir = dataset_dir / "Navigation"

        if not data_dir.exists():
            log(f"No Data/ folder in {dataset_dir.name} — skipping", level="WARN")
            continue
        if not nav_dir.exists():
            log(f"No Navigation/ folder in {dataset_dir.name} — skipping", level="WARN")
            continue

        nav_files = list(nav_dir.glob("*.csv"))
        if not nav_files:
            log(f"No CSV in {dataset_dir.name}/Navigation/ — skipping", level="WARN")
            continue
        nav_path = str(nav_files[0])

        for sgy in sorted(data_dir.glob("*.sgy")):
            entries.append((sgy.stem, str(sgy), nav_path, dataset_dir.name))

    return entries


def find_single_line(line, airgun_data_root):
    """Find a specific line by name across all datasets."""
    for line_name, sgy_path, nav_path, dataset in discover_all_lines(airgun_data_root):
        if line_name == line:
            return line_name, sgy_path, nav_path, dataset
    raise FileNotFoundError(f"{line}.sgy not found under {airgun_data_root}")


# ══════════════════════════════════════════════════════════════════════════════
# AVERAGED SA
# ══════════════════════════════════════════════════════════════════════════════


def detect_with_averaged_sa(
    filtered, coherence, seafloor_idx, horizon_indices, depth_axis, sample_rate, n_runs
):
    from fault_detection import (
        build_fault_probability_map,
        find_seed_columns,
        trace_fault_path_sa,
        merge_nearby_faults,
        classify_confidence,
    )

    config = FAULT_CONFIG.copy()
    n_traces = filtered.shape[0]
    prob_map, attributes = build_fault_probability_map(filtered, coherence, config)
    bottom_idx = horizon_indices[-1] if horizon_indices else seafloor_idx + 400
    for i in range(n_traces):
        prob_map[i, : seafloor_idx[i]] = 0
        prob_map[i, bottom_idx[i] :] = 0

    seeds, column_scores = find_seed_columns(prob_map, seafloor_idx, bottom_idx, config)
    log(f"  Averaging SA over {n_runs} runs per seed ({len(seeds)} seeds)")

    raw_faults = []
    for idx, seed in enumerate(seeds):
        sf, bt = seafloor_idx[seed], bottom_idx[seed]
        run_probs = []
        best_fault, best_cost = None, np.inf

        for _ in range(n_runs):
            fault, cost = trace_fault_path_sa(prob_map, seed, sf, bt, config)
            if fault is not None:
                run_probs.append(fault["avg_probability"])
                if cost < best_cost:
                    best_cost, best_fault = cost, fault

        if best_fault is not None and run_probs:
            avg_prob = float(np.mean(run_probs))
            best_fault["avg_probability"] = avg_prob
            path_length = len(best_fault["sample_positions"])
            window_length = (bt - sf) / config["sa_path_step"]
            long_enough = (
                window_length > 0
                and path_length / window_length >= config["min_fault_length_frac"]
            )
            strong_enough = avg_prob >= config["min_path_score"]
            status = "PASS" if (long_enough and strong_enough) else "FAIL"
            log(
                f"    Seed {idx+1}/{len(seeds)} trace {seed}: avg_prob={avg_prob:.3f} [{status}]"
            )
            if long_enough and strong_enough:
                raw_faults.append(best_fault)

    faults = merge_nearby_faults(raw_faults, config)
    for i, f in enumerate(faults):
        f["depth_positions"] = depth_axis[f["sample_positions"]]
        f["center_trace"] = int(np.mean(f["trace_positions"]))
        f["confidence"] = classify_confidence(f, config)
        f["feature_class"] = "vertical_fault"
        f["fault_id"] = i + 1
        f["feature_id"] = i + 1

    faults = sorted(faults, key=lambda f: f["center_trace"])
    if config.get("only_high_confidence", False):
        faults = [f for f in faults if f["confidence"] == "high"]
    for i, f in enumerate(faults):
        f["fault_id"] = i + 1
        f["feature_id"] = i + 1

    return {
        "all_features": faults,
        "vertical_faults": faults,
        "listric_faults": [],
        "prob_map": prob_map,
        "attributes": attributes,
        "column_scores": column_scores,
        "vf_column_scores": column_scores,
        "seeds": seeds,
    }


# ══════════════════════════════════════════════════════════════════════════════
# PER-LINE PROCESSING
# ══════════════════════════════════════════════════════════════════════════════


def process_line(line, sgy_path, nav_path, dataset_name):
    log_section(f"LINE: {line}")
    log(f"Dataset  : {dataset_name}")
    log(f"SGY      : {sgy_path}")
    log(f"NAV      : {nav_path}")
    log(
        f"SA mode  : {'averaged ' + str(SA_RUNS) + 'x ' if USE_AVERAGED_SA else 'single (seed=' + str(RANDOM_SEED) + ')'}"
    )

    diag_labeled = DIAG_DIR / f"{line}_labeled.png"
    section_png = IMAGES_DIR / f"{line}_section_cnn.png"

    log()
    log("STEP 1 — Feature extraction")

    log("  Loading navigation...")
    nav = pd.read_csv(
        nav_path,
        comment="#",
        header=None,
        names=["permit", "line", "SP", "lat", "lon"],
        dtype=str,
    )
    nav["line"] = nav["line"].str.strip()
    nav["SP"] = nav["SP"].str.strip().astype(int)

    log("  Loading SEG-Y...")
    st = read(sgy_path)
    sample_rate = st[0].stats.sampling_rate
    dt = 1.0 / sample_rate
    n_samples = len(st[0].data)
    time_axis = np.arange(n_samples) * dt
    depth_axis = (time_axis * VELOCITY) / 2
    log(f"  Traces: {len(st)}  Samples: {n_samples}  dt: {dt*1000:.3f} ms")
    log(f"  Depth:  {depth_axis[0]:.1f} m → {depth_axis[-1]:.1f} m")

    log("  Matching traces to navigation...")
    line_nav = resolve_nav_line(line, nav, log_fn=log)
    trace_coords = []
    for tr in st:
        sp = tr.stats.segy.trace_header.energy_source_point_number
        match = line_nav[line_nav["SP"] == sp]
        if not match.empty:
            trace_coords.append(
                {
                    "SP": sp,
                    "lat": match.iloc[0]["lat"],
                    "lon": match.iloc[0]["lon"],
                    "trace_data": tr.data.copy(),
                }
            )
    log(f"  Matched {len(trace_coords)} / {len(st)} traces")
    if not trace_coords:
        raise RuntimeError(
            "No traces matched navigation — check line name vs NAV file."
        )
    raw_data = np.array([t["trace_data"] for t in trace_coords])

    log("  Bandpass filtering...")

    def bandpass(data, low, high, fs, order=4):
        nyq = fs / 2.0
        low_n = low / nyq
        high_n = min(high / nyq, 0.99)
        if low_n >= high_n:
            log(
                f"  WARNING: sample rate {fs} Hz too low for bandpass "
                f"[{low}–{high} Hz], skipping filter",
                level="WARN",
            )
            return data
        b, a = butter(order, [low_n, high_n], btype="band")
        return filtfilt(b, a, data, axis=1)

    filtered = bandpass(raw_data, BANDPASS_LOW_HZ, BANDPASS_HIGH_HZ, sample_rate)

    log("  Computing seismic attributes...")
    analytic = hilbert(filtered, axis=1)
    envelope = np.abs(analytic)
    inst_phase = np.unwrap(np.angle(analytic), axis=1)
    cosine_phase = np.cos(np.angle(analytic))
    inst_freq = np.diff(inst_phase, axis=1) / (2 * np.pi * dt)
    inst_freq = np.concatenate([inst_freq, inst_freq[:, -1:]], axis=1)

    log("  Picking seafloor...")

    def pick_seafloor(trace_abs, percentile=90):
        threshold = np.percentile(trace_abs, percentile)
        peaks, _ = find_peaks(trace_abs, height=threshold, prominence=threshold * 0.1)
        if len(peaks) > 0:
            return peaks[0]
        above = np.where(trace_abs > threshold)[0]
        return above[0] if len(above) > 0 else 0

    seafloor_idx = np.array(
        [pick_seafloor(np.abs(filtered[i])) for i in range(len(trace_coords))]
    )
    seafloor_idx = medfilt(seafloor_idx.astype(float), SMOOTH_KERNEL).astype(int)
    seafloor_idx = np.clip(seafloor_idx, 0, n_samples - 1)
    seafloor_depth = depth_axis[seafloor_idx]
    deep_enough = seafloor_depth >= MIN_SEAFLOOR_DEPTH_M
    log(
        f"  Seafloor depth: {seafloor_depth.min():.0f}–{seafloor_depth.max():.0f} m  "
        f"({np.sum(~deep_enough)} shallow traces masked)"
    )

    log("  Picking horizons...")

    def pick_horizons_below(
        trace_abs, env_trace, start_idx, n, suppress, min_sep, max_offset
    ):
        hard_ceil = min(start_idx + max_offset, len(trace_abs))
        search = trace_abs.copy()
        search[: start_idx + suppress] = 0
        search[hard_ceil:] = 0
        region = search[start_idx:hard_ceil]
        min_height = (
            np.percentile(region[region > 0], 30) if region[region > 0].size > 0 else 0
        )
        peaks, _ = find_peaks(region, height=min_height, distance=min_sep)
        peaks = peaks + start_idx
        picks, used, last_pick = [], search.copy(), start_idx
        if len(peaks) >= n:
            valid = [p for p in peaks if p >= last_pick + min_sep]
            for p in valid[:n]:
                picks.append(p)
        else:
            for _ in range(n):
                ms = last_pick + min_sep
                if ms >= hard_ceil:
                    break
                local = used[ms:hard_ceil]
                if local.max() == 0:
                    break
                idx = np.argmax(local) + ms
                picks.append(idx)
                last_pick = idx
                used[max(0, idx - suppress) : min(len(used), idx + suppress)] = 0
        return picks

    raw_h_idx = [[] for _ in range(N_HORIZONS)]
    for i in range(len(trace_coords)):
        if not deep_enough[i]:
            for h in range(N_HORIZONS):
                raw_h_idx[h].append(-1)
            continue
        picks = pick_horizons_below(
            np.abs(filtered[i]),
            envelope[i],
            seafloor_idx[i],
            N_HORIZONS,
            SUPPRESS_SAMPLES,
            MIN_HORIZON_SEP,
            MAX_HORIZON_OFFSET,
        )
        for h in range(N_HORIZONS):
            val = (
                picks[h]
                if h < len(picks)
                else min(
                    seafloor_idx[i] + SUPPRESS_SAMPLES * (h + 1),
                    seafloor_idx[i] + MAX_HORIZON_OFFSET - 1,
                )
            )
            raw_h_idx[h].append(int(np.clip(val, 0, n_samples - 1)))

    horizon_indices, horizon_depths = [], []
    for h in range(N_HORIZONS):
        arr = np.array(raw_h_idx[h], dtype=float)
        sentinel = arr == -1
        if sentinel.any() and (~sentinel).any():
            vi = np.where(~sentinel)[0]
            arr[sentinel] = np.interp(np.where(sentinel)[0], vi, arr[vi])
        smoothed = np.clip(medfilt(arr, SMOOTH_KERNEL).astype(int), 0, n_samples - 1)
        horizon_indices.append(smoothed)
        depths = depth_axis[smoothed].astype(float)
        depths[~deep_enough] = np.nan
        horizon_depths.append(depths)

    log("  Computing NCC coherence (slow)...")

    def ncc_coherence(data, window):
        n_tr, n_s = data.shape
        coherence = np.ones((n_tr, n_s))
        norm = (data - data.mean(axis=1, keepdims=True)) / (
            data.std(axis=1, keepdims=True) + 1e-9
        )
        for i in range(window, n_tr - window):
            c, c_n = norm[i], np.sqrt(np.sum(norm[i] ** 2)) + 1e-9
            vals = []
            for j in range(i - window, i + window + 1):
                if j == i:
                    continue
                nb = norm[j]
                n_n = np.sqrt(np.sum(nb**2)) + 1e-9
                vals.append(np.clip((c * nb) / (c_n * n_n / n_s), 0, 1))
            coherence[i] = np.mean(vals, axis=0)
        return coherence

    coherence = ncc_coherence(filtered, COHERENCE_WINDOW)
    n_traces = len(trace_coords)

    log("  Saving CNN section image...")
    power = np.log1p(np.abs(filtered.T))
    fig2, ax2 = plt.subplots(figsize=(SECTION_WIDTH_IN, SECTION_HEIGHT_IN))
    fig2.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax2.imshow(
        power,
        cmap="inferno",
        aspect="auto",
        extent=[0, n_traces, depth_axis[-1], depth_axis[0]],
        vmin=np.percentile(power, 5),
        vmax=np.percentile(power, 95),
    )
    ax2.set_xlim(0, n_traces)
    ax2.set_ylim(depth_axis[-1], depth_axis[0])
    ax2.axis("off")
    fig2.savefig(str(section_png), dpi=SECTION_DPI, bbox_inches="tight", pad_inches=0)
    plt.close(fig2)
    log(f"  Saved: {section_png.name}  ({IMG_W}x{IMG_H}px)")

    depth_min = float(depth_axis[0])
    depth_max = float(depth_axis[-1])
    depth_range = depth_max - depth_min
    log("  Step 1 complete")

    log()
    log("STEP 2 — Fault detection")

    if USE_AVERAGED_SA:
        log(f"  Mode: averaged SA ({SA_RUNS} runs per seed)")
        results = detect_with_averaged_sa(
            filtered,
            coherence,
            seafloor_idx,
            horizon_indices,
            depth_axis,
            sample_rate,
            SA_RUNS,
        )
    else:
        log(f"  Mode: single SA run (seed={RANDOM_SEED})")
        np.random.seed(RANDOM_SEED)
        results = detect_all_features(
            filtered_data=filtered,
            coherence=coherence,
            seafloor_idx=seafloor_idx,
            horizon_indices=horizon_indices,
            depth_axis=depth_axis,
            sample_rate=sample_rate,
        )

    faults = results["all_features"]
    log(f"  Detected {len(faults)} high-confidence faults")
    for f in faults:
        log(
            f"    Fault {f['fault_id']:2d} | trace ~{f['center_trace']:5d} | prob={f['avg_probability']:.3f}"
        )
    log("  Step 2 complete")

    log()
    log("STEP 3 — Bounding boxes")

    def fault_path_to_box(fault, n_traces, img_w, img_h, depth_min, depth_range):
        tr = fault["trace_positions"]
        dp = fault["depth_positions"]
        x_scale = img_w / n_traces
        core_x1 = float(np.min(tr)) * x_scale
        core_x2 = float(np.max(tr)) * x_scale
        core_w = max(core_x2 - core_x1, 1.0)
        y_scale = img_h / depth_range
        core_y1 = (float(np.min(dp)) - depth_min) * y_scale
        core_y2 = (float(np.max(dp)) - depth_min) * y_scale
        core_h = max(core_y2 - core_y1, 1.0)
        lat_pad = max(core_w * LATERAL_CONTEXT_MULTIPLIER, MIN_CONTEXT_PX)
        vert_pad = max(
            core_h * (VERTICAL_CONTEXT_MULTIPLIER - 1) / 2, MIN_CONTEXT_PX * 0.3
        )
        x1, x2 = core_x1 - lat_pad, core_x2 + lat_pad
        y1, y2 = core_y1 - vert_pad, core_y2 + vert_pad
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        bw = max(x2 - x1, MIN_BOX_WIDTH_PX)
        bh = max(y2 - y1, MIN_BOX_HEIGHT_PX)
        x1, x2 = cx - bw / 2, cx + bw / 2
        y1, y2 = cy - bh / 2, cy + bh / 2
        x1 = max(0.0, x1)
        y1 = max(0.0, y1)
        x2 = min(float(img_w), x2)
        y2 = min(float(img_h), y2)
        return [round(x1, 2), round(y1, 2), round(x2 - x1, 2), round(y2 - y1, 2)]

    boxes = []
    for fault in faults:
        box = fault_path_to_box(fault, n_traces, IMG_W, IMG_H, depth_min, depth_range)
        boxes.append(
            {
                "box": box,
                "fault_id": fault["fault_id"],
                "confidence": fault["confidence"],
                "avg_prob": fault["avg_probability"],
            }
        )
        log(
            f"  Box {fault['fault_id']:2d} | [{box[0]:.0f}, {box[1]:.0f}, "
            f"{box[2]:.0f}w, {box[3]:.0f}h px]"
        )

    log(f"  Generated {len(boxes)} boxes")
    log("  Step 3 complete")

    log()
    log("STEP 4 — Dataset building")

    log("  Saving labeled review image...")
    img = Image.open(str(section_png)).convert("RGBA")
    over = Image.new("RGBA", img.size, (0, 0, 0, 0))
    drw = ImageDraw.Draw(over)
    for b in boxes:
        x, y, bw, bh = b["box"]
        drw.rectangle([x, y, x + bw, y + bh], fill=(255, 50, 50, 35))
        drw.rectangle([x, y, x + bw, y + bh], outline=(255, 50, 50, 220), width=2)
        drw.text(
            (x + 4, y + 4),
            f"F{b['fault_id']} p={b['avg_prob']:.2f}",
            fill=(255, 200, 200),
        )
    labeled = Image.alpha_composite(img, over).convert("RGB")
    drw2 = ImageDraw.Draw(labeled)
    drw2.rectangle([6, IMG_H - 28, 240, IMG_H - 4], fill=(15, 15, 15))
    drw2.rectangle([10, IMG_H - 24, 26, IMG_H - 8], fill=(255, 50, 50))
    drw2.text(
        (30, IMG_H - 26),
        f"SA fault paths + context boxes ({len(boxes)} total)",
        fill=(240, 240, 240),
    )
    labeled.save(str(diag_labeled))
    log(f"  Saved: {diag_labeled.name}")

    line_index = ALL_LINES.index(line) if line in ALL_LINES else 0
    is_val = line_index % 5 == 4
    split = "val" if is_val else "train"
    log(f"  Split: {split}")

    def load_or_init_coco(path):
        if path.exists():
            with open(path) as f:
                return json.load(f)
        return {
            "info": {"description": "NAMSS seismic fault dataset"},
            "categories": [{"id": 1, "name": "fault", "supercategory": "geology"}],
            "images": [],
            "annotations": [],
        }

    def next_id(coco, key):
        items = coco.get(key, [])
        return max((x["id"] for x in items), default=0) + 1

    train_json_path = ANNOT_DIR / "train.json"
    val_json_path = ANNOT_DIR / "val.json"
    train_coco = load_or_init_coco(train_json_path)
    val_coco = load_or_init_coco(val_json_path)
    img_filename = section_png.name

    for coco_data in [train_coco, val_coco]:
        existing_ids = {
            img["id"] for img in coco_data["images"] if img.get("source") == line
        }
        if existing_ids:
            log(f"  Removing existing entry for {line} (re-run)")
            coco_data["images"] = [
                img for img in coco_data["images"] if img.get("source") != line
            ]
            coco_data["annotations"] = [
                ann
                for ann in coco_data["annotations"]
                if ann["image_id"] not in existing_ids
            ]

    coco = val_coco if is_val else train_coco
    img_id = next_id(coco, "images")
    ann_id = next_id(coco, "annotations")

    coco["images"].append(
        {
            "id": img_id,
            "file_name": img_filename,
            "width": IMG_W,
            "height": IMG_H,
            "source": line,
        }
    )
    n_annotations = 0
    for b in boxes:
        x, y, bw, bh = b["box"]
        coco["annotations"].append(
            {
                "id": ann_id,
                "image_id": img_id,
                "category_id": 1,
                "bbox": [x, y, bw, bh],
                "area": round(bw * bh, 2),
                "iscrowd": 0,
                "fault_id": b["fault_id"],
                "confidence": b["confidence"],
                "avg_prob": round(b["avg_prob"], 4),
            }
        )
        ann_id += 1
        n_annotations += 1

    with open(str(train_json_path), "w") as f:
        json.dump(train_coco, f, indent=2)
    with open(str(val_json_path), "w") as f:
        json.dump(val_coco, f, indent=2)

    log(f"  Added {n_annotations} annotations to {split}.json")
    log(
        f"  train.json: {len(train_coco['images'])} images, {len(train_coco['annotations'])} annotations"
    )
    log(
        f"  val.json:   {len(val_coco['images'])} images,  {len(val_coco['annotations'])} annotations"
    )
    log("  Step 4 complete")
    log()
    log(
        f"  DONE  {line}  →  {split}  |  {len(faults)} faults  |  {n_annotations} boxes"
    )


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

log(f"{'='*60}")
log(f"  NAMSS Seismic Fault Pipeline v2")
log(f"  Started : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
log(f"  Run dir : {RUN_DIR}")
log(
    f"  SA mode : {'averaged ' + str(SA_RUNS) + 'x ' if USE_AVERAGED_SA else 'single (seed=' + str(RANDOM_SEED) + ')'}"
)
log(f"{'='*60}")

# Discover all lines and build the global sorted list for train/val splitting
_all_entries = discover_all_lines(AIRGUN_DATA_ROOT)
ALL_LINES = sorted(set(e[0] for e in _all_entries))
log(f"\nDiscovered {len(ALL_LINES)} unique lines across all datasets")
log(f"Val lines (every 5th): {[l for i, l in enumerate(ALL_LINES) if i % 5 == 4]}")

if LINE is not None:
    _, sgy_path, nav_path, dataset = find_single_line(LINE, AIRGUN_DATA_ROOT)
    try:
        process_line(LINE, sgy_path, nav_path, dataset)
    except Exception as e:
        log_error(f"Failed on line {LINE}: {e}", e)
else:
    entries = _all_entries
    log(
        f"\nProcessing {len(entries)} SGY files across all datasets in {AIRGUN_DATA_ROOT}"
    )
    for entry in entries:
        log(f"  {entry[3]:40s}  {entry[0]}")

    results_summary = []
    for line, sgy_path, nav_path, dataset in entries:
        try:
            process_line(line, sgy_path, nav_path, dataset)
            results_summary.append((line, dataset, "OK"))
        except Exception as e:
            log_error(f"SKIPPING {line} ({dataset}): {e}", e)
            results_summary.append((line, dataset, f"FAILED: {e}"))

    log()
    log("=" * 60)
    log("  FINAL SUMMARY")
    log("=" * 60)
    ok = [r for r in results_summary if r[2] == "OK"]
    failed = [r for r in results_summary if r[2] != "OK"]
    log(f"  Completed : {len(ok)} / {len(results_summary)}")
    log(f"  Failed    : {len(failed)}")
    if failed:
        log()
        log("  Failed lines:")
        for line, dataset, reason in failed:
            log(f"    {line:40s}  {reason}", level="ERROR")

log()
log(f"  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
log(f"  Run dir : {RUN_DIR}")

_log_file.close()
