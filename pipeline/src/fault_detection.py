"""
APPROACH
--------
1: Multi-attribute fault probability map
    - NCC coherence
    - Horizontal gradient magnitude
    - Structure tensor eigenvalue ratio
    - Dip variance
    Each attribute is normalized [0,1] and combined into a composite score.

2: Fault path tracing via Simulated Annealing
    - Seed candidate faults from high-probability columns
    - For each seed, SA optimizes a vertical path through the probability field
    - Cost function rewards: high fault probability along path, near-vertical geometry
    - Cost function penalizes: large lateral jumps, non-smooth paths
    - SA escapes local optima (noise, branching) to find geologically plausible paths

3: Filtering and output
    - Merge nearby faults, discard weak detections
    - Annotate the seismic section image
    - Export fault locations to CSV
"""

import numpy as np
from scipy.ndimage import gaussian_filter, uniform_filter, sobel
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd


# CONFIG

FAULT_CONFIG = {
    "gradient_sigma": 2.0,
    "structure_tensor_sigma": 3.0,
    "structure_tensor_rho": 8.0,
    "dip_window_traces": 7,
    "dip_window_samples": 21,
    "weight_coherence": 0.35,
    "weight_gradient": 0.25,
    "weight_structure": 0.25,
    "weight_dip_variance": 0.15,
    "prob_smooth_sigma": (3, 1),
    "prob_threshold": 0.45,
    "column_score_percentile": 85,
    "min_seed_separation": 200,
    "vertical_collapse_method": "mean",
    "sa_initial_temp": 5.0,
    "sa_cooling_rate": 0.995,
    "sa_min_temp": 0.01,
    "sa_max_lateral_jump": 15,
    "sa_path_step": 5,
    "sa_verticality_penalty": 0.3,
    "sa_smoothness_penalty": 0.2,
    "sa_restarts": 3,
    "min_fault_length_frac": 0.15,
    "min_path_score": 0.45,
    "merge_distance_traces": 100,
    "only_high_confidence": True,
    "fault_confidence_levels": {
        "high": 0.55,
        "medium": 0.45,
        "low": 0.25,
    },
}


# MULTI-ATTRIBUTE FAULT PROBABILITY MAP


def compute_horizontal_gradient(data, sigma=2.0):
    smoothed = gaussian_filter(data, sigma=(sigma, sigma))
    grad = np.abs(sobel(smoothed, axis=0))
    p1, p99 = np.percentile(grad, [1, 99])
    if p99 > p1:
        grad = np.clip((grad - p1) / (p99 - p1), 0, 1)
    return grad


def compute_structure_tensor(data, sigma_inner=3.0, sigma_outer=8.0):
    smoothed = gaussian_filter(data.astype(float), sigma=sigma_inner)
    gx = sobel(smoothed, axis=0)
    gz = sobel(smoothed, axis=1)
    gxx = gaussian_filter(gx * gx, sigma=sigma_outer)
    gzz = gaussian_filter(gz * gz, sigma=sigma_outer)
    gxz = gaussian_filter(gx * gz, sigma=sigma_outer)
    trace = gxx + gzz
    det = gxx * gzz - gxz * gxz
    sqrt_disc = np.sqrt(np.maximum(trace**2 - 4 * det, 0))
    lambda1 = (trace + sqrt_disc) / 2
    lambda2 = (trace - sqrt_disc) / 2
    lineament = (lambda1 - lambda2) / (lambda1 + lambda2 + 1e-10)
    p1, p99 = np.percentile(lineament, [1, 99])
    if p99 > p1:
        lineament = np.clip((lineament - p1) / (p99 - p1), 0, 1)
    return lineament


def compute_dip_variance_fast(data, win_traces=7, win_samples=21):
    smoothed = gaussian_filter(data.astype(float), sigma=2.0)
    gx = sobel(smoothed, axis=0)
    gz = sobel(smoothed, axis=1)
    dip_est = np.arctan2(gx, np.abs(gz) + 1e-10)
    dip_mean = uniform_filter(dip_est, size=(win_traces, win_samples))
    dip_sq_mean = uniform_filter(dip_est**2, size=(win_traces, win_samples))
    dip_var = np.maximum(dip_sq_mean - dip_mean**2, 0)
    p1, p99 = np.percentile(dip_var, [1, 99])
    if p99 > p1:
        dip_var = np.clip((dip_var - p1) / (p99 - p1), 0, 1)
    return dip_var


def build_fault_probability_map(filtered_data, coherence, config=None):
    if config is None:
        config = FAULT_CONFIG

    print("  [Phase 1] Computing fault attributes...")

    print("    • Coherence (inverted)...")
    coh_norm = coherence.copy()
    p1, p99 = np.percentile(coh_norm, [1, 99])
    if p99 > p1:
        coh_norm = np.clip((coh_norm - p1) / (p99 - p1), 0, 1)
    fault_from_coherence = 1.0 - coh_norm

    print("    • Horizontal gradient...")
    gradient = compute_horizontal_gradient(
        filtered_data, sigma=config["gradient_sigma"]
    )

    print("    • Structure tensor...")
    lineament = compute_structure_tensor(
        filtered_data, config["structure_tensor_sigma"], config["structure_tensor_rho"]
    )

    print("    • Dip variance...")
    dip_var = compute_dip_variance_fast(
        filtered_data, config["dip_window_traces"], config["dip_window_samples"]
    )

    print("    • Combining attributes...")
    w = config
    prob_map = (
        w["weight_coherence"] * fault_from_coherence
        + w["weight_gradient"] * gradient
        + w["weight_structure"] * lineament
        + w["weight_dip_variance"] * dip_var
    )
    prob_map = gaussian_filter(prob_map, sigma=config["prob_smooth_sigma"])
    p_min, p_max = prob_map.min(), prob_map.max()
    if p_max > p_min:
        prob_map = (prob_map - p_min) / (p_max - p_min)

    attributes = {
        "coherence_inv": fault_from_coherence,
        "gradient": gradient,
        "lineament": lineament,
        "dip_variance": dip_var,
    }
    return prob_map, attributes


# FAULT PATH TRACING VIA SIMULATED ANNEALING


def find_seed_columns(prob_map, seafloor_idx, bottom_idx, config=None):
    if config is None:
        config = FAULT_CONFIG

    n_traces = prob_map.shape[0]
    column_scores = np.zeros(n_traces)
    method = config["vertical_collapse_method"]

    for i in range(n_traces):
        sf, bt = seafloor_idx[i], bottom_idx[i]
        if bt > sf:
            window = prob_map[i, sf:bt]
            if method == "max":
                column_scores[i] = np.max(window)
            elif method == "median":
                column_scores[i] = np.median(window)
            else:
                column_scores[i] = np.mean(window)

    valid = column_scores[column_scores > 0]
    threshold = (
        np.percentile(valid, config["column_score_percentile"]) if len(valid) else 0.0
    )
    peaks, _ = find_peaks(
        column_scores, height=threshold, distance=config["min_seed_separation"]
    )
    order = np.argsort(column_scores[peaks])[::-1]
    seeds = peaks[order]
    print(f"    Found {len(seeds)} seed columns (threshold={threshold:.3f})")
    return seeds, column_scores


def trace_fault_path_sa(prob_map, seed_trace, sf_idx, bt_idx, config=None):
    if config is None:
        config = FAULT_CONFIG

    step = config["sa_path_step"]
    max_jump = config["sa_max_lateral_jump"]
    n_traces = prob_map.shape[0]
    depths = np.arange(sf_idx, bt_idx, step)

    if len(depths) < 3:
        return None, -np.inf

    n_points = len(depths)

    def compute_cost(path):
        reward = (
            sum(
                prob_map[int(np.clip(path[k], 0, n_traces - 1)), d]
                for k, d in enumerate(depths)
            )
            / n_points
        )
        dx = np.diff(path)
        vert_penalty = np.mean(np.abs(dx)) * config["sa_verticality_penalty"]
        smooth_penalty = (
            np.mean(np.abs(np.diff(dx))) * config["sa_smoothness_penalty"]
            if len(dx) > 1
            else 0
        )
        return -reward + vert_penalty + smooth_penalty

    def generate_neighbor(path):
        new_path = path.copy()
        k = np.random.randint(0, n_points)
        new_path[k] = np.clip(
            new_path[k] + np.random.randint(-max_jump, max_jump + 1), 0, n_traces - 1
        )
        return new_path

    best_path, best_cost = None, np.inf

    for _ in range(config["sa_restarts"]):
        path = np.clip(
            np.full(n_points, seed_trace, dtype=float)
            + np.random.randint(-5, 6, n_points),
            0,
            n_traces - 1,
        )
        current_cost = compute_cost(path)
        T = config["sa_initial_temp"]

        while T > config["sa_min_temp"]:
            neighbor = generate_neighbor(path)
            neighbor_cost = compute_cost(neighbor)
            delta = neighbor_cost - current_cost
            if delta < 0 or np.random.random() < np.exp(-delta / T):
                path, current_cost = neighbor, neighbor_cost
            T *= config["sa_cooling_rate"]

        if current_cost < best_cost:
            best_cost, best_path = current_cost, path.copy()

    if best_path is not None:
        avg_prob = (
            sum(
                prob_map[int(np.clip(best_path[k], 0, n_traces - 1)), d]
                for k, d in enumerate(depths)
            )
            / n_points
        )
        return {
            "trace_positions": best_path.astype(int),
            "sample_positions": depths,
            "avg_probability": avg_prob,
            "cost": best_cost,
        }, best_cost

    return None, np.inf


# FILTERING, MERGING, AND OUTPUT


def merge_nearby_faults(faults, config=None):
    if config is None:
        config = FAULT_CONFIG
    if not faults:
        return []

    merge_dist = config["merge_distance_traces"]
    faults = sorted(faults, key=lambda f: f["avg_probability"], reverse=True)
    kept = []

    for fault in faults:
        center = np.mean(fault["trace_positions"])
        if all(abs(center - np.mean(e["trace_positions"])) >= merge_dist for e in kept):
            kept.append(fault)

    return kept


def classify_confidence(fault, config=None):
    if config is None:
        config = FAULT_CONFIG
    p = fault["avg_probability"]
    levels = config["fault_confidence_levels"]
    if p >= levels["high"]:
        return "high"
    elif p >= levels["medium"]:
        return "medium"
    return "low"


# MAIN DETECTION PIPELINE


def detect_all_features(
    filtered_data,
    coherence,
    seafloor_idx,
    horizon_indices,
    depth_axis,
    sample_rate=None,
    config=None,
    envelope=None,
    inst_freq=None,
):
    if config is None:
        config = FAULT_CONFIG

    n_traces, n_samples = filtered_data.shape
    print(f"\n{'='*60}")
    print(f"  AUTOMATED FAULT DETECTION")
    print(f"  {n_traces} traces x {n_samples} samples")
    print(f"{'='*60}\n")

    # Phase 1
    print("[Phase 1] Building fault probability map...")
    prob_map, attributes = build_fault_probability_map(filtered_data, coherence, config)

    bottom_idx = horizon_indices[-1] if horizon_indices else seafloor_idx + 400
    for i in range(n_traces):
        prob_map[i, : seafloor_idx[i]] = 0
        prob_map[i, bottom_idx[i] :] = 0

    print(f"  Probability map range: [{prob_map.min():.3f}, {prob_map.max():.3f}]")

    # Phase 2
    print("\n[Phase 2] Fault path tracing via Simulated Annealing...")
    seeds, column_scores = find_seed_columns(prob_map, seafloor_idx, bottom_idx, config)

    print(f"  Tracing {len(seeds)} candidate fault paths...")
    raw_faults = []
    for idx, seed in enumerate(seeds):
        sf, bt = seafloor_idx[seed], bottom_idx[seed]
        fault, cost = trace_fault_path_sa(prob_map, seed, sf, bt, config)

        if fault is not None:
            path_length = len(fault["sample_positions"])
            window_length = (bt - sf) / config["sa_path_step"]
            long_enough = (
                window_length > 0
                and path_length / window_length >= config["min_fault_length_frac"]
            )
            strong_enough = fault["avg_probability"] >= config["min_path_score"]

            status = "✓" if (long_enough and strong_enough) else "✗"
            reason = (
                ""
                if (long_enough and strong_enough)
                else (
                    " (too short)"
                    if not long_enough
                    else f" (prob={fault['avg_probability']:.3f} below threshold)"
                )
            )
            print(
                f"    Seed {idx+1}/{len(seeds)} trace {seed}: "
                f"prob={fault['avg_probability']:.3f} {status}{reason}"
            )

            if long_enough and strong_enough:
                raw_faults.append(fault)

    # Phase 3
    print(f"\n[Phase 3] Filtering and merging {len(raw_faults)} raw detections...")
    faults = merge_nearby_faults(raw_faults, config)
    print(f"  After merging: {len(faults)}")

    for i, f in enumerate(faults):
        f["depth_positions"] = depth_axis[f["sample_positions"]]
        f["center_trace"] = int(np.mean(f["trace_positions"]))
        f["confidence"] = classify_confidence(f, config)
        f["feature_class"] = "vertical_fault"
        f["fault_id"] = i + 1
        f["feature_id"] = i + 1

    faults = sorted(faults, key=lambda f: f["center_trace"])

    if config.get("only_high_confidence", False):
        before = len(faults)
        faults = [f for f in faults if f["confidence"] == "high"]
        print(f"  High-confidence filter: {before} → {len(faults)}")

    for i, f in enumerate(faults):
        f["fault_id"] = i + 1
        f["feature_id"] = i + 1

    print(f"\n  DETECTED {len(faults)} FAULTS:")
    for f in faults:
        print(
            f"    Fault {f['fault_id']}: trace ~{f['center_trace']}, "
            f"prob={f['avg_probability']:.3f}, confidence={f['confidence']}"
        )

    return {
        "vertical_faults": faults,
        "listric_faults": [],
        "all_features": faults,
        "prob_map": prob_map,
        "attributes": attributes,
        "column_scores": column_scores,
        "vf_column_scores": column_scores,
        "seeds": seeds,
    }


# VISUALIZATION & EXPORT


def annotate_features(
    filtered_data,
    results,
    seafloor_depth,
    horizon_depths,
    depth_axis,
    deep_enough=None,
    output_path="faults_annotated.png",
    line_name="",
):
    faults = results["all_features"]
    prob_map = results["prob_map"]
    column_scores = results["column_scores"]
    n_traces = filtered_data.shape[0]
    power = np.log1p(np.abs(filtered_data.T))
    horizon_colors = ["lime", "yellow", "magenta"]

    fig = plt.figure(figsize=(22, 22))
    fig.suptitle(
        f"Automated Fault Detection — {line_name}",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )

    ax1 = fig.add_subplot(3, 1, 1)
    ax1.imshow(
        power,
        cmap="inferno",
        aspect="auto",
        extent=[0, n_traces, depth_axis[-1], depth_axis[0]],
        vmin=np.percentile(power, 5),
        vmax=np.percentile(power, 95),
    )
    ax1.plot(range(n_traces), seafloor_depth, "c-", lw=0.8, alpha=0.6, label="Seafloor")
    for h, hd in enumerate(horizon_depths):
        valid = ~np.isnan(hd)
        ax1.plot(
            np.where(valid)[0],
            hd[valid],
            "-",
            color=horizon_colors[h],
            lw=0.5,
            alpha=0.5,
            label=f"Horizon {h+1}",
        )
    for f in faults:
        ax1.plot(
            f["trace_positions"],
            f["depth_positions"],
            "-",
            color="#00FFFF",
            lw=5,
            alpha=0.5,
        )
        ax1.plot(
            f["trace_positions"], f["depth_positions"], "-", color="#FFFFFF", lw=1.2
        )
        ax1.annotate(
            f"F{f['fault_id']}",
            xy=(f["trace_positions"][0], f["depth_positions"][0]),
            xytext=(f["trace_positions"][0] + 150, f["depth_positions"][0] - 80),
            fontsize=9,
            fontweight="bold",
            color="#FFFFFF",
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="black",
                alpha=0.8,
                edgecolor="#00FFFF",
                linewidth=1.5,
            ),
            arrowprops=dict(
                arrowstyle="-|>",
                color="#00FFFF",
                lw=1.8,
                connectionstyle="arc3,rad=0.15",
            ),
        )
    ax1.set_title("Seismic Section + Detected Faults")
    ax1.set_xlabel("Trace")
    ax1.set_ylabel("Depth (m)")
    ax1.legend(loc="lower right", fontsize=7)
    ax1.grid(True, alpha=0.15)

    ax2 = fig.add_subplot(3, 2, 3)
    im = ax2.imshow(
        prob_map.T,
        cmap="hot",
        aspect="auto",
        extent=[0, n_traces, depth_axis[-1], depth_axis[0]],
        vmin=0,
        vmax=1,
    )
    for f in faults:
        ax2.plot(
            f["trace_positions"], f["depth_positions"], "-", color="#00FFFF", lw=1.5
        )
    fig.colorbar(im, ax=ax2, label="Fault Probability", shrink=0.8)
    ax2.set_title("Fault Probability Map")
    ax2.set_xlabel("Trace")
    ax2.set_ylabel("Depth (m)")
    ax2.grid(True, alpha=0.15)

    ax3 = fig.add_subplot(3, 2, 4)
    ax3.plot(range(n_traces), column_scores, color="crimson", lw=0.8)
    ax3.fill_between(range(n_traces), column_scores, alpha=0.15, color="crimson")
    for f in faults:
        ax3.axvline(f["center_trace"], color="#00FFFF", lw=2, alpha=0.8)
    ax3.set_title("Column Fault Scores")
    ax3.set_xlabel("Trace")
    ax3.set_ylabel("Mean fault probability")
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"  Saved annotated figure: {output_path}")
    return fig


def export_features_csv(results, output_path="features_detected.csv"):
    rows = []
    for f in results["all_features"]:
        for k in range(len(f["trace_positions"])):
            rows.append(
                {
                    "feature_id": f.get("feature_id", f.get("fault_id")),
                    "feature_class": f.get("feature_class", "vertical_fault"),
                    "confidence": f["confidence"],
                    "avg_probability": f["avg_probability"],
                    "trace_idx": f["trace_positions"][k],
                    "sample_idx": f["sample_positions"][k],
                    "depth_m": f["depth_positions"][k],
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"  Saved fault CSV: {output_path}")
    return df


if __name__ == "__main__":
    print("fault_detection.py - DO NOT RUN THIS, JUST IMPORT IT")
