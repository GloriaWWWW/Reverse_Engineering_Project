import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spikes.spikes import (
    baseline as spike_baseline,
    detect as detect_spikes,
    THR as SPIKE_THRESHOLD,
    DEAD as SPIKE_DEAD_TIME,
)
from drift_noise.drift_noise import estimate_drift_noise, generate

DATA_PATH = ROOT / "data" / "spike_data.npz"
OUT_DIR = Path(__file__).resolve().parent
FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Could not find {DATA_PATH}")

    data = np.load(DATA_PATH)
    t = np.asarray(data["t"], dtype=float)
    raw = np.asarray(data["raw_data"], dtype=float)
    fs = 1.0 / np.median(np.diff(t))

    # Reuse the spike team's baseline and detector for consistent masking.
    baseline_estimate = spike_baseline(raw)
    residual_for_detection = raw - baseline_estimate
    spike_idx = detect_spikes(residual_for_detection)

    params, drift, noise_samples, mask, _ = estimate_drift_noise(
        t, raw, spike_idx, fs,
        mask_pre_ms=3.0, mask_post_ms=20.0
    )

    params["baseline_window_samples"] = 101
    params["spike_detection_threshold_v"] = float(SPIKE_THRESHOLD)
    params["spike_detection_dead_time_samples"] = int(SPIKE_DEAD_TIME)

    params_path = OUT_DIR / "drift_noise_params.json"
    with open(params_path, "w", encoding="utf-8") as f:
        json.dump(params, f, indent=4)

    # Figure 1: baseline estimate and fitted sine drift.
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(t, raw, color="0.75", linewidth=0.5, label="Raw signal")
    ax.plot(t, baseline_estimate, color="tab:blue", linewidth=1.5,
            label="Spike-masked running median (101 ms)")
    ax.plot(t, drift, color="tab:red", linewidth=2,
            label="Fitted sine drift")
    ax.set(xlabel="Time (s)", ylabel="Voltage (V)", title="Drift Estimation")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "drift_fit.png", dpi=200)
    plt.close(fig)

    # Figure 2: residual noise distribution and Gaussian fit.
    mu = params["noise_mean_estimate_v"]
    sigma = params["noise_sd_v"]
    x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 400)
    pdf = np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(noise_samples, bins=50, density=True, alpha=0.65,
            label="Residual noise samples")
    ax.plot(x, pdf, color="tab:red", linewidth=2, label="Gaussian fit")
    ax.set(xlabel="Voltage (V)", ylabel="Probability density",
           title="Noise Distribution")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "noise_distribution.png", dpi=200)
    plt.close(fig)

    # Figure 3: generated background component (drift + noise only).
    sim_drift, sim_noise = generate(t, params, seed=42)
    sim_background = sim_drift + sim_noise
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(t, sim_drift, linewidth=1.5, label="Simulated drift")
    ax.plot(t, sim_background, linewidth=0.6, alpha=0.75,
            label="Simulated drift + noise")
    ax.set(xlabel="Time (s)", ylabel="Voltage (V)",
           title="Simulated Background: Drift + Noise")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "simulation.png", dpi=200)
    plt.close(fig)

    print("Drift and noise estimation complete")
    print(f"Detected spikes for masking: {params['n_spikes_used_for_mask']}")
    print(f"Noise samples retained: {params['n_noise_samples']}")
    print(f"Drift offset: {params['offset_v']:.4f} V")
    print(f"Drift amplitude: {params['amplitude_v']:.4f} V")
    print(f"Drift frequency: {params['frequency_hz']:.4f} Hz")
    print(f"Drift phase: {params['phase_rad']:.4f} rad")
    print(f"Noise mean (estimated): {params['noise_mean_estimate_v']:.4f} V")
    print(f"Noise SD: {params['noise_sd_v']:.4f} V")
    print(f"Saved parameters: {params_path}")
    print(f"Saved figures: {FIG_DIR}")


if __name__ == "__main__":
    main()
