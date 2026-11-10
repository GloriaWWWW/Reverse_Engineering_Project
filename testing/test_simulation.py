import os
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats, signal


def detect_spikes(trace, fs=10000.0, threshold_std=3.5, min_distance_ms=2.0):
    """Detect spikes using robust median absolute deviation (MAD) thresholding."""
    # High-pass filter to remove slow drift before spike detection
    b, a = signal.butter(3, 100.0 / (0.5 * fs), btype="high")
    filtered = signal.filtfilt(b, a, trace)

    # Robust noise estimate via MAD
    mad = np.median(np.abs(filtered - np.median(filtered))) / 0.6745
    threshold = threshold_std * mad

    min_samples = max(1, int((min_distance_ms / 1000.0) * fs))
    # Handle positive or negative polarity spikes based on skewness
    polarity = 1 if stats.skew(filtered) >= 0 else -1
    peaks, properties = signal.find_peaks(
        polarity * filtered, height=threshold, distance=min_samples
    )
    amplitudes = properties["peak_heights"]
    return peaks, amplitudes, filtered


def evaluate_simulation(
    original_path="data/spike_data.npz",
    simulated_path="results/spike_data.npz",
    fs=10000.0,
    save_figures_dir="testing/figures",
    report_json_path="testing/comparison_report.json"
):
    """
    Compares the simulated .npz file against the original .npz file and outputs
    a quantitative similarity report and diagnostic figures.
    """
    orig_npz = np.load(original_path, allow_pickle=True)
    sim_npz = np.load(simulated_path, allow_pickle=True)

    report = {
        "structure_match": {},
        "time_domain_metrics": {},
        "frequency_domain_metrics": {},
        "spike_metrics": {},
        "overall_similarity_score": 0.0
    }

    # 1. Structure Verification
    orig_keys = set(orig_npz.files)
    sim_keys = set(sim_npz.files)
    keys_match = orig_keys == sim_keys
    shapes_match = all(orig_npz[k].shape == sim_npz[k].shape for k in orig_keys.intersection(sim_keys))
    dtypes_match = all(orig_npz[k].dtype == sim_npz[k].dtype for k in orig_keys.intersection(sim_keys))

    report["structure_match"] = {
        "keys_identical": bool(keys_match),
        "shapes_identical": bool(shapes_match),
        "dtypes_identical": bool(dtypes_match),
        "keys": list(orig_keys)
    }

    # Identify primary signal array
    signal_key = next((k for k in orig_npz.files if k.lower() in ["data", "signal", "voltage", "v", "trace", "traces", "y"]), None)
    if signal_key is None:
        signal_key = max(orig_npz.files, key=lambda k: orig_npz[k].size)

    # Flatten to 1D for global statistical comparison if multi-dimensional
    orig_sig = orig_npz[signal_key].ravel().astype(np.float64)
    sim_sig = sim_npz[signal_key].ravel().astype(np.float64)

    # Infer sampling rate if a time vector exists
    time_key = next((k for k in orig_npz.files if k.lower() in ["t", "time", "times"]), None)
    if time_key is not None and len(orig_npz[time_key]) > 1:
        dt = float(orig_npz[time_key][1] - orig_npz[time_key][0])
        if dt > 0:
            fs = 1.0 / dt

    # 2. Time-Domain & Distribution Metrics
    mean_orig, mean_sim = float(np.mean(orig_sig)), float(np.mean(sim_sig))
    std_orig, std_sim = float(np.std(orig_sig)), float(np.std(sim_sig))
    
    # Wasserstein distance normalized by original standard deviation
    w_dist = float(stats.wasserstein_distance(orig_sig, sim_sig))
    norm_w_dist = w_dist / (std_orig + 1e-12)
    ks_stat, ks_pval = stats.ks_2samp(orig_sig, sim_sig)

    report["time_domain_metrics"] = {
        "mean_original": mean_orig,
        "mean_simulated": mean_sim,
        "std_original": std_orig,
        "std_simulated": std_sim,
        "std_relative_error": float(abs(std_orig - std_sim) / (std_orig + 1e-12)),
        "wasserstein_distance": w_dist,
        "normalized_wasserstein": norm_w_dist,
        "ks_statistic": float(ks_stat),
        "ks_pvalue": float(ks_pval)
    }

    # 3. Frequency-Domain (Power Spectral Density) Metrics
    nperseg = min(len(orig_sig), int(fs * 1.0))
    freqs, psd_orig = signal.welch(orig_sig, fs=fs, nperseg=nperseg)
    _, psd_sim = signal.welch(sim_sig, fs=fs, nperseg=nperseg)

    # Cosine similarity of PSDs
    psd_cos_sim = float(
        np.dot(psd_orig, psd_sim) / (np.linalg.norm(psd_orig) * np.linalg.norm(psd_sim) + 1e-12)
    )
    # Log-Spectral Distance (in dB)
    lsd = float(np.sqrt(np.mean((10 * np.log10((psd_orig + 1e-15) / (psd_sim + 1e-15))) ** 2)))

    report["frequency_domain_metrics"] = {
        "psd_cosine_similarity": psd_cos_sim,
        "log_spectral_distance_db": lsd
    }

    # 4. Spike Train & Waveform Metrics
    peaks_orig, amps_orig, filt_orig = detect_spikes(orig_sig, fs=fs)
    peaks_sim, amps_sim, filt_sim = detect_spikes(sim_sig, fs=fs)

    duration_sec = len(orig_sig) / fs
    rate_orig = float(len(peaks_orig) / duration_sec)
    rate_sim = float(len(peaks_sim) / duration_sec)

    isi_orig = np.diff(peaks_orig) / fs if len(peaks_orig) > 1 else np.array([0.0])
    isi_sim = np.diff(peaks_sim) / fs if len(peaks_sim) > 1 else np.array([0.0])

    amp_mean_orig = float(np.mean(amps_orig)) if len(amps_orig) > 0 else 0.0
    amp_mean_sim = float(np.mean(amps_sim)) if len(amps_sim) > 0 else 0.0

    report["spike_metrics"] = {
        "spike_count_original": int(len(peaks_orig)),
        "spike_count_simulated": int(len(peaks_sim)),
        "firing_rate_hz_original": rate_orig,
        "firing_rate_hz_simulated": rate_sim,
        "firing_rate_rel_error": float(abs(rate_orig - rate_sim) / (rate_orig + 1e-12)),
        "mean_spike_amp_original": amp_mean_orig,
        "mean_spike_amp_simulated": amp_mean_sim,
        "mean_isi_sec_original": float(np.mean(isi_orig)),
        "mean_isi_sec_simulated": float(np.mean(isi_sim))
    }

    # 5. Composite Similarity Score (0 to 100%)
    dist_score = np.exp(-norm_w_dist)
    std_score = np.exp(-report["time_domain_metrics"]["std_relative_error"])
    psd_score = max(0.0, psd_cos_sim)
    rate_score = np.exp(-report["spike_metrics"]["firing_rate_rel_error"])

    overall_score = 100.0 * (0.25 * dist_score + 0.25 * std_score + 0.25 * psd_score + 0.25 * rate_score)
    report["overall_similarity_score"] = round(float(overall_score), 2)

    # Save Report JSON
    os.makedirs(os.path.dirname(report_json_path), exist_ok=True)
    with open(report_json_path, "w") as f:
        json.dump(report, f, indent=4)

    # Generate Diagnostic Comparison Plots
    os.makedirs(save_figures_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Time trace snippet (first 1 second)
    snippet_len = min(len(orig_sig), int(fs * 1.0))
    t_snip = np.arange(snippet_len) / fs
    axes[0, 0].plot(t_snip, orig_sig[:snippet_len], label="Original", alpha=0.8, lw=1)
    axes[0, 0].plot(t_snip, sim_sig[:snippet_len], label="Simulated", alpha=0.7, lw=1)
    axes[0, 0].set_title("Signal Snippet (First 1.0s)")
    axes[0, 0].set_xlabel("Time (s)")
    axes[0, 0].set_ylabel("Amplitude")
    axes[0, 0].legend()

    # Plot 2: Amplitude Distribution Histogram
    axes[0, 1].hist(orig_sig, bins=100, density=True, alpha=0.5, label="Original")
    axes[0, 1].hist(sim_sig, bins=100, density=True, alpha=0.5, label="Simulated")
    axes[0, 1].set_title(f"Amplitude Distribution (Wasserstein={w_dist:.4f})")
    axes[0, 1].set_xlabel("Amplitude")
    axes[0, 1].set_ylabel("Density")
    axes[0, 1].set_yscale("log")
    axes[0, 1].legend()

    # Plot 3: Power Spectral Density
    axes[1, 0].semilogy(freqs, psd_orig, label="Original", alpha=0.8)
    axes[1, 0].semilogy(freqs, psd_sim, label="Simulated", alpha=0.8)
    axes[1, 0].set_title(f"Power Spectral Density (CosSim={psd_cos_sim:.4f})")
    axes[1, 0].set_xlabel("Frequency (Hz)")
    axes[1, 0].set_ylabel("PSD")
    axes[1, 0].legend()

    # Plot 4: Inter-Spike Interval (ISI) Histogram
    if len(isi_orig) > 1 and len(isi_sim) > 1:
        max_isi = np.percentile(np.concatenate([isi_orig, isi_sim]), 98)
        bins = np.linspace(0, max_isi, 40)
        axes[1, 1].hist(isi_orig, bins=bins, density=True, alpha=0.5, label=f"Original ({rate_orig:.1f} Hz)")
        axes[1, 1].hist(isi_sim, bins=bins, density=True, alpha=0.5, label=f"Simulated ({rate_sim:.1f} Hz)")
    axes[1, 1].set_title("Inter-Spike Interval (ISI) Distribution")
    axes[1, 1].set_xlabel("ISI (s)")
    axes[1, 1].set_ylabel("Density")
    axes[1, 1].legend()

    plt.suptitle(f"Simulation vs. Original Comparison — Overall Similarity: {report['overall_similarity_score']}%", fontsize=14)
    plt.tight_layout()
    fig_path = os.path.join(save_figures_dir, "comparison_summary.png")
    plt.savefig(fig_path, dpi=150)
    plt.close(fig)

    print("=== Simulation Comparison Summary ===")
    print(json.dumps(report, indent=2))
    print(f"\nSaved visual comparison to: {fig_path}")
    return report


if __name__ == "__main__":
    evaluate_simulation()
