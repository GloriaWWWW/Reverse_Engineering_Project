import numpy as np
from scipy.optimize import curve_fit


def sine_function(t, offset, amplitude, frequency_hz, phase_rad):
    return offset + amplitude * np.sin(2 * np.pi * frequency_hz * t + phase_rad)


def make_spike_mask(n_samples, spike_idx, fs, pre_ms=3.0, post_ms=20.0):
    """Return True for samples retained for noise estimation."""
    pre = int(round(pre_ms * fs / 1000.0))
    post = int(round(post_ms * fs / 1000.0))
    mask = np.ones(n_samples, dtype=bool)
    for s in np.asarray(spike_idx, dtype=int):
        mask[max(0, s - pre):min(n_samples, s + post)] = False
    return mask


def estimate_drift_noise(t, raw, spike_idx, fs,
                         mask_pre_ms=3.0, mask_post_ms=20.0):
    """Estimate sinusoidal drift and Gaussian noise using the team's spike mask."""
    mask = make_spike_mask(
        len(raw), spike_idx, fs,
        pre_ms=mask_pre_ms, post_ms=mask_post_ms
    )

    p0 = [0.004, 0.648, 0.091, 0.283]
    bounds = ([-1.0, 0.0, 0.03, -np.pi], [1.0, 2.0, 0.20, np.pi])
    params, covariance = curve_fit(
        sine_function,
        t[mask],
        raw[mask],
        p0=p0,
        bounds=bounds,
        maxfev=20000
    )

    drift = sine_function(t, *params)
    residual = raw - drift
    noise_samples = residual[mask]
    noise_mean = float(np.mean(noise_samples))
    noise_sd = float(np.std(noise_samples, ddof=0))

    result = {
        "offset_v": float(params[0]),
        "amplitude_v": float(params[1]),
        "frequency_hz": float(params[2]),
        "phase_rad": float(params[3]),
        "noise_mean_estimate_v": noise_mean,
        "noise_mean_used_v": 0.0,
        "noise_sd_v": noise_sd,
        "sampling_rate_hz": float(fs),
        "baseline_window_samples": 101,
        "spike_detection_threshold_v": 3.0,
        "spike_detection_dead_time_samples": 5,
        "duration_s": float(len(raw) / fs),
        "spike_mask_pre_ms": float(mask_pre_ms),
        "spike_mask_post_ms": float(mask_post_ms),
        "n_spikes_used_for_mask": int(len(spike_idx)),
        "n_noise_samples": int(mask.sum()),
    }
    return result, drift, noise_samples, mask, covariance


def generate(t, params, seed=42):
    """Generate drift and zero-mean Gaussian noise for a supplied time axis."""
    drift = sine_function(
        t,
        params["offset_v"],
        params["amplitude_v"],
        params["frequency_hz"],
        params["phase_rad"],
    )
    rng = np.random.default_rng(seed)
    noise = rng.normal(params.get("noise_mean_used_v", 0.0),
                       params["noise_sd_v"], size=len(t))
    return drift, noise
