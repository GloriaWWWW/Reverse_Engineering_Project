import os
import json
import numpy as np

# Attempt to import your existing simulation functions from the project modules.
# Adjust the function names below if your drift_noise.py or spikes.py use different names.
try:
    from drift_noise.drift_noise import simulate_drift_noise
except ImportError:
    simulate_drift_noise = None

try:
    from spikes.spikes import simulate_spikes
except ImportError:
    simulate_spikes = None


def load_json_params(filepath):
    """Load parameters from a JSON file if it exists."""
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            return json.load(f)
    print(f"Warning: {filepath} not found. Using default fallback parameters.")
    return {}


def fallback_drift_noise(t, params):
    """Fallback generator for baseline drift + noise if module function isn't linked."""
    # Common parameter keys: 'drift_amp', 'drift_freq', 'drift_slope', 'noise_std'
    slope = params.get("slope", params.get("drift_slope", 0.0))
    amp = params.get("amplitude", params.get("drift_amp", 1.0))
    freq = params.get("frequency", params.get("drift_freq", 0.5))
    phase = params.get("phase", 0.0)
    offset = params.get("offset", params.get("mean", 0.0))
    noise_std = params.get("noise_std", params.get("std", 0.2))

    drift = offset + slope * t + amp * np.sin(2 * np.pi * freq * t + phase)
    noise = np.random.normal(0, noise_std, size=len(t))
    return drift + noise


def fallback_spikes(t, params):
    """Fallback generator for spike trains if module function isn't linked."""
    dt = t[1] - t[0] if len(t) > 1 else 1e-4
    n_samples = len(t)
    signal = np.zeros(n_samples)

    # Timing parameters (e.g., Poisson rate or refractory period)
    rate = params.get("firing_rate", params.get("rate", 10.0))  # Hz
    refractory = params.get("refractory_period", params.get("recovery_tau", 0.005))  # sec
    amp_mean = params.get("amplitude_mean", params.get("amplitude", 5.0))
    amp_std = params.get("amplitude_std", 0.5)
    tau_rise = params.get("tau_rise", 0.0005)
    tau_decay = params.get("tau_decay", 0.002)

    # Generate spike times via Poisson process with refractory period
    spike_indices = []
    current_t = t[0]
    while current_t < t[-1]:
        isi = np.random.exponential(1.0 / rate) + refractory
        current_t += isi
        if current_t < t[-1]:
            idx = int((current_t - t[0]) / dt)
            if 0 <= idx < n_samples:
                spike_indices.append(idx)

    # Construct spike waveform kernel
    kernel_len = int(0.02 / dt)  # 20 ms window
    t_kernel = np.arange(kernel_len) * dt
    waveform = (1 - np.exp(-t_kernel / tau_rise)) * np.exp(-t_kernel / tau_decay)
    if np.max( np.abs(waveform) ) > 0:
        waveform = waveform / np.max(np.abs(waveform))

    # Superimpose spikes onto signal
    for idx in spike_indices:
        amp = np.random.normal(amp_mean, amp_std)
        end_idx = min(idx + kernel_len, n_samples)
        seg_len = end_idx - idx
        signal[idx:end_idx] += amp * waveform[:seg_len]

    return signal, np.array(spike_indices)


def generate_simulated_dataset(
    original_npz_path="data/spike_data.npz",
    drift_params_path="drift_noise/drift_noise_params.json",
    spike_params_path="spikes/spike_params.json",
    output_npz_path="results/spike_data.npz",
    seed=42
):
    """
    Generates a simulated dataset matching the exact keys, shapes, and dtypes
    of the original data/spike_data.npz file.
    """
    np.random.seed(seed)

    if not os.path.exists(original_npz_path):
        raise FileNotFoundError(f"Original data file not found at: {original_npz_path}")

    orig_data = np.load(original_npz_path, allow_pickle=True)
    orig_keys = list(orig_data.files)
    print(f"Original .npz keys detected: {orig_keys}")
    for k in orig_keys:
        print(f"  - {k}: shape={orig_data[k].shape}, dtype={orig_data[k].dtype}")

    drift_params = load_json_params(drift_params_path)
    spike_params = load_json_params(spike_params_path)

    # Identify time vector or infer from signal length and sampling rate
    time_key = next((k for k in orig_keys if k.lower() in ["t", "time", "times", "timestamps"]), None)
    signal_key = next((k for k in orig_keys if k.lower() in ["data", "signal", "voltage", "v", "trace", "traces", "y"]), None)

    if signal_key is None:
        # Default to the largest array if standard naming isn't used
        signal_key = max(orig_keys, key=lambda k: orig_data[k].size)

    orig_signal = orig_data[signal_key]
    n_samples = orig_signal.shape[-1]

    if time_key is not None:
        t = orig_data[time_key]
    else:
        fs = drift_params.get("fs", spike_params.get("fs", 10000.0))
        t = np.arange(n_samples) / fs

    # Helper to generate a single 1D trace
    def simulate_single_trace(t_vec):
        if simulate_drift_noise is not None:
            bg = simulate_drift_noise(t_vec, drift_params)
        else:
            bg = fallback_drift_noise(t_vec, drift_params)

        if simulate_spikes is not None:
            spk_out = simulate_spikes(t_vec, spike_params)
            spk_sig = spk_out[0] if isinstance(spk_out, tuple) else spk_out
        else:
            spk_sig, _ = fallback_spikes(t_vec, spike_params)

        return bg + spk_sig

    # Match 1D or 2D (multi-trial/multi-channel) structure
    if orig_signal.ndim == 1:
        sim_signal = simulate_single_trace(t).astype(orig_signal.dtype)
    elif orig_signal.ndim == 2:
        sim_signal = np.zeros_like(orig_signal)
        for i in range(orig_signal.shape[0]):
            sim_signal[i] = simulate_single_trace(t).astype(orig_signal.dtype)
    else:
        raise ValueError(f"Unsupported signal dimension: {orig_signal.ndim}")

    # Build output dictionary with identical structure to original file
    output_dict = {}
    for k in orig_keys:
        if k == signal_key:
            output_dict[k] = sim_signal
        elif k == time_key:
            output_dict[k] = orig_data[k].copy()
        else:
            # Copy metadata keys (e.g., fs, dt, channel_names) directly to preserve structure
            output_dict[k] = orig_data[k].copy()

    os.makedirs(os.path.dirname(output_npz_path), exist_ok=True)
    np.savez(output_npz_path, **output_dict)
    print(f"Successfully saved simulated data to: {output_npz_path}")
    return output_dict


if __name__ == "__main__":
    generate_simulated_dataset()
