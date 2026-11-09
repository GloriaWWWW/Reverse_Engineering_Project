"""Integrate drift/noise and spike team models; simulate and validate."""
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from drift_noise.drift_noise import generate as generate_background, sine_function, make_spike_mask
from spikes.spikes import generate as generate_spikes, detect, baseline

HERE = Path(__file__).resolve().parent
FIG = HERE / 'figures'
FIG.mkdir(exist_ok=True)


def get_spike_peaks(raw):
    # Identical algorithm for real and simulated recordings.
    return detect(raw - baseline(raw))


def metrics(raw, t, drift_model, fs):
    peaks = get_spike_peaks(raw)
    residual = raw - drift_model
    mask = make_spike_mask(len(raw), peaks, fs, pre_ms=3, post_ms=20)
    height = residual[peaks] if len(peaks) else np.array([])
    isi = np.diff(t[peaks]) * 1000
    return {
        'spike_count': int(len(peaks)),
        'firing_rate_hz': len(peaks) / (len(t) / fs),
        'mean_spike_height_v': float(np.mean(height)) if len(height) else float('nan'),
        'median_isi_ms': float(np.median(isi)) if len(isi) else float('nan'),
        'noise_sd_v': float(np.std(residual[mask], ddof=0)),
        'signal_min_v': float(np.min(raw)),
        'signal_max_v': float(np.max(raw)),
    }, peaks


def main(n_simulations=300, seed=42):
    with np.load(ROOT / 'data' / 'spike_data.npz') as arr:
        t = arr['t'].astype(float)
        real = arr['raw_data'].astype(float)
    fs = 1 / np.median(np.diff(t))
    bg = json.loads((ROOT / 'drift_noise' / 'drift_noise_params.json').read_text())
    sp = json.loads((ROOT / 'spikes' / 'spike_params.json').read_text())
    if abs(float(bg['sampling_rate_hz']) - fs) > 0.01:
        raise ValueError('Background sampling rate mismatches recording')

    drift, _ = generate_background(t, bg, seed=seed)
    real_stats, real_peaks = metrics(real, t, drift, fs)

    # Separate deterministic per-trial seeds. Spike and noise RNGs are independent.
    def one_simulation(i):
        d, noise = generate_background(t, bg, seed=seed + 100000 + i)
        spike_signal, true_idx = generate_spikes(
            len(t), fs, sp['rate_hz'], sp['amplitude'], sp['tau_ms'],
            sp['refractory_ms'], rng=seed + i,
        )
        simulated = d + noise + spike_signal
        stats, detected_idx = metrics(simulated, t, d, fs)
        stats['trial'] = i + 1
        stats['generated_spikes'] = len(true_idx)
        # A detector peak may lag the true onset by a sample or so.
        tol_samples = max(1, int(round(0.002 * fs)))
        matched_truth = set()
        n_match = 0
        for peak in detected_idx:
            available = [(abs(int(peak) - int(truth)), j) for j, truth in enumerate(true_idx)
                         if j not in matched_truth and abs(int(peak) - int(truth)) <= tol_samples]
            if available:
                _, j = min(available)
                matched_truth.add(j)
                n_match += 1
        stats['precision'] = n_match / len(detected_idx) if len(detected_idx) else float('nan')
        stats['recall'] = n_match / len(true_idx) if len(true_idx) else float('nan')
        return simulated, stats

    first_signal, first_stats = one_simulation(0)
    all_stats = [first_stats]
    for i in range(1, n_simulations):
        _, stats = one_simulation(i)
        all_stats.append(stats)

    cols = list(all_stats[0].keys())
    with (HERE / 'simulation_results.csv').open('w', newline='') as out:
        writer = csv.DictWriter(out, fieldnames=cols)
        writer.writeheader()
        writer.writerows(all_stats)

    summary = {}
    print(f'Comparing one simulation and {n_simulations} repeated simulations')
    print(f"{'Metric':24} {'Real':>11} {'One sim':>11} {'Mean 300':>12} {'95% range':>22}")
    for metric in ['spike_count','firing_rate_hz','mean_spike_height_v','median_isi_ms','noise_sd_v','signal_min_v','signal_max_v']:
        values = np.asarray([s[metric] for s in all_stats], dtype=float)
        average = float(np.nanmean(values))
        lo, hi = np.nanpercentile(values, [2.5, 97.5])
        summary[metric] = {'real': real_stats[metric], 'single_simulation': first_stats[metric],
                           'mean': average, 'p2_5': float(lo), 'p97_5': float(hi)}
        print(f'{metric:24} {real_stats[metric]:11.3f} {first_stats[metric]:11.3f} {average:12.3f} [{lo:8.3f}, {hi:8.3f}]')
    summary['n_simulations'] = n_simulations
    summary['first_generated_spike_count'] = first_stats['generated_spikes']
    summary['mean_precision'] = float(np.nanmean([s['precision'] for s in all_stats]))
    summary['mean_recall'] = float(np.nanmean([s['recall'] for s in all_stats]))
    summary['notes'] = '95% ranges are empirical simulation quantiles, not confidence intervals for parameters.'
    (HERE / 'summary.json').write_text(json.dumps(summary, indent=2))
    np.savez(HERE / 'simulated_spike_data.npz', t=t, raw_data=first_signal)

    # Plot original and simulated signals separately, with consistent axes and distinct colors.
    for x, label, color, filename, count in [
        (real, 'Original', '#2563EB', 'original_recording.png', real_stats['spike_count']),
        (first_signal, 'Simulated', '#EA580C', 'simulated_recording.png', first_stats['spike_count']),
    ]:
        fig, ax = plt.subplots(figsize=(12, 4.4))
        fig.patch.set_facecolor('white')
        ax.set_facecolor('#FAFBFD')
        ax.plot(t, x, color=color, lw=.83)
        ax.set(title=f'{label} Spike Recording  |  {count} detected spikes',
               xlabel='Time (s)', ylabel='Voltage (V)', xlim=(t[0], t[-1]), ylim=(-2, 8))
        ax.grid(alpha=.17, linestyle='--')
        ax.spines[['top','right']].set_visible(False)
        fig.tight_layout()
        fig.savefig(FIG / filename, dpi=190, bbox_inches='tight')
        plt.close(fig)

    # Two-trace comparison for users who prefer to view the signals together.
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    axes[0].plot(t, real, lw=.8, color='#2563EB')
    axes[0].set(title=f'Original recording ({real_stats["spike_count"]} spikes)', ylabel='Voltage (V)')
    axes[1].plot(t, first_signal, lw=.8, color='#EA580C')
    axes[1].set(title=f'Simulated recording ({first_stats["spike_count"]} spikes)',
                xlabel='Time (s)', ylabel='Voltage (V)')
    for ax in axes:
        ax.grid(alpha=.17, linestyle='--')
        ax.set_ylim(-2, 8)
        ax.spines[['top','right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / 'real_vs_simulated.png', dpi=190, bbox_inches='tight')
    plt.close(fig)

    counts = np.asarray([s['spike_count'] for s in all_stats])
    fig, ax = plt.subplots(figsize=(10, 4.7))
    ax.set_facecolor('#FAFBFD')
    ax.hist(counts, bins=np.arange(min(counts)-.5, max(counts)+1.5),
            color='#6474C9', edgecolor='white', alpha=.9)
    ax.axvline(real_stats['spike_count'], color='#DC2626', ls='--', lw=2.3,
               label=f'Original: {real_stats["spike_count"]}')
    ax.axvline(np.mean(counts), color='#15803D', ls='--', lw=2.3,
               label=f'Simulation mean: {np.mean(counts):.2f}')
    ax.set(xlabel='Detected spike count (10 s)', ylabel='Number of simulations',
           title=f'Spike Count Distribution  |  {n_simulations} Simulations')
    ax.grid(axis='y', alpha=.17, linestyle='--')
    ax.spines[['top','right']].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / 'spike_count_distribution.png', dpi=190, bbox_inches='tight')
    plt.close(fig)
    print(f'\nMean precision: {summary["mean_precision"]:.3f}; mean recall: {summary["mean_recall"]:.3f}')
    print(f'Saved results to {HERE}')


if __name__ == '__main__':
    main()
