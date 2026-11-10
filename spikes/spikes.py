import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy import optimize, stats

THR = 3.0   # spike threshold above the baseline
DEAD = 5    # dead time, samples


from pathlib import Path

def load(filepath):
    """Load time and raw signal arrays from spike_data.npz."""
    filepath = Path(filepath)
    with np.load(filepath, allow_pickle=True) as data:
        files = list(data.files)

        # Find time vector
        time_key = next(
            (k for k in files if k.lower() in ["t", "time", "times", "timestamps"]),
            None,
        )
        # Find signal/voltage vector
        signal_key = next(
            (
                k
                for k in files
                if k.lower() in ["data", "signal", "voltage", "v", "trace", "traces", "raw", "y"]
            ),
            None,
        )

        if signal_key is None:
            # Fallback to the array with the largest element count
            signal_key = max(files, key=lambda k: data[k].size)

        raw = data[signal_key]

        if time_key is not None:
            t = data[time_key]
        else:
            # Default to sample index or 10 kHz default if time vector is not explicitly saved
            t = np.arange(raw.shape[-1]) / 10000.0

    return t, raw

def running_median(x, w=101):
    padded = np.pad(x, w // 2, mode="reflect")
    return np.nanmedian(sliding_window_view(padded, w), axis=1)


def detect(y):
    above = y > THR
    onsets = np.flatnonzero(above & ~np.r_[False, above[:-1]])
    peaks, last = [], -DEAD
    for i in onsets:
        if i - last >= DEAD:
            peaks.append(i + np.argmax(y[i:i + DEAD]))
            last = i
    return np.array(peaks)


def baseline(raw):
    masked = raw.copy()
    for s in detect(raw - running_median(raw)):
        masked[max(s - 3, 0):s + 20] = np.nan      # leave the spikes out
    b = running_median(masked)
    bad = np.isnan(b)
    b[bad] = np.interp(np.flatnonzero(bad), np.flatnonzero(~bad), b[~bad])
    return b


def isolated(peaks, y, after=0, before=20):
    prev = np.diff(peaks, prepend=-10**9)
    nxt = np.diff(peaks, append=10**9)
    merged = np.array([np.max(y[max(s - 5, 0):s], initial=-np.inf) > THR for s in peaks])
    return peaks[(prev > before) & (nxt > after) & ~merged]


def exp_decay(m, a, tau):
    return a * np.exp(-m / tau)


def estimate(t, raw):
    fs = 1 / np.median(np.diff(t))
    T = len(raw) / fs
    y = raw - baseline(raw)
    peaks = detect(y)

    noise = np.ones(len(y), bool)
    for s in peaks:
        noise[max(s - 3, 0):s + 20] = False
    noise_sd = y[noise].std()

    # shape
    iso = isolated(peaks, y, after=15)
    iso = iso[(iso >= 3) & (iso + 15 < len(y))]
    wave = y[iso[:, None] + np.arange(-3, 15)].mean(axis=0)
    (_, tau), cov = optimize.curve_fit(exp_decay, np.arange(15), wave[3:], p0=[5, 3])

    # amplitude
    amps = y[isolated(peaks, y)]
    chi2 = (len(amps) - 1) * amps.var(ddof=1) / noise_sd**2

    # timing
    k = len(peaks)
    isi = np.diff(peaks) / fs
    counts = np.bincount((peaks / fs).astype(int), minlength=int(T))

    return {
        "fs": fs, "y": y, "peaks": peaks, "noise_sd": noise_sd,
        "wave": wave, "n_isolated": len(iso),
        "tau_ms": tau / fs * 1000, "tau_se_ms": np.sqrt(cov[1, 1]) / fs * 1000,
        "amps": amps, "amplitude": amps.mean(),
        "amp_spread_p": stats.chi2.sf(chi2, len(amps) - 1),
        "rate_hz": k / T,
        "rate_ci": (stats.chi2.ppf(0.025, 2 * k) / (2 * T), stats.chi2.ppf(0.975, 2 * k + 2) / (2 * T)),
        "isi": isi, "cv": isi.std(ddof=1) / isi.mean(),
        "ks_p": stats.kstest(isi, "expon", args=(0, isi.mean())).pvalue,
        "counts": counts, "fano": counts.var(ddof=1) / counts.mean(),
    }


def generate(n, fs, rate_hz, amplitude, tau_ms, refractory_ms=2.0, rng=None):
    rng = np.random.default_rng(rng)
    ref = refractory_ms / 1000
    times, t_now = [], rng.exponential(1 / rate_hz - ref)
    while t_now < n / fs:
        times.append(t_now)
        t_now += ref + rng.exponential(1 / rate_hz - ref)
    idx = (np.array(times) * fs).astype(int)

    tau = tau_ms * fs / 1000
    kernel = np.exp(-np.arange(int(20 * tau)) / tau)
    impulses = np.zeros(n)
    impulses[idx] = amplitude
    return np.convolve(impulses, kernel)[:n], idx
