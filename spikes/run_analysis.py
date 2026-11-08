import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

import spikes as sp

HERE = Path(__file__).parent
FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)

t, raw = sp.load(HERE.parent / "data" / "spike_data.npz")
r = sp.estimate(t, raw)
fs, n = r["fs"], len(raw)

print(f"spikes: {len(r['peaks'])}, noise SD: {r['noise_sd']:.3f}")
print(f"tau = {r['tau_ms']:.2f} ± {r['tau_se_ms']:.2f} ms ({r['n_isolated']} isolated spikes)")
print(f"amplitude = {r['amplitude']:.2f}, SD of peaks {r['amps'].std(ddof=1):.3f}, "
      f"same-height test p = {r['amp_spread_p']:.2f}")
print(f"rate = {r['rate_hz']:.2f} Hz (95% CI {r['rate_ci'][0]:.1f}-{r['rate_ci'][1]:.1f}), "
      f"CV = {r['cv']:.2f}, KS p = {r['ks_p']:.2f}, Fano = {r['fano']:.2f}")

# 1. shape
m = np.arange(15)
fit = sp.exp_decay(m, r["wave"][3], r["tau_ms"] * fs / 1000)
fig, ax = plt.subplots(1, 2, figsize=(11, 3.5))
ax[0].plot(np.arange(-3, 15), r["wave"], "o", color="0.2", label="mean spike (data)")
ax[0].plot(m, fit, color="tab:red", label=f"a·exp(-t/τ), τ = {r['tau_ms']:.2f} ms")
ax[0].set_xlabel("ms from peak")
ax[0].legend()
ax[1].semilogy(m[:11], r["wave"][3:14], "o", color="0.2", label="data")
ax[1].semilogy(m[:11], fit[:11], color="tab:red", label="exponential")
ax[1].set_xlabel("ms from peak")
ax[1].set_title("log scale: straight line")
ax[1].legend()
fig.tight_layout()
fig.savefig(FIG / "shape.png", dpi=120)

# 2. amplitude
x = np.linspace(3.8, 6.2, 300)
fig, ax = plt.subplots(figsize=(7, 3.5))
ax.hist(r["amps"], bins=np.linspace(3.8, 6.2, 17), density=True, color="0.75", label="peak heights")
ax.plot(x, stats.norm.pdf(x, r["amplitude"], r["noise_sd"]), color="tab:red", lw=2,
        label="one height + noise")
ax.set_xlabel("height above baseline")
ax.legend()
fig.tight_layout()
fig.savefig(FIG / "amplitude.png", dpi=120)

# 3. timing
isi, lam = r["isi"], r["rate_hz"]
xs = np.linspace(0, isi.max(), 200)
fig, ax = plt.subplots(1, 2, figsize=(11, 3.5))
ax[0].hist(isi, bins=15, density=True, color="0.75", label="intervals between spikes")
ax[0].plot(xs, stats.expon.pdf(xs, scale=1 / lam), color="tab:red", lw=2, label="exponential (Poisson)")
ax[1].step(np.sort(isi), np.arange(1, len(isi) + 1) / len(isi), where="post", color="0.2", label="data")
ax[1].plot(xs, stats.expon.cdf(xs, scale=1 / lam), color="tab:red", lw=2, label="exponential")
for a in ax:
    a.set_xlabel("interval (s)")
    a.legend()
fig.tight_layout()
fig.savefig(FIG / "timing.png", dpi=120)

# 4. parameter recovery: real baseline + generated spikes + noise, then estimate again
rng = np.random.default_rng(2026)
base = raw - r["y"]
true_sets = [(4.2, 5.0, 3.0), (8.0, 4.0, 2.0), (3.0, 6.0, 5.0)]
results = []
print("\nparameter recovery (50 recordings each)")
for rate, amp, tau in true_sets:
    est = []
    for _ in range(50):
        s, idx = sp.generate(n, fs, rate, amp, tau, rng=rng)
        q = sp.estimate(t, base + s + rng.normal(0, r["noise_sd"], n))
        est.append((len(q["peaks"]) / len(idx), q["amplitude"], q["tau_ms"]))
    e = np.array(est)
    results.append(e)
    print(f"true rate {rate}, a {amp}, tau {tau}: found {e[:, 0].mean():.3f} of spikes, "
          f"a = {e[:, 1].mean():.3f} ± {e[:, 1].std():.3f}, tau = {e[:, 2].mean():.3f} ± {e[:, 2].std():.3f}")

fig, ax = plt.subplots(1, 2, figsize=(11, 3.5))
for (rate, amp, tau), e in zip(true_sets, results):
    ax[0].plot(amp, e[:, 1].mean(), "o", ms=8)
    ax[0].errorbar(amp, e[:, 1].mean(), yerr=e[:, 1].std(), color="0.3")
    ax[1].plot(tau, e[:, 2].mean(), "o", ms=8, label=f"λ={rate} Hz, a={amp}, τ={tau} ms")
    ax[1].errorbar(tau, e[:, 2].mean(), yerr=e[:, 2].std(), color="0.3")
ax[0].plot([3.5, 6.5], [3.5, 6.5], "k--", lw=1)
ax[1].plot([1.5, 5.5], [1.5, 5.5], "k--", lw=1)
ax[0].set_xlabel("true amplitude")
ax[0].set_ylabel("estimated")
ax[1].set_xlabel("true τ (ms)")
ax[1].set_ylabel("estimated")
ax[1].legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIG / "recovery.png", dpi=120)

params = {"rate_hz": round(float(r["rate_hz"]), 3), "amplitude": round(float(r["amplitude"]), 3),
          "tau_ms": round(float(r["tau_ms"]), 3), "refractory_ms": 2.0}
(HERE / "spike_params.json").write_text(json.dumps(params, indent=2))
print("\nsaved figures/ and spike_params.json:", params)
