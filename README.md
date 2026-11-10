# Spike recording: reverse engineering, simulation

Course project for **Data Science & Advanced Python**.
Team: Mariya Zastenskaya, Gloria, Amarnath.

**Final Working Code:** **[Python_Reverse_Eng.ipynb](Python_Reverse_Eng.ipynb)**.

The final notebook contains the signal analysis, reconstruction, and evaluation code for the inter-team competition.




We find the process that generated `data/spike_data.npz` (10 s, 1000 Hz), simulate it, extend it by 10 s,
and score how close any simulation is to the original.

## The model

```
raw(t) = c + A·sin(2πft + φ)                      slow drift
       + Σ_k a·exp(-(t - t_k)/τ)·[t ≥ t_k]        spikes: rise within 1 sample, exponential decay
       + ε(t),  ε ~ N(0, σ²) independent          white Gaussian noise
t_k: Poisson process, rate λ, refractory period r, on the sample grid
```

| Parameter | Estimate | Source |
|---|---|---|
| drift `c, A, f, φ` | 0.003, 0.647, 0.0908 Hz (period 11.0 s), 0.29 rad | data (joint least squares) |
| spike height `a` | 5.02, the same for every spike | data |
| spike decay `τ` | 2.98 ms | data |
| firing rate `λ` | 4.2 Hz (95 % CI 3.0–5.7) | data |
| noise `σ` | 0.299 | data |
| refractory period `r` | 2 ms | **assumption** (data: < 11 ms) |
| drift keeps the same sinusoid after 10 s | — | **assumption** |

The evidence for every component, and why we did not use the other models proposed in class, is in the notebook
(section 4).

## Files

| File | What |
|---|---|
| `Python_Reverse_Eng.ipynb` | Colab notebook: all code, the evidence, plots, calibration |
| `spike_simulator.py` | the model: `fit_model`, `simulate`, `make_results` |
| `testing_module.py` | **testing module**: how close is a simulation to the original (score 0–100) |
| `results/simulated_spike_data.npz` | **main result**: a fully simulated 20 s recording, same structure as the original |
| `results/extended_spike_data.npz` | the original 10 s + the simulated 10 s continuation |
| `results/simulated_ground_truth.npz` | true spike times and components of the simulation |
| `results/fitted_parameters.json` | fitted parameters, their sources, the seed |
| `tests/` | pytest tests for both modules |
| `tools/build_notebook.py` | rebuilds the notebook from the `.py` modules |

Both results files contain exactly the arrays of the original (`t`, `raw_data`, 1-D float64, dt = 1 ms, t[0] = 0).
Only the length is 20 s.

## Quick start

**Colab:** open the notebook (badge above) → *Runtime → Run all* (about 1 minute). It downloads the data itself.

**Locally:**
```bash
pip install -r requirements.txt
python spike_simulator.py data/spike_data.npz --out results --extra 10 --seed 2026
python testing_module.py data/spike_data.npz results/simulated_spike_data.npz
python -m pytest -q
```

## Testing module

```bash
python testing_module.py ORIGINAL.npz SIM_1.npz [SIM_2.npz ...]     # several files → ranking
```
```python
from testing_module import compare_files
report = compare_files("data/spike_data.npz", "results/simulated_spike_data.npz")
print(report)          # per-property table + overall score
```

* Checks the file structure first (`t`, `raw_data`, 1-D, uniform sampling, same sampling rate).
* Puts both recordings through **the same analysis** and compares 8 properties with statistical tests:
  spike rate, ISI distribution, amplitude distribution, waveform shape, noise level, noise autocorrelation,
  noise distribution shape, and the drift (time-aligned).
* Each test's p-value becomes an equivalent |z|; similarity `s = exp(-½(z/1.96)²)`;
  score = 100 × weighted geometric mean of `s` (spikes 50 %, noise 25 %, drift 25 %).
* Calibration (notebook, section 8): two independent recordings of the same process score about 87 ± 7,
  our simulation vs the original about 91 ± 5, and models with one wrong component 16–54.
  A perfect simulator does not get 100, because two random recordings always differ a little.
* The simulation may be longer than the original. Spikes and noise are compared over the whole files,
  the drift on the common time range.

## Team workflow

* The `.py` files are the source of truth. After changing one, rebuild the notebook:
  `python tools/build_notebook.py --execute`, then commit both.
* One branch per task → Pull Request → another team member reviews → merge into `main`.
* Run `python -m pytest -q` before opening a PR.

| Part | Owner | Files / notebook sections |
|---|---|---|
| Drift and noise: estimation, evidence, the "data vs assumption" table |Gloria | `fit_model`, `drift`; sections 3, 4.1, 4.5, 4.6 |
| Spikes: template, amplitude, timing, parameter recovery | Mariya Zastenskaya | `spike_kernel`, `render_spikes`, `simulate_spike_samples`; sections 4.2–4.4, 5 |
| Testing module, calibration, results files | Amarnath | `testing_module.py`, `tests/`, `results/`; sections 7, 8 |
