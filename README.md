# Team Reconstruction Submission README

### Inferred Model & Parameters
- **Baseline Drift Model:** Low-frequency drift extracted via a 3rd-order zero-phase lowpass Butterworth filter with a 15 Hz cutoff.
- **Background Noise:** Structured colored Gaussian noise model synthesized using Fourier Phase synthesis to replicate PSD exactly (SD = 0.298).
- **Spike Model:** Biophysically constrained Gamma-Interval renewal process ($a=0.56, \beta=0.36$) matching an empirical firing rate of 4.70 Hz.

### Assumptions
- Spikes follow a stationary distribution with biphasic depolarization and negative hyperpolarization recovery phases.

### Reconstruction Method
- Matched-filtering and peak-finding for event alignment combined with an iterative amplitude recovery regression.
