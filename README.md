# Reverse Engineering Project

Reverse engineering of `data/spike_data.npz` (10 s at 1000 Hz): 
simulate the recording and extend it by 10 s, save the result in the same structure as the original,
and test how close the simulation is to the original.

| Folder | Content | Owner              |
|---|---|--------------------|
| `data/` | original recording `spike_data.npz` | —                  |
| `spikes/` | spike shape, amplitude, timing, parameter recovery | Mariya Zastenskaya |
| `drift_noise/` | baseline drift and noise | Gloria             |
| `testing/` | testing module: how close the simulation is to the original | Amarnath           |
| `results/` | final simulated recording (same structure as the original: `t`, `raw_data`) | team               |

The final simulation adds the three parts together:
`raw = drift + noise + spikes`.
