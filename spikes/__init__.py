from pathlib import Path
import numpy as np


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
