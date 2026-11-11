import numpy as np
from scipy import signal, stats
import matplotlib.pyplot as plt

def evaluate(reference_file, candidate_file):
    # Load datasets
    ref = np.load(reference_file, allow_pickle=True)
    cand = np.load(candidate_file, allow_pickle=True)

    ref_raw = ref['raw_data'].ravel()
    cand_raw = cand['reconstruction'].ravel()

    # Metric 1: Correlation
    r_val = np.corrcoef(ref_raw, cand_raw)[0, 1]

    # Metric 2: Spectral overlap
    fs = 1000.0
    _, psd_ref = signal.welch(ref_raw, fs=fs, nperseg=512)
    _, psd_cand = signal.welch(cand_raw, fs=fs, nperseg=512)
    spectral_score = np.dot(psd_ref, psd_cand) / (np.linalg.norm(psd_ref) * np.linalg.norm(psd_cand) + 1e-12)

    # Scoring math
    final_score = 100.0 * (0.5 * max(0, r_val) + 0.5 * spectral_score)

    return {
        "validity_check": "PASS" if len(ref_raw) == len(cand_raw) else "FAIL",
        "correlation": r_val,
        "spectral_cosine_similarity": spectral_score,
        "final_score": round(final_score, 2)
    }
