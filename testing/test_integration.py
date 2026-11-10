"""Run from the project root: python -m unittest testing.test_integration"""
import json
import unittest
from pathlib import Path
import numpy as np

from drift_noise.drift_noise import generate as generate_background
from spikes.spikes import generate as generate_spikes, detect, baseline

ROOT = Path(__file__).resolve().parents[1]

class IntegratedSimulationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with np.load(ROOT / 'data' / 'spike_data.npz') as d:
            cls.t = d['t']
            cls.raw = d['raw_data']
        cls.fs = 1 / np.median(np.diff(cls.t))
        cls.bg = json.loads((ROOT / 'drift_noise/drift_noise_params.json').read_text())
        cls.sp = json.loads((ROOT / 'spikes/spike_params.json').read_text())

    def test_compatible_lengths(self):
        drift, noise = generate_background(self.t, self.bg, seed=42)
        spikes, idx = generate_spikes(len(self.t), self.fs, self.sp['rate_hz'],
                                      self.sp['amplitude'], self.sp['tau_ms'],
                                      self.sp['refractory_ms'], rng=42)
        self.assertEqual(drift.shape, self.raw.shape)
        self.assertEqual(noise.shape, self.raw.shape)
        self.assertEqual(spikes.shape, self.raw.shape)
        self.assertTrue(np.all(np.isfinite(drift + noise + spikes)))

    def test_repeatable_seed(self):
        a = generate_background(self.t, self.bg, seed=42)[1]
        b = generate_background(self.t, self.bg, seed=42)[1]
        self.assertTrue(np.array_equal(a, b))
        sa, ia = generate_spikes(len(self.t), self.fs, self.sp['rate_hz'],
                                 self.sp['amplitude'], self.sp['tau_ms'],
                                 self.sp['refractory_ms'], rng=42)
        sb, ib = generate_spikes(len(self.t), self.fs, self.sp['rate_hz'],
                                 self.sp['amplitude'], self.sp['tau_ms'],
                                 self.sp['refractory_ms'], rng=42)
        self.assertTrue(np.array_equal(sa, sb))
        self.assertTrue(np.array_equal(ia, ib))

    def test_real_spike_detector(self):
        self.assertEqual(len(detect(self.raw-baseline(self.raw))), 42)

    def test_refractory(self):
        _, idx = generate_spikes(len(self.t), self.fs, self.sp['rate_hz'],
                                 self.sp['amplitude'], self.sp['tau_ms'],
                                 self.sp['refractory_ms'], rng=42)
        if len(idx) > 1:
            # Integer sample binning may floor a strict continuous-time gap.
            self.assertGreaterEqual(np.diff(idx).min(), int(np.floor(self.sp['refractory_ms']*self.fs/1000)))

if __name__ == '__main__':
    unittest.main()
