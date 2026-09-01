"""Trajectory conditioning: reject, fill, smooth, without inventing motion."""

from __future__ import annotations

import numpy as np
import pytest

from smartdive.signal import (
    SmoothingConfig,
    condition,
    fill_gaps,
    hampel_mask,
    jitter,
    noise_scale,
)


def smooth_path(n: int = 60) -> np.ndarray:
    t = np.linspace(0, 1, n)
    return np.stack([100 + 300 * t, 200 + 400 * t**2], axis=1)


def test_hampel_finds_an_injected_spike_and_nothing_else():
    values = np.linspace(0, 10, 40)
    values[17] += 50.0

    mask = hampel_mask(values, window=5, sigma=3.0)

    assert mask[17]
    assert mask.sum() == 1


def test_hampel_leaves_clean_data_alone():
    assert not hampel_mask(np.linspace(0, 10, 40)).any()


def test_hampel_survives_missing_values():
    values = np.linspace(0, 10, 40)
    values[5:8] = np.nan
    values[20] += 40.0
    mask = hampel_mask(values)
    assert mask[20]
    assert not mask[5:8].any()


class TestGapFilling:
    def test_short_gaps_are_filled(self):
        t = np.arange(20, dtype=float)
        values = t * 2.0
        values[8:11] = np.nan

        filled, invented = fill_gaps(t, values, max_gap=5)

        assert np.allclose(filled[8:11], t[8:11] * 2.0)
        assert invented[8:11].all()

    def test_long_gaps_are_left_alone(self):
        """A pipeline that invents a second of missing trajectory is lying."""
        t = np.arange(20, dtype=float)
        values = t * 2.0
        values[5:15] = np.nan

        filled, invented = fill_gaps(t, values, max_gap=5)

        assert np.isnan(filled[5:15]).all()
        assert not invented.any()

    def test_gaps_at_the_ends_are_never_extrapolated(self):
        t = np.arange(20, dtype=float)
        values = t * 2.0
        values[:3] = np.nan
        values[-3:] = np.nan

        filled, invented = fill_gaps(t, values, max_gap=5)

        assert np.isnan(filled[:3]).all()
        assert np.isnan(filled[-3:]).all()
        assert not invented.any()

    def test_monotone_interpolation_does_not_overshoot(self):
        t = np.arange(11, dtype=float)
        values = np.array([0, 0, 0, 0, 0, np.nan, 10, 10, 10, 10, 10], dtype=float)
        filled, _ = fill_gaps(t, values, max_gap=3)
        assert 0.0 <= filled[5] <= 10.0


class TestConditioning:
    def test_savgol_order_two_preserves_constant_acceleration(self):
        """Smoothing must not bias the gravity estimate the physics depends on."""
        frames = np.arange(60)
        t = frames / 30.0
        xy = np.stack([50 * t, 100 + 20 * t + 0.5 * 900 * t**2], axis=1)

        out = condition(frames, xy, 30.0)

        assert np.allclose(out.xy, xy, atol=1e-6)

    def test_smoothing_reduces_jitter(self):
        rng = np.random.default_rng(0)
        clean = smooth_path()
        noisy = clean + rng.normal(0, 4.0, clean.shape)

        out = condition(np.arange(len(noisy)), noisy, 30.0)

        assert jitter(out.xy) < jitter(noisy) / 2

    def test_a_spike_is_rejected_and_then_filled(self):
        xy = smooth_path()
        xy[25] += [200.0, -150.0]

        out = condition(np.arange(len(xy)), xy, 30.0)

        assert out.outliers[25]
        assert out.interpolated[25]
        assert np.linalg.norm(out.xy[25] - smooth_path()[25]) < 20.0

    def test_coverage_counts_original_measurements_only(self):
        xy = smooth_path(50)
        xy[10:14] = np.nan

        out = condition(np.arange(50), xy, 30.0)

        assert out.coverage == pytest.approx(46 / 50)
        assert out.interpolated[10:14].all()  # filled, but not counted as measured

    def test_derivatives_do_not_bridge_unfilled_gaps(self):
        xy = smooth_path(60)
        xy[20:40] = np.nan  # longer than max_gap, so it stays missing

        out = condition(np.arange(60), xy, 30.0, SmoothingConfig(max_gap=4))

        assert np.isnan(out.velocity()[20:40, 0]).all()

    def test_velocity_matches_a_known_constant(self):
        frames = np.arange(60)
        t = frames / 30.0
        xy = np.stack([120.0 * t, np.zeros_like(t)], axis=1)

        out = condition(frames, xy, 30.0)

        assert np.nanmedian(out.velocity()[:, 0]) == pytest.approx(120.0, rel=1e-6)


def test_noise_scale_recovers_the_injected_sigma():
    """The estimator the flight search leans on for its inlier tolerance."""
    rng = np.random.default_rng(7)
    clean = smooth_path(400)
    sigma = 5.0
    noisy = clean + rng.normal(0, sigma, clean.shape)

    # jitter() is the 2-D second-difference RMS, so the recovered scale is the
    # radial noise magnitude, sqrt(2) larger than the per-axis sigma.
    assert noise_scale(noisy) == pytest.approx(sigma * np.sqrt(2), rel=0.15)


def test_jitter_is_nan_when_there_is_nothing_to_measure():
    assert np.isnan(jitter(np.full((3, 2), np.nan)))
