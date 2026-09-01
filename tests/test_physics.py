"""Physics: does the ballistic model recover what we put into it?"""

from __future__ import annotations

import numpy as np
import pytest

from smartdive.physics import (
    FlightSearchConfig,
    G,
    bootstrap_uncertainty,
    detect_flight,
    fit_ballistic,
)

FPS = 120.0
PX_PER_M = 200.0
#: A 1.75 m athlete at this scale spans roughly this many pixels.
BODY_SCALE = 200.0


def synthetic_flight(
    n: int = 40,
    vx_ms: float = 5.0,
    vy_ms: float = 1.5,
    fps: float = FPS,
    px_per_m: float = PX_PER_M,
    noise: float = 0.0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """A projectile trajectory in image coordinates (y grows downward)."""
    rng = np.random.default_rng(seed)
    frames = np.arange(n)
    t = frames / fps
    x = 500.0 + vx_ms * px_per_m * t
    y = 400.0 - vy_ms * px_per_m * t + 0.5 * G * px_per_m * t**2
    xy = np.stack([x, y], axis=1)
    if noise:
        xy = xy + rng.normal(0.0, noise, xy.shape)
    return frames, xy


def test_fit_recovers_launch_velocity_exactly_when_noiseless():
    frames, xy = synthetic_flight(vx_ms=5.0, vy_ms=1.5)
    fit = fit_ballistic(frames, xy, FPS)

    assert fit is not None
    assert fit.vx / PX_PER_M == pytest.approx(5.0, abs=1e-6)
    assert -fit.vy / PX_PER_M == pytest.approx(1.5, abs=1e-6)
    assert fit.residual_rms == pytest.approx(0.0, abs=1e-6)
    assert fit.r_squared == pytest.approx(1.0, abs=1e-9)


def test_fitted_acceleration_recovers_the_scene_scale():
    """The whole point of the gravity calibration: a_y / g is px per metre."""
    frames, xy = synthetic_flight(px_per_m=173.0)
    fit = fit_ballistic(frames, xy, FPS)

    assert fit is not None
    assert fit.ay / G == pytest.approx(173.0, rel=1e-6)


def test_takeoff_angle_is_signed_upward_and_direction_agnostic():
    _, rightward = synthetic_flight(vx_ms=5.0, vy_ms=1.5)
    frames = np.arange(len(rightward))
    leftward = rightward.copy()
    leftward[:, 0] = 2000.0 - leftward[:, 0]  # mirror horizontally

    a = fit_ballistic(frames, rightward, FPS)
    b = fit_ballistic(frames, leftward, FPS)

    assert a is not None and b is not None
    assert a.angle_deg > 0  # launched upward
    assert a.angle_deg == pytest.approx(b.angle_deg, abs=1e-6)


def test_a_free_fit_cannot_measure_gravity_from_a_short_noisy_window():
    """The motivating measurement: this is why ay_prior exists at all."""
    errors = []
    for seed in range(12):
        frames, xy = synthetic_flight(n=12, noise=6.0, seed=seed)
        fit = fit_ballistic(frames, xy, FPS)
        assert fit is not None
        errors.append(abs(fit.ay / G - PX_PER_M) / PX_PER_M)

    # Typical error in the scale is tens of percent, not a fraction of one.
    assert np.median(errors) > 0.10


def test_constraining_gravity_improves_the_vertical_velocity():
    """Fixing a_y frees the fit from trading curvature against launch speed."""
    free_errors, fixed_errors = [], []
    for seed in range(12):
        frames, xy = synthetic_flight(n=12, noise=6.0, seed=seed)
        free = fit_ballistic(frames, xy, FPS)
        fixed = fit_ballistic(frames, xy, FPS, ay=G * PX_PER_M)
        assert free is not None and fixed is not None
        assert fixed.ay == pytest.approx(G * PX_PER_M)
        target = -1.5 * PX_PER_M
        free_errors.append(abs(free.vy - target))
        fixed_errors.append(abs(fixed.vy - target))

    assert np.median(fixed_errors) < np.median(free_errors)


def test_the_horizontal_fit_is_unaffected_by_the_gravity_constraint():
    """x(t) is linear either way, so constraining a_y must not disturb it."""
    frames, xy = synthetic_flight(n=12, noise=6.0, seed=3)
    free = fit_ballistic(frames, xy, FPS)
    fixed = fit_ballistic(frames, xy, FPS, ay=G * PX_PER_M)
    assert free is not None and fixed is not None
    assert fixed.vx == pytest.approx(free.vx)


def test_fit_ignores_missing_samples():
    frames, xy = synthetic_flight(n=30)
    xy[5:9] = np.nan

    fit = fit_ballistic(frames, xy, FPS)

    assert fit is not None
    assert fit.n_points == 26
    assert fit.residual_rms == pytest.approx(0.0, abs=1e-6)


def test_fit_returns_none_when_there_is_nothing_to_fit():
    frames = np.arange(10)
    xy = np.full((10, 2), np.nan)
    assert fit_ballistic(frames, xy, FPS) is None


def test_apex_and_duration():
    frames, xy = synthetic_flight(n=60, vy_ms=2.0)
    fit = fit_ballistic(frames, xy, FPS)

    assert fit is not None
    # Time to apex is v_up / g, independent of scale.
    assert fit.apex_time() == pytest.approx(2.0 / G, rel=1e-6)
    assert fit.duration == pytest.approx(59 / FPS)


class TestFlightDetection:
    """detect_flight must find flight embedded in non-flight."""

    @staticmethod
    def sequence() -> tuple[np.ndarray, np.ndarray, int, int]:
        """Stationary on a block, then 30 frames of flight, then decelerating."""
        takeoff, entry = 25, 55
        frames = np.arange(90)
        xy = np.zeros((90, 2))

        xy[:takeoff] = [500.0, 400.0]  # on the block
        _, flight = synthetic_flight(n=entry - takeoff + 1)
        xy[takeoff : entry + 1] = flight

        # Decelerating in the water: horizontal speed bleeds off.
        last = xy[entry].copy()
        step = xy[entry] - xy[entry - 1]
        for i in range(entry + 1, 90):
            step = step * 0.85
            last = last + step
            xy[i] = last
        return frames, xy, takeoff, entry

    def test_finds_the_airborne_window(self):
        frames, xy, takeoff, entry = self.sequence()
        airborne = np.ones(len(frames), dtype=bool)
        airborne[entry + 1 :] = False  # visibility collapses at water entry

        phase = detect_flight(
            frames,
            xy,
            FPS,
            body_scale_px=BODY_SCALE,
            airborne=airborne,
            noise_sigma=1.0,
            ay_prior=G * PX_PER_M,
        )

        assert phase is not None
        assert phase.takeoff_frame == pytest.approx(takeoff, abs=2)
        # Entry is only ever good to a few frames. The cue that finds it fires
        # when the detector stops resolving the athlete's legs, and the first
        # instants underwater are still very nearly ballistic, so a fit has
        # nothing to object to for a frame or two past the surface.
        assert phase.entry_frame == pytest.approx(entry, abs=4)

    def test_does_not_reach_back_into_the_block_phase(self):
        """The failure mode that flattens the parabola and ruins every metric."""
        frames, xy, takeoff, _ = self.sequence()
        phase = detect_flight(
            frames,
            xy,
            FPS,
            body_scale_px=BODY_SCALE,
            noise_sigma=1.0,
            ay_prior=G * PX_PER_M,
        )
        assert phase is not None
        assert phase.takeoff_frame >= takeoff - 2

    def test_reports_nothing_when_nobody_leaves_the_ground(self):
        frames = np.arange(60)
        xy = np.tile([500.0, 400.0], (60, 1)) + np.random.default_rng(0).normal(0, 1, (60, 2))
        assert detect_flight(frames, xy, FPS, BODY_SCALE, noise_sigma=1.0) is None

    def test_rejects_a_scale_that_would_imply_a_giant(self):
        frames, xy, _, _ = self.sequence()
        config = FlightSearchConfig(min_implied_stature_m=0.5, max_implied_stature_m=0.6)
        assert detect_flight(frames, xy, FPS, BODY_SCALE, config, noise_sigma=1.0) is None


def test_bootstrap_brackets_the_true_value():
    frames, xy = synthetic_flight(n=40, noise=3.0, seed=11)
    intervals = bootstrap_uncertainty(frames, xy, FPS, n_resamples=200, seed=1)

    lo, hi = intervals["vx"]
    assert lo < 5.0 * PX_PER_M < hi
    assert hi > lo
