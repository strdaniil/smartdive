"""Metrics, calibration and the centre-of-mass model."""

from __future__ import annotations

import numpy as np
import pytest

from smartdive import anthro
from smartdive import calibration as calib
from smartdive.com import center_of_mass, visible_mass_fraction
from smartdive.keypoints import (
    LEFT_ANKLE,
    LEFT_HIP,
    LEFT_KNEE,
    NUM_KEYPOINTS,
    RIGHT_ANKLE,
    RIGHT_HIP,
    RIGHT_KNEE,
    body_scale,
    resolve_landmark,
)
from smartdive.metrics import (
    compute_metrics,
    front_knee_angle,
    joint_angle,
    knee_angle_feedback,
)
from smartdive.physics import G, fit_ballistic
from tests.test_physics import PX_PER_M, synthetic_flight

FPS = 120.0


def full_body(cx: float = 500.0, cy: float = 400.0, size: float = 200.0) -> np.ndarray:
    """A standing figure with every keypoint in a sane place."""
    xy = np.zeros((NUM_KEYPOINTS, 2))
    layout = {
        0: (0.0, -0.50),
        1: (-0.03, -0.53),
        2: (0.03, -0.53),
        3: (-0.07, -0.52),
        4: (0.07, -0.52),
        5: (-0.12, -0.38),
        6: (0.12, -0.38),
        7: (-0.16, -0.20),
        8: (0.16, -0.20),
        9: (-0.18, -0.02),
        10: (0.18, -0.02),
        11: (-0.09, 0.00),
        12: (0.09, 0.00),
        13: (-0.09, 0.25),
        14: (0.09, 0.25),
        15: (-0.09, 0.50),
        16: (0.09, 0.50),
    }
    for i, (dx, dy) in layout.items():
        xy[i] = (cx + dx * size, cy + dy * size)
    return xy


class TestCentreOfMass:
    def test_com_lands_between_hips_and_shoulders(self):
        xy = full_body()
        conf = np.full(NUM_KEYPOINTS, 0.9)

        com = center_of_mass(xy, conf)

        # Anatomically the whole-body CoM sits just above the hips.
        assert com[1] < xy[LEFT_HIP, 1]
        assert com[1] > xy[5, 1]
        assert com[0] == pytest.approx(500.0, abs=5.0)

    def test_com_is_unmoved_by_a_horizontal_flip(self):
        xy = full_body()
        conf = np.full(NUM_KEYPOINTS, 0.9)
        flipped = xy.copy()
        flipped[:, 0] = 1000.0 - flipped[:, 0]

        assert center_of_mass(xy, conf)[0] == pytest.approx(
            1000.0 - center_of_mass(flipped, conf)[0], abs=1e-6
        )

    def test_com_refuses_to_guess_from_a_couple_of_limbs(self):
        conf = np.zeros(NUM_KEYPOINTS)
        conf[[LEFT_HIP, LEFT_KNEE, LEFT_ANKLE]] = 0.9

        assert np.isnan(center_of_mass(full_body(), conf)).all()

    def test_visible_mass_fraction_tracks_submersion(self):
        conf = np.full(NUM_KEYPOINTS, 0.9)
        assert visible_mass_fraction(conf) == pytest.approx(1.0, abs=0.02)

        # Legs underwater: trunk, head and arms remain.
        conf[[LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE]] = 0.0
        assert visible_mass_fraction(conf) < 0.7


class TestLandmarks:
    def test_pair_landmark_is_the_midpoint(self):
        xy = full_body()
        conf = np.full(NUM_KEYPOINTS, 0.9)
        hips = resolve_landmark(xy, conf, "hips")
        assert hips == pytest.approx((xy[LEFT_HIP] + xy[RIGHT_HIP]) / 2)

    def test_low_confidence_yields_nan_not_a_guess(self):
        conf = np.full(NUM_KEYPOINTS, 0.9)
        conf[LEFT_HIP] = 0.1
        assert np.isnan(resolve_landmark(full_body(), conf, "hips")).all()

    def test_unknown_landmark_is_rejected(self):
        with pytest.raises(ValueError, match="unknown landmark"):
            resolve_landmark(full_body(), np.ones(NUM_KEYPOINTS), "elbow_of_doom")

    def test_body_scale_is_the_widest_keypoint_separation(self):
        xy = full_body(size=200.0)
        conf = np.full(NUM_KEYPOINTS, 0.9)
        assert body_scale(xy, conf) == pytest.approx(200.0, rel=0.05)


def _body(stature_px: float) -> anthro.BodySize:
    """A minimal :class:`BodySize` standing in for a real measurement."""
    return anthro.BodySize(
        stature_px=stature_px,
        per_segment={"trunk": stature_px, "left_leg": stature_px},
        dispersion=0.02,
        n_frames=10,
    )


class TestCalibration:
    def test_gravity_calibration_inverts_the_fit(self):
        frames, xy = synthetic_flight(px_per_m=180.0)
        fit = fit_ballistic(frames, xy, FPS)
        assert calib.from_gravity(fit).px_per_metre == pytest.approx(180.0, rel=1e-6)

    def test_gravity_calibration_refuses_a_non_flight_window(self):
        frames = np.arange(20)
        xy = np.stack([np.arange(20.0), -(np.arange(20.0) ** 2)], axis=1)  # accelerating upward
        fit = fit_ballistic(frames, xy, FPS)
        with pytest.raises(ValueError, match="not free flight"):
            calib.from_gravity(fit)

    def test_reference_calibration(self):
        c = calib.from_reference(500.0, 2.5, "lane width")
        assert c.px_per_metre == pytest.approx(200.0)
        assert c.to_metres(400.0) == pytest.approx(2.0)

    def test_resolve_prefers_a_reference_over_everything(self):
        frames, xy = synthetic_flight(px_per_m=180.0)
        fit = fit_ballistic(frames, xy, FPS)
        c, _ = calib.resolve(_body(300.0), reference_px=500.0, reference_m=2.5, fit=fit)
        assert c.method == "reference"
        assert c.px_per_metre == pytest.approx(200.0)

    def test_resolve_notes_a_gravity_cross_check_that_agrees(self):
        frames, xy = synthetic_flight(px_per_m=180.0)
        fit = fit_ballistic(frames, xy, FPS)

        c, notes = calib.resolve(_body(180.0 * 1.78), fit=fit)

        # The body scale is the reported scale; gravity is only ever a check.
        assert c.method == "stature_assumed"
        assert c.px_per_metre == pytest.approx(180.0, rel=1e-3)
        assert notes and "within" in notes[0]

    def test_resolve_explains_when_the_gravity_cross_check_disagrees(self):
        frames, xy = synthetic_flight(px_per_m=180.0)
        fit = fit_ballistic(frames, xy, FPS)
        # A body scale implying a wildly different scene scale.
        c, notes = calib.resolve(_body(2000.0), fit=fit)

        assert c.method == "stature_assumed"
        assert notes and "disagrees" in notes[0]

    def test_resolve_without_a_fit_uses_the_body_scale(self):
        c, notes = calib.resolve(_body(350.0), stature_m=1.80)
        assert c.method == "stature"
        assert c.px_per_metre == pytest.approx(350.0 / 1.80)
        assert not notes

    def test_a_varying_scale_is_read_per_frame(self):
        """Oblique footage: the athlete shrinks as they fly away from the camera."""
        relative = np.linspace(1.0, 0.8, 20)
        c, _ = calib.resolve(_body(360.0), stature_m=1.80, relative_size=relative)

        assert c.varies
        assert c.at(0) == pytest.approx(200.0)
        assert c.at(19) == pytest.approx(160.0)
        assert c.depth_change() == pytest.approx(0.25, rel=1e-3)

    def test_a_supplied_stature_is_used_verbatim(self):
        c, _ = calib.resolve(_body(370.0), stature_m=1.85)
        assert c.method == "stature"
        assert c.stature_m == pytest.approx(1.85)
        assert c.px_per_metre == pytest.approx(370.0 / 1.85)

    def test_no_stature_falls_back_to_the_documented_default(self):
        c, _ = calib.resolve(_body(370.0))
        assert c.method == "stature_assumed"
        assert c.stature_m == pytest.approx(anthro.DEFAULT_STATURE_M)
        assert c.px_per_metre == pytest.approx(370.0 / anthro.DEFAULT_STATURE_M)

    def test_plausibility_flags_a_nonsense_scale(self):
        ok, _ = calib.plausibility(calib.Calibration(1000.0, "reference"), 350.0)
        assert not ok


class TestTimeBase:
    """Recovering the capture rate of slow-motion footage from gravity."""

    @staticmethod
    def _fit(px_per_m: float, factor: float, n: int = 40):
        """A flight sampled as if the clip plays back ``1 / factor`` times slow."""
        frames, xy = synthetic_flight(n=n, px_per_m=px_per_m)
        t = frames / FPS
        stretched = np.arange(n) / FPS
        # Re-sample the same motion against a stretched clock.
        return fit_ballistic(
            np.arange(n),
            np.stack(
                [
                    np.interp(stretched * factor, t, xy[:, 0]),
                    np.interp(stretched * factor, t, xy[:, 1]),
                ],
                axis=1,
            ),
            FPS,
        )

    def test_real_time_footage_is_left_alone(self):
        base, notes = calib.resolve_time_base(FPS, self._fit(180.0, 1.0), 180.0, n_points=40)
        assert not base.corrected
        assert base.effective_fps == pytest.approx(FPS)
        assert not notes

    def test_four_times_slow_motion_is_recovered(self):
        base, notes = calib.resolve_time_base(FPS, self._fit(180.0, 0.25), 180.0, n_points=40)
        assert base.corrected
        assert base.effective_fps == pytest.approx(FPS * 4, rel=0.05)
        assert notes and "slow motion" in notes[0]

    def test_a_poorly_determined_gravity_never_triggers_a_correction(self):
        """The guard that stops ordinary footage declaring itself slow motion."""
        frames, xy = synthetic_flight(n=11, noise=8.0, seed=5)
        fit = fit_ballistic(frames, xy, FPS)
        assert fit.ay_stderr > 0.25 * fit.ay  # the window cannot measure gravity
        base, _ = calib.resolve_time_base(FPS, fit, PX_PER_M, n_points=fit.n_points)
        assert not base.corrected


class TestMetrics:
    def test_metrics_round_trip_a_known_dive(self):
        frames, xy = synthetic_flight(n=40, vx_ms=5.0, vy_ms=1.5, px_per_m=PX_PER_M)
        fit = fit_ballistic(frames, xy, FPS)
        c = calib.Calibration(PX_PER_M, "test")

        m = compute_metrics(fit, c)

        assert m.horizontal_velocity_ms == pytest.approx(5.0, rel=1e-4)
        assert m.vertical_velocity_ms == pytest.approx(1.5, rel=1e-4)
        assert m.takeoff_speed_ms == pytest.approx(np.hypot(5.0, 1.5), rel=1e-4)
        assert m.flight_time_s == pytest.approx(39 / FPS)
        # Apex height for a 1.5 m/s launch is v^2 / 2g.
        assert m.peak_height_m == pytest.approx(1.5**2 / (2 * G), rel=1e-3)

    def test_flight_distance_is_speed_times_time(self):
        frames, xy = synthetic_flight(n=40, vx_ms=5.0)
        fit = fit_ballistic(frames, xy, FPS)
        m = compute_metrics(fit, calib.Calibration(PX_PER_M, "test"))
        assert m.flight_distance_m == pytest.approx(5.0 * (39 / FPS), rel=1e-3)

    def test_format_table_mentions_the_calibration_source(self):
        frames, xy = synthetic_flight()
        fit = fit_ballistic(frames, xy, FPS)
        table = compute_metrics(fit, calib.Calibration(PX_PER_M, "gravity")).format_table()
        assert "gravity" in table
        assert "m/s" in table


class TestJointAngles:
    def test_right_angle(self):
        assert joint_angle((0, 1), (0, 0), (1, 0)) == pytest.approx(90.0)

    def test_straight_limb(self):
        assert joint_angle((0, 2), (0, 1), (0, 0)) == pytest.approx(180.0)

    def test_degenerate_returns_nan(self):
        assert np.isnan(joint_angle((0, 0), (0, 0), (1, 0)))

    def test_front_knee_is_found_without_being_told_which_way_they_face(self):
        xy = full_body()
        conf = np.full(NUM_KEYPOINTS, 0.9)
        # Push the left leg forward and bend it, as in a track start.
        xy[LEFT_KNEE] += [70.0, 0.0]
        xy[LEFT_ANKLE] += [40.0, 0.0]

        angle, side = front_knee_angle(xy, conf)

        assert side == "left"
        assert 90.0 < angle < 175.0

    def test_knee_feedback_covers_all_three_bands(self):
        assert "crouched" in knee_angle_feedback(100.0)
        assert "straighter" in knee_angle_feedback(170.0)
        assert "within" in knee_angle_feedback(140.0)
        assert "not measurable" in knee_angle_feedback(float("nan"))


class TestStatureValidation:
    """The one number a camera cannot check, so the code checks it instead.

    Every distance and speed is inversely proportional to the athlete's height,
    so a units slip is not a small error, it is a factor of a hundred, and
    nothing downstream can notice, because a scale is precisely what a single
    camera has no independent handle on.
    """

    def test_a_plausible_height_passes_through(self):
        assert anthro.check_stature(1.85) == pytest.approx(1.85)
        assert anthro.check_stature(anthro.MIN_STATURE_M) == pytest.approx(anthro.MIN_STATURE_M)
        assert anthro.check_stature(anthro.MAX_STATURE_M) == pytest.approx(anthro.MAX_STATURE_M)

    def test_centimetres_are_caught_and_named(self):
        with pytest.raises(ValueError, match="centimetres"):
            anthro.check_stature(185)

    def test_the_error_says_what_to_pass_instead(self):
        with pytest.raises(ValueError, match=r"pass 1\.85"):
            anthro.check_stature(185)

    def test_a_height_that_is_merely_wrong_is_reported_as_out_of_range(self):
        with pytest.raises(ValueError, match="plausible"):
            anthro.check_stature(4.0)

    def test_nonsense_is_rejected(self):
        for bad in (0.0, -1.8, float("nan")):
            with pytest.raises(ValueError):
                anthro.check_stature(bad)

    def test_the_calibration_rejects_it_too_not_just_the_cli(self):
        """Belt and braces: the library is usable without going through argparse."""
        with pytest.raises(ValueError, match="centimetres"):
            calib.resolve(_body(370.0), stature_m=185)
