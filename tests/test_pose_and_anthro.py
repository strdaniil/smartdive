"""Skeleton conditioning and the anthropometric scale."""

from __future__ import annotations

import numpy as np
import pytest

from smartdive import anthro
from smartdive.keypoints import (
    LEFT_ANKLE,
    LEFT_HIP,
    LEFT_KNEE,
    LEFT_SHOULDER,
    NUM_KEYPOINTS,
    RIGHT_ANKLE,
    RIGHT_HIP,
    RIGHT_KNEE,
    RIGHT_SHOULDER,
)
from smartdive.pose import (
    build_pose_series,
    centre_of_mass,
    condition_pose,
    lateral_view_score,
    mirror_missing,
    prepare,
)
from smartdive.tracking import Track
from tests.synthetic import detection, skeleton

FPS = 60.0
STATURE_PX = 400.0


def _track(n: int = 30, drop: dict[int, list[int]] | None = None) -> Track:
    """A track of a body drifting rightward, with optional keypoint dropouts."""
    track = Track(track_id=1)
    for index in range(n):
        det = detection((300.0 + 12.0 * index, 400.0), STATURE_PX, rotation_deg=90.0)
        for keypoint in (drop or {}).get(index, []):
            det.conf[keypoint] = 0.05
        track.observations[index] = det
    track.last_frame = n - 1
    track.box = track.observations[n - 1].box.copy()
    return track


class TestAnthropometry:
    def test_stature_is_recovered_from_bone_lengths(self):
        series = build_pose_series(_track(), 30, FPS)
        size = anthro.measure(series.xy, series.measured)
        assert size.stature_px == pytest.approx(STATURE_PX, rel=0.02)
        assert size.dispersion < 0.05

    def test_a_foreshortened_pose_is_never_measured_as_larger(self):
        """Projection only shortens, so the high percentile is the right read."""
        upright = skeleton((500.0, 500.0), STATURE_PX, rotation_deg=0.0)
        # Squash horizontally, as an out-of-plane rotation would.
        squashed = upright.copy()
        squashed[:, 0] = 500.0 + (squashed[:, 0] - 500.0) * 0.4

        present = np.ones((1, NUM_KEYPOINTS), dtype=bool)
        a = anthro.measure(upright[None], present, min_observations=1)
        b = anthro.measure(squashed[None], present, min_observations=1)
        assert b.stature_px <= a.stature_px * 1.01

    def test_one_wild_limb_cannot_move_the_estimate(self):
        series = build_pose_series(_track(), 30, FPS)
        xy = series.xy.copy()
        xy[:, LEFT_ANKLE, :] += 900.0  # a badly mislocated ankle, every frame

        size = anthro.measure(xy, series.measured)
        assert size.stature_px == pytest.approx(STATURE_PX, rel=0.05)
        assert "left_shank" not in size.per_segment  # trimmed as an outlier

    def test_relative_size_tracks_a_receding_athlete(self):
        """The signal the depth correction rides on."""
        track = Track(track_id=1)
        scales = np.linspace(1.0, 0.7, 20)
        for index, scale in enumerate(scales):
            track.observations[index] = detection(
                (300.0 + 12.0 * index, 400.0), STATURE_PX * scale, rotation_deg=90.0
            )
        track.last_frame = 19
        track.box = track.observations[19].box.copy()

        series = build_pose_series(track, 20, FPS)
        reference = anthro.measure(series.xy, series.measured)
        relative = anthro.relative_size_series(series.xy, series.measured, reference)

        assert relative[0] > relative[-1]
        assert relative[-1] / relative[0] == pytest.approx(0.7, rel=0.1)


class TestPoseConditioning:
    def test_a_missing_keypoint_is_filled_from_its_mirror(self):
        track = _track(drop={5: [LEFT_KNEE], 6: [LEFT_KNEE]})
        series = build_pose_series(track, 30, FPS)
        assert np.isnan(series.xy[5, LEFT_KNEE, 0])

        mirrored = mirror_missing(series)
        assert mirrored.mirrored[5, LEFT_KNEE]
        assert mirrored.xy[5, LEFT_KNEE] == pytest.approx(series.xy[5, RIGHT_KNEE])

    def test_conditioning_fills_short_gaps_in_a_single_keypoint(self):
        # Drop the same keypoint on both sides, so mirroring cannot help and
        # temporal interpolation has to.
        track = _track(drop={9: [LEFT_KNEE, RIGHT_KNEE], 10: [LEFT_KNEE, RIGHT_KNEE]})
        conditioned = condition_pose(mirror_missing(build_pose_series(track, 30, FPS)))
        assert conditioned.present()[9, LEFT_KNEE]
        assert conditioned.interpolated[9, LEFT_KNEE]

    def test_the_centre_of_mass_no_longer_jumps_when_a_limb_drops_out(self):
        """The bias that motivated conditioning keypoints instead of the landmark.

        Recomputing a mass-weighted sum over a *different* set of segments moves
        the answer even when the body has not moved. Completing the skeleton
        first means the same segments contribute in every frame.
        """
        dropped = {12: [LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE]}

        naive = centre_of_mass(build_pose_series(_track(drop=dropped), 30, FPS), min_mass=0.6)
        clean = centre_of_mass(prepare(_track(drop=dropped), 30, FPS))
        intact = centre_of_mass(prepare(_track(), 30, FPS))

        naive_error = np.linalg.norm(naive[12] - intact[12])
        clean_error = np.linalg.norm(clean[12] - intact[12])
        assert naive_error > 5.0
        assert clean_error < 1.0

    def test_visible_mass_reports_the_detector_not_the_reconstruction(self):
        """Water entry must stay visible after the skeleton has been completed."""
        submerged = Track(track_id=1)
        for index in range(20):
            submerged.observations[index] = detection(
                (300.0 + 12.0 * index, 400.0),
                STATURE_PX,
                rotation_deg=90.0,
                submerged=index >= 14,
            )
        submerged.last_frame = 19
        submerged.box = submerged.observations[19].box.copy()

        series = prepare(submerged, 20, FPS)
        mass = series.visible_mass()
        assert mass[13] > 0.9
        assert mass[15] < 0.6

    def test_lateral_view_score_separates_side_on_from_head_on(self):
        side_on = _track()
        series = build_pose_series(side_on, 30, FPS)
        # The synthetic body has both shoulders at the same point: perfectly
        # side-on, which is what the score should say.
        assert lateral_view_score(series) == pytest.approx(1.0, abs=0.05)

        frontal = build_pose_series(_track(), 30, FPS)
        for index in range(30):
            hips = frontal.xy[index, [LEFT_HIP, RIGHT_HIP]].mean(axis=0)
            shoulders = frontal.xy[index, [LEFT_SHOULDER, RIGHT_SHOULDER]].mean(axis=0)
            trunk = float(np.linalg.norm(shoulders - hips))
            frontal.xy[index, LEFT_SHOULDER, 1] -= 0.45 * trunk
            frontal.xy[index, RIGHT_SHOULDER, 1] += 0.45 * trunk
        assert lateral_view_score(frontal) < 0.2
