"""Tracking and subject selection, the bug that made the old pipeline lie."""

from __future__ import annotations

import numpy as np
import pytest

from smartdive.backends.base import FrameDetections, PersonDetection
from smartdive.keypoints import LEFT_HIP, NUM_KEYPOINTS, RIGHT_HIP
from smartdive.legacy import legacy_trajectory
from smartdive.tracking import (
    Track,
    TrackerConfig,
    iou,
    keypoint_distance,
    select_subject,
    stitch_tracks,
    track_all,
)


def person(cx: float, cy: float, size: float = 200.0, score: float = 0.8) -> PersonDetection:
    """A detection whose keypoints sit in a plausible arrangement around a box."""
    xy = np.zeros((NUM_KEYPOINTS, 2))
    offsets = np.linspace(-0.4, 0.4, NUM_KEYPOINTS)
    xy[:, 0] = cx + offsets * size * 0.3
    xy[:, 1] = cy + offsets * size
    xy[LEFT_HIP] = [cx - size * 0.08, cy]
    xy[RIGHT_HIP] = [cx + size * 0.08, cy]
    return PersonDetection(
        box=np.array([cx - size / 2, cy - size / 2, cx + size / 2, cy + size / 2]),
        score=score,
        xy=xy,
        conf=np.full(NUM_KEYPOINTS, 0.9),
    )


def scene(n_frames: int = 40) -> list[FrameDetections]:
    """A diver crossing the frame, plus a brighter, stationary bystander.

    This is the situation in the sample footage: the bystander is better lit and
    unoccluded, so the detector is consistently more confident about them than
    about a motion-blurred athlete in mid-flight.
    """
    frames = []
    for i in range(n_frames):
        diver = person(1800 - 35 * i, 500 + 0.6 * i * i, size=220, score=0.55)
        bystander = person(1750, 560, size=180, score=0.9)
        frames.append(FrameDetections(index=i, people=[diver, bystander]))
    return frames


def test_iou_basics():
    a = np.array([0.0, 0.0, 10.0, 10.0])
    assert iou(a, a) == pytest.approx(1.0)
    assert iou(a, np.array([20.0, 20.0, 30.0, 30.0])) == 0.0
    assert iou(a, np.array([5.0, 0.0, 15.0, 10.0])) == pytest.approx(1 / 3)


def test_tracker_keeps_two_people_apart():
    tracks = track_all(scene())
    assert len(tracks) == 2
    assert all(len(t.observations) >= 35 for t in tracks)


def test_tracker_bridges_a_detector_dropout():
    """A diver at full extension over the water is routinely missed."""
    frames = scene()
    for i in range(18, 23):
        frames[i] = FrameDetections(index=i, people=[frames[i].people[1]])

    tracks = track_all(frames)
    moving = max(tracks, key=lambda t: np.ptp([d.center[0] for d in t.observations.values()]))

    # The same track resumes after the gap rather than a new one starting.
    assert moving.frames[0] < 18
    assert moving.frames[-1] > 22


def test_subject_selection_picks_the_person_who_moves():
    subject, ranking = select_subject(track_all(scene()))
    displacement = np.ptp([d.center[0] for d in subject.observations.values()])

    assert displacement > 1000  # the diver crossed the frame
    assert ranking[0].displacement_bodies > ranking[1].displacement_bodies


def test_the_original_argmax_rule_picks_the_bystander_instead():
    """Documents the defect this project exists to fix."""
    frames = scene()
    legacy = legacy_trajectory(frames, landmark="hips")
    travel = np.nanmax(legacy[:, 0]) - np.nanmin(legacy[:, 0])

    assert travel < 1.0  # the bystander never moved

    subject, _ = select_subject(track_all(frames))
    tracked = np.ptp([d.center[0] for d in subject.observations.values()])
    assert tracked > 1000


def test_a_low_confidence_phantom_cannot_win_on_displacement_alone():
    frames = []
    for i in range(30):
        real = person(900, 500, size=200, score=0.85)
        phantom = person(100 + 60 * i, 300, size=60, score=0.18)
        frames.append(FrameDetections(index=i, people=[real, phantom]))

    subject, _ = select_subject(track_all(frames), TrackerConfig())
    scores = [d.score for d in subject.observations.values()]
    assert np.mean(scores) > 0.5


def test_no_tracks_is_an_error_not_a_silent_wrong_answer():
    with pytest.raises(ValueError, match="found nobody"):
        select_subject([])


def test_keypoint_distance_needs_enough_shared_confident_points():
    xy = np.zeros((NUM_KEYPOINTS, 2))
    high = np.full(NUM_KEYPOINTS, 0.9)
    low = np.full(NUM_KEYPOINTS, 0.1)
    assert keypoint_distance(xy, high, xy + 5.0, high) == pytest.approx(5.0 * np.sqrt(2))
    assert np.isnan(keypoint_distance(xy, high, xy, low))


class TestStitching:
    """Reconnecting the fragments a strict tracker leaves behind."""

    @staticmethod
    def _fragment(track_id: int, frames: range, x0: float, step: float) -> Track:
        track = Track(track_id=track_id)
        for i in frames:
            track.observations[i] = person(x0 + step * i, 500.0, size=200.0)
        track.last_frame = max(frames)
        track.box = track.observations[track.last_frame].box.copy()
        return track

    def test_a_gap_between_two_halves_of_one_person_is_bridged(self):
        """What takeoff does to a tracker: the body changes shape and it lets go."""
        a = self._fragment(1, range(0, 10), 400.0, 30.0)
        b = self._fragment(2, range(14, 26), 400.0, 30.0)

        merged = stitch_tracks([a, b], TrackerConfig())

        assert len(merged) == 1
        assert len(merged[0].observations) == 22

    def test_two_people_moving_apart_are_not_stitched(self):
        a = self._fragment(1, range(0, 10), 400.0, 30.0)
        b = self._fragment(2, range(14, 26), 2400.0, -30.0)

        assert len(stitch_tracks([a, b], TrackerConfig())) == 2

    def test_fragments_that_overlap_in_time_are_merged_when_they_agree(self):
        """A strict tracker opens the new fragment before it closes the old one."""
        a = self._fragment(1, range(0, 20), 400.0, 30.0)
        b = self._fragment(2, range(14, 30), 400.0, 30.0)

        merged = stitch_tracks([a, b], TrackerConfig())

        assert len(merged) == 1
        assert len(merged[0].observations) == 30

    def test_stitching_rescues_a_subject_broken_into_pieces(self):
        """Displacement per fragment is small; peak speed is not, which is why
        subject selection ranks on speed."""
        frames = []
        for i in range(40):
            people = [person(1750, 560, size=180, score=0.9)]  # the bystander
            if i < 12 or i > 17:  # the diver, lost through takeoff
                people.append(person(1800 - 40 * i, 500.0, size=220, score=0.6))
            frames.append(FrameDetections(index=i, people=people))

        subject, _ = select_subject(track_all(frames))
        assert np.ptp([d.center[0] for d in subject.observations.values()]) > 1000
