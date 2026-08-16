"""Tracking and subject selection -- the bug that made the old pipeline lie."""

from __future__ import annotations

import numpy as np
import pytest

from smartdive.backends.base import FrameDetections, PersonDetection
from smartdive.keypoints import LEFT_HIP, NUM_KEYPOINTS, RIGHT_HIP
from smartdive.legacy import legacy_trajectory
from smartdive.tracking import TrackerConfig, iou, select_subject, track_all


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
