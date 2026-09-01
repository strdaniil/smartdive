"""Camera-motion estimation: can a known pan be measured and undone?"""

from __future__ import annotations

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from smartdive.backends.base import FrameDetections  # noqa: E402
from smartdive.camera import (  # noqa: E402
    CameraConfig,
    estimate_motion,
    identity_motion,
    stabilise_detections,
)
from tests.synthetic import detection  # noqa: E402

WIDTH, HEIGHT, N_FRAMES = 960, 540, 40
PAN_PX_PER_FRAME = 2.5


def _textured_background(rng: np.random.Generator) -> np.ndarray:
    """A wide, feature-rich plate to pan a window across."""
    plate = rng.integers(0, 255, (HEIGHT + 120, WIDTH + 320, 3), dtype=np.uint8)
    return cv2.GaussianBlur(plate, (5, 5), 0)


@pytest.fixture(scope="module")
def panning_clip(tmp_path_factory):
    """A clip of a static scene shot by a camera panning at a known rate."""
    rng = np.random.default_rng(7)
    plate = _textured_background(rng)
    path = tmp_path_factory.mktemp("pan") / "pan.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (WIDTH, HEIGHT))
    for index in range(N_FRAMES):
        x = round(index * PAN_PX_PER_FRAME)
        writer.write(plate[0:HEIGHT, x : x + WIDTH].copy())
    writer.release()
    return path


def test_a_known_pan_is_measured(panning_clip):
    motion = estimate_motion(panning_clip, None, CameraConfig(work_size=WIDTH))

    assert len(motion) == N_FRAMES
    assert motion.ok.all()
    # The camera moved right, so a point observed in a later frame belongs
    # further right in the reference frame.
    drift = motion.to_reference[:, 0, 2]
    assert drift[-1] == pytest.approx((N_FRAMES - 1) * PAN_PX_PER_FRAME, abs=3.0)
    assert abs(motion.to_reference[-1, 1, 2]) < 3.0  # no vertical component


def test_stabilising_a_point_that_moved_with_the_camera_holds_it_still(panning_clip):
    """A world-fixed point tracks across the frame; stabilised, it must not."""
    motion = estimate_motion(panning_clip, None, CameraConfig(work_size=WIDTH))

    positions = np.array(
        [
            motion.stabilise(index, np.array([[400.0 - index * PAN_PX_PER_FRAME, 300.0]]))[0]
            for index in range(N_FRAMES)
        ]
    )
    assert np.ptp(positions[:, 0]) < 5.0
    assert np.ptp(positions[:, 1]) < 5.0


def test_stabilise_and_unstabilise_are_inverses(panning_clip):
    motion = estimate_motion(panning_clip, None, CameraConfig(work_size=WIDTH))
    points = np.array([[100.0, 200.0], [640.0, 480.0]])
    for index in (0, 7, N_FRAMES - 1):
        back = motion.unstabilise(index, motion.stabilise(index, points))
        assert back == pytest.approx(points, abs=1e-6)


def test_identity_motion_changes_nothing():
    motion = identity_motion(5)
    points = np.array([[10.0, 20.0]])
    assert motion.stabilise(3, points) == pytest.approx(points)
    assert motion.zoom(3) == pytest.approx(1.0)
    assert motion.to_dict()["max_drift_px"] == 0.0


def test_detections_are_carried_into_the_stabilised_frame():
    motion = identity_motion(2)
    motion.to_reference[1] = np.array([[1.0, 0.0, 25.0], [0.0, 1.0, -10.0]])
    person = detection((500.0, 400.0), 300.0)
    frames = [
        FrameDetections(index=0, people=[person]),
        FrameDetections(index=1, people=[person]),
    ]

    moved = stabilise_detections(frames, motion)

    assert moved[0].people[0].xy == pytest.approx(person.xy)
    assert moved[1].people[0].xy == pytest.approx(person.xy + np.array([25.0, -10.0]))
    # The original detections must not have been mutated in place.
    assert frames[1].people[0].xy == pytest.approx(person.xy)
