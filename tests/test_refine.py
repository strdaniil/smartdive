"""The second-pass follow-crop, and the checks that keep it from doing harm."""

from __future__ import annotations

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from smartdive.backends.base import FrameDetections, PoseBackend  # noqa: E402
from smartdive.camera import identity_motion  # noqa: E402
from smartdive.refine import (  # noqa: E402
    RefineConfig,
    apply_refinement,
    refine_subject,
    reject_inconsistent,
)
from smartdive.tracking import Track  # noqa: E402
from tests.synthetic import detection  # noqa: E402

N_FRAMES, STATURE_PX = 30, 300.0
LOST = range(12, 18)


def _position(index: int) -> tuple[float, float]:
    return (400.0 + 40.0 * index, 500.0)


@pytest.fixture(scope="module")
def clip(tmp_path_factory):
    path = tmp_path_factory.mktemp("refine") / "clip.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (1920, 1080))
    for _ in range(N_FRAMES):
        writer.write(np.zeros((1080, 1920, 3), dtype=np.uint8))
    writer.release()
    return path


def _track_with_a_hole() -> Track:
    """What the first pass produces: the athlete lost through the fast frames."""
    track = Track(track_id=1)
    for index in range(N_FRAMES):
        if index in LOST:
            continue
        track.observations[index] = detection(_position(index), STATURE_PX, rotation_deg=90.0)
    track.last_frame = N_FRAMES - 1
    track.box = track.observations[N_FRAMES - 1].box.copy()
    return track


class CropBackend(PoseBackend):
    """A backend that always finds the athlete, wherever the crop is cut.

    Stands in for what actually happens on real footage: the athlete is there in
    the pixels the whole time, and it is only the resolution the whole-frame
    pass could spare that made them undetectable.
    """

    name = "crop"

    def __init__(self, decoy: bool = False) -> None:
        self.decoy = decoy
        self.crops: list[tuple[int, int]] = []

    def detect(self, frame: np.ndarray, index: int) -> FrameDetections:
        height, width = frame.shape[:2]
        self.crops.append((width, height))
        # The crop's own coordinates: the athlete sits at its centre.
        people = [detection((width / 2.0, height / 2.0), STATURE_PX, rotation_deg=90.0)]
        if self.decoy:
            people.append(
                detection((width / 2.0 + 260.0, height / 2.0), STATURE_PX, rotation_deg=0.0)
            )
        return FrameDetections(index=index, people=people)


def test_the_crop_pass_recovers_frames_the_first_pass_lost(clip):
    track = _track_with_a_hole()
    backend = CropBackend()

    refined, report = refine_subject(
        clip, track, backend, identity_motion(N_FRAMES), N_FRAMES, RefineConfig()
    )

    assert report.frames_recovered == len(LOST)
    for index in LOST:
        assert index in refined
        assert refined[index].xy.shape == (17, 2)


def test_recovered_frames_land_where_the_athlete_actually_was(clip):
    track = _track_with_a_hole()
    refined, _ = refine_subject(
        clip, track, CropBackend(), identity_motion(N_FRAMES), N_FRAMES, RefineConfig()
    )
    merged = apply_refinement(track, refined)

    for index in LOST:
        hip = merged.observations[index].xy[11:13].mean(axis=0)
        assert hip[0] == pytest.approx(_position(index)[0], abs=0.35 * STATURE_PX)


def test_the_crop_is_much_smaller_than_the_frame(clip):
    """The whole point: the athlete fills the network's input."""
    backend = CropBackend()
    refine_subject(
        clip, _track_with_a_hole(), backend, identity_motion(N_FRAMES), N_FRAMES, RefineConfig()
    )
    widths = [w for w, _ in backend.crops]
    assert max(widths) < 1920 / 2


def test_a_neighbour_standing_nearby_is_not_mistaken_for_the_athlete(clip):
    """A crowded pool deck is the normal case, not the exception."""
    track = _track_with_a_hole()
    refined, _ = refine_subject(
        clip,
        track,
        CropBackend(decoy=True),
        identity_motion(N_FRAMES),
        N_FRAMES,
        RefineConfig(),
    )
    merged = apply_refinement(track, refined)
    for index in LOST:
        if index not in merged.observations:
            continue
        hip = merged.observations[index].xy[11:13].mean(axis=0)
        assert hip[0] == pytest.approx(_position(index)[0], abs=0.5 * STATURE_PX)


def test_an_inconsistent_refinement_is_thrown_away():
    """A refined frame that disagrees with its own neighbours is worse than a gap."""
    track = _track_with_a_hole()
    good = {i: detection(_position(i), STATURE_PX, rotation_deg=90.0) for i in LOST}
    bad = dict(good)
    bad[14] = detection((_position(14)[0] + 900.0, 500.0), STATURE_PX, rotation_deg=90.0)

    assert set(reject_inconsistent(track, good)) == set(LOST)
    kept = reject_inconsistent(track, bad)
    assert 14 not in kept
    assert set(kept) == set(LOST) - {14}


def test_refinement_never_removes_an_original_observation():
    track = _track_with_a_hole()
    nonsense = {i: detection((50.0, 50.0), STATURE_PX) for i in range(N_FRAMES) if i not in LOST}
    merged = apply_refinement(track, nonsense)
    assert set(track.observations).issubset(set(merged.observations))
