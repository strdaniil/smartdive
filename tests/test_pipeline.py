"""End-to-end pipeline tests against a synthetic video and a fake backend.

These run in CI without model weights, network access, or sample footage: the
backend is stubbed, so what is under test is the pipeline's own wiring --
tracking, conditioning, phase detection, calibration and export.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

# OpenCV is needed to synthesise the test clip; skip the module rather than
# error if a minimal environment lacks it. The imports below therefore have to
# come after this line.
cv2 = pytest.importorskip("cv2")

from smartdive.analysis import AnalysisConfig, analyze, subject_switches  # noqa: E402
from smartdive.backends.base import FrameDetections, PersonDetection, PoseBackend  # noqa: E402
from smartdive.calibration import DEFAULT_STATURE_M, EXTENDED_BODY_RATIO  # noqa: E402
from smartdive.keypoints import NUM_KEYPOINTS  # noqa: E402
from smartdive.legacy import legacy_metrics, legacy_trajectory  # noqa: E402
from smartdive.physics import G  # noqa: E402
from smartdive.report import write_csv, write_json  # noqa: E402

FPS = 60.0
PX_PER_M = 150.0
TAKEOFF, ENTRY, N_FRAMES = 20, 50, 80

#: Keep the synthetic world self-consistent: a figure this many pixels tall is
#: exactly a default-stature athlete at PX_PER_M, so the body-scale prior the
#: pipeline derives should land on the scale the fixture actually used.
BODY_PX = DEFAULT_STATURE_M * EXTENDED_BODY_RATIO * PX_PER_M


def _pose(
    cx: float,
    cy: float,
    size: float,
    score: float = 0.8,
    submerged: bool = False,
) -> PersonDetection:
    xy = np.zeros((NUM_KEYPOINTS, 2))
    spread = np.linspace(-0.5, 0.5, NUM_KEYPOINTS)
    xy[:, 0] = cx + spread * size * 0.25
    xy[:, 1] = cy + spread * size
    conf = np.full(NUM_KEYPOINTS, 0.9)
    if submerged:
        # Everything from the hips down disappears below the surface, which is
        # what the real footage shows: the visible mass fraction falls off a
        # cliff the moment the athlete breaks the water.
        conf[11:] = 0.05
    return PersonDetection(
        box=np.array([cx - size / 2, cy - size / 2, cx + size / 2, cy + size / 2]),
        score=score,
        xy=xy,
        conf=conf,
    )


class ScriptedBackend(PoseBackend):
    """A backend that replays a known dive plus a stationary bystander."""

    name = "scripted"

    def detect(self, frame: np.ndarray, index: int) -> FrameDetections:
        # The bystander is stationary, well lit and unoccluded, so the detector
        # is consistently more confident about them than about the athlete.
        people = [_pose(1500.0, 500.0, BODY_PX * 0.9, score=0.93)]

        if index < TAKEOFF:
            people.insert(0, _pose(200.0, 400.0, BODY_PX, score=0.78))
        else:
            t = (index - TAKEOFF) / FPS
            cx = 200.0 + 5.0 * PX_PER_M * t
            cy = 400.0 - 1.2 * PX_PER_M * t + 0.5 * G * PX_PER_M * t**2
            people.insert(0, _pose(cx, cy, BODY_PX, score=0.78, submerged=index > ENTRY))
        return FrameDetections(index=index, people=people)


@pytest.fixture(scope="module")
def clip(tmp_path_factory) -> object:
    path = tmp_path_factory.mktemp("video") / "dive.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (1920, 1080))
    for _ in range(N_FRAMES):
        writer.write(np.zeros((1080, 1920, 3), dtype=np.uint8))
    writer.release()
    return path


@pytest.fixture(scope="module")
def result(clip):
    config = AnalysisConfig(landmark="hips", backend="scripted", bootstrap=50)
    return analyze(clip, config, backend=ScriptedBackend())


def test_pipeline_selects_the_diver_not_the_bystander(result):
    xs = [d.center[0] for d in result.subject.observations.values()]
    assert np.ptp(xs) > 400  # the diver crossed the frame
    assert result.quality()["subject_switches"] == 0


def test_the_original_pipeline_would_have_picked_the_bystander(result):
    """The regression this project exists to prevent."""
    legacy = legacy_trajectory(result.detections, landmark="hips")
    travel = np.nanmax(legacy[:, 0]) - np.nanmin(legacy[:, 0])
    assert travel < 1.0

    old = legacy_metrics(legacy, result.video.fps)
    assert old is not None
    # It still reports a confident-looking "flight time" spanning the whole clip.
    assert old.flight_time_s == pytest.approx((N_FRAMES - 1) / FPS)


def test_flight_phase_matches_the_script(result):
    assert result.flight is not None
    assert result.metrics is not None
    assert result.metrics.takeoff_frame == pytest.approx(TAKEOFF, abs=3)
    assert result.metrics.entry_frame == pytest.approx(ENTRY, abs=3)


def test_metrics_are_in_si_units_and_near_the_truth(result):
    m = result.metrics
    assert m is not None
    assert m.horizontal_velocity_ms == pytest.approx(5.0, rel=0.2)
    assert m.takeoff_angle_deg > 0
    assert 0.0 < m.flight_time_s < 1.0
    assert m.flight_distance_m > 1.0


def test_uncertainty_is_reported(result):
    assert "angle_deg" in result.uncertainty
    lo, hi = result.uncertainty["angle_deg"]
    assert lo <= result.metrics.takeoff_angle_deg <= hi


def test_report_round_trips_through_json(result, tmp_path):
    path = write_json(result, tmp_path / "r.json")
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["metrics"]["flight_time_s"] == pytest.approx(result.metrics.flight_time_s)
    assert payload["video"]["fps"] == pytest.approx(FPS)
    assert payload["flight"]["takeoff_frame"] == result.metrics.takeoff_frame
    assert isinstance(payload["warnings"], list)


def test_csv_labels_every_frame_with_its_phase(result, tmp_path):
    import csv

    path = write_csv(result, tmp_path / "t.csv")
    rows = list(csv.DictReader(path.open(encoding="utf-8")))

    assert len(rows) == N_FRAMES
    phases = {r["phase"] for r in rows}
    assert phases == {"block", "flight", "water"}
    assert rows[TAKEOFF + 2]["phase"] == "flight"


def test_manual_window_override_is_honoured(clip):
    config = AnalysisConfig(
        landmark="hips", backend="scripted", bootstrap=0, flight_window=(25, 45)
    )
    manual = analyze(clip, config, backend=ScriptedBackend())

    assert manual.metrics is not None
    assert manual.metrics.takeoff_frame == 25
    assert manual.metrics.entry_frame == 45


def test_detections_are_cached_and_reused(clip, tmp_path):
    config = AnalysisConfig(landmark="hips", backend="scripted", bootstrap=0)
    first = analyze(clip, config, backend=ScriptedBackend(), cache=tmp_path)
    assert list(tmp_path.glob("*.pkl"))

    # Second call must not need a backend at all.
    second = analyze(clip, config, backend=None, cache=tmp_path)
    assert second.metrics.flight_time_s == pytest.approx(first.metrics.flight_time_s)


def test_subject_switches_counts_impossible_jumps():
    xy = np.array([[0.0, 0.0], [10.0, 0.0], [500.0, 0.0], [510.0, 0.0]])
    assert subject_switches(xy, body_scale_px=100.0) == 1
