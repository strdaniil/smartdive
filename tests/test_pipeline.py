"""End-to-end pipeline tests against a synthetic video and a fake backend.

These run in CI without model weights, network access, or sample footage: the
backend is stubbed, so what is under test is the pipeline's own wiring,
tracking, conditioning, phase detection, calibration and export.

The clip is a black rectangle, which means the camera-motion estimator finds no
features and correctly reports no motion. That is a real path through the code
and worth exercising, but it is also why these tests say nothing about
stabilisation; :mod:`tests.test_camera` covers that separately, on a synthetic
scene that actually moves.
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
from smartdive.legacy import legacy_metrics, legacy_trajectory  # noqa: E402
from smartdive.report import write_csv, write_json  # noqa: E402
from tests.synthetic import ScriptedBackend  # noqa: E402

FPS = 60.0
PX_PER_M = 150.0
STATURE_M = 1.80
STATURE_PX = STATURE_M * PX_PER_M
TAKEOFF, ENTRY, N_FRAMES = 20, 50, 80
VX_MS, VY_MS = 5.0, 1.2


def _backend(**kwargs) -> ScriptedBackend:
    options = dict(
        takeoff=TAKEOFF,
        entry=ENTRY,
        fps=FPS,
        px_per_m=PX_PER_M,
        stature_px=STATURE_PX,
        vx_ms=VX_MS,
        vy_ms=VY_MS,
    )
    options.update(kwargs)
    return ScriptedBackend(**options)


def _config(**kwargs) -> AnalysisConfig:
    options = dict(backend="scripted", bootstrap=50, refine=False, stature_m=STATURE_M)
    options.update(kwargs)
    return AnalysisConfig(**options)


def _blank_clip(path, n_frames: int):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (1920, 1080))
    for _ in range(n_frames):
        writer.write(np.zeros((1080, 1920, 3), dtype=np.uint8))
    writer.release()
    return path


@pytest.fixture(scope="module")
def clip(tmp_path_factory):
    return _blank_clip(tmp_path_factory.mktemp("video") / "dive.mp4", N_FRAMES)


@pytest.fixture(scope="module")
def result(clip):
    return analyze(clip, _config(), backend=_backend())


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


def test_the_scale_is_recovered_from_the_athletes_own_proportions(result):
    """Generate a body at a known pixel stature; the calibration must find it."""
    assert result.body.stature_px == pytest.approx(STATURE_PX, rel=0.05)
    assert result.calibration is not None
    assert result.calibration.px_per_metre == pytest.approx(PX_PER_M, rel=0.05)
    assert result.calibration.method == "stature"


def test_metrics_are_in_si_units_and_near_the_truth(result):
    m = result.metrics
    assert m is not None
    assert m.horizontal_velocity_ms == pytest.approx(VX_MS, rel=0.1)
    assert m.vertical_velocity_ms == pytest.approx(VY_MS, rel=0.25)
    assert m.flight_time_s == pytest.approx((ENTRY - TAKEOFF) / FPS, rel=0.2)
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
    assert payload["time_base"]["method"] == "container"
    assert payload["body"]["stature_px"] == pytest.approx(result.body.stature_px, abs=0.1)
    assert isinstance(payload["warnings"], list)


def test_csv_labels_every_frame_with_its_phase(result, tmp_path):
    import csv

    path = write_csv(result, tmp_path / "t.csv")
    rows = list(csv.DictReader(path.open(encoding="utf-8")))

    assert len(rows) == N_FRAMES
    phases = {r["phase"] for r in rows}
    assert phases == {"block", "flight", "water"}
    assert rows[result.metrics.takeoff_frame + 2]["phase"] == "flight"


def test_manual_window_override_is_honoured(clip):
    manual = analyze(clip, _config(bootstrap=0, flight_window=(25, 45)), backend=_backend())

    assert manual.metrics is not None
    assert manual.metrics.takeoff_frame == 25
    assert manual.metrics.entry_frame == 45


def test_detections_are_cached_and_reused(clip, tmp_path):
    config = _config(bootstrap=0)
    first = analyze(clip, config, backend=_backend(), cache=tmp_path)
    assert list(tmp_path.glob("*.pkl"))

    # Second call must not need a backend at all.
    second = analyze(clip, config, backend=None, cache=tmp_path)
    assert second.metrics.flight_time_s == pytest.approx(first.metrics.flight_time_s)


def test_slow_motion_is_detected_and_the_clock_corrected(tmp_path):
    """A clip whose container frame rate is four times slower than capture.

    The same dive, filmed at 240 fps and written into a 60 fps container: every
    pixel of it is identical to the real-time version, and only the relationship
    between frames and seconds differs. Without the correction the pipeline
    reports a quarter of the real speed and four times the real flight time,
    with nothing in the footage to say otherwise.
    """
    slow_takeoff, slow_entry, n = 40, 40 + 4 * (ENTRY - TAKEOFF), 4 * N_FRAMES
    path = _blank_clip(tmp_path / "slow.mp4", n)
    slow = analyze(
        path,
        _config(bootstrap=0),
        backend=_backend(time_scale=0.25, takeoff=slow_takeoff, entry=slow_entry),
    )

    assert slow.time_base.corrected
    assert slow.fps == pytest.approx(FPS * 4, rel=0.2)
    assert slow.metrics is not None
    assert slow.metrics.horizontal_velocity_ms == pytest.approx(VX_MS, rel=0.15)
    assert slow.metrics.flight_time_s == pytest.approx((ENTRY - TAKEOFF) / FPS, rel=0.25)


def test_real_time_footage_is_not_mistaken_for_slow_motion(result):
    """The other half of the same test, and the easier one to get wrong."""
    assert not result.time_base.corrected
    assert result.fps == pytest.approx(FPS)


def test_subject_switches_counts_impossible_jumps():
    xy = np.array([[0.0, 0.0], [10.0, 0.0], [500.0, 0.0], [510.0, 0.0]])
    assert subject_switches(xy, body_scale_px=100.0) == 1
