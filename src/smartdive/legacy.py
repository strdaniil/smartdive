"""A faithful re-implementation of the original SmartDive v1/v2 pipeline.

This exists so the benchmark is honest. Rather than quoting numbers from memory,
:mod:`smartdive.benchmark` runs *this* code and the new pipeline over the same
frames, from the same detections where possible, and compares the results.

The behaviour reproduced here, and the file it came from:

* ``argmax`` over per-frame detector confidence to choose the subject
  (``process_video.py:run_pose_estimation``) -- no association across frames.
* A fixed 0.5 keypoint-confidence gate, emitting NaN below it, with no gap
  filling, outlier rejection or smoothing (``utils.py:get_landmark_point``).
* A degree-3 polynomial in ``t`` fit to ``x`` and ``y`` independently over the
  whole clip (``polyfit.py:fit_polynomial_time_param``).
* Metrics in pixels, with "flight time" defined as the span between the first
  and last frame that produced any measurement at all
  (``compute_metrics.py:compute_metrics``).

Nothing here is a strawman: the constants, thresholds and formulas are the ones
in the original repository.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from smartdive.backends.base import FrameDetections
from smartdive.keypoints import NAME_TO_INDEX, PAIR_LANDMARKS

#: The original hard-coded confidence gate (utils.py).
LEGACY_MIN_CONF = 0.5
#: The original polynomial order (polyfit.py, and the README's --fit-degree).
LEGACY_POLY_DEGREE = 3


def legacy_landmark(xy: np.ndarray, conf: np.ndarray, landmark: str) -> np.ndarray:
    """``utils.get_landmark_point``: a fixed 0.5 gate, NaN otherwise."""
    if landmark in PAIR_LANDMARKS:
        i, j = PAIR_LANDMARKS[landmark]
        if conf[i] > LEGACY_MIN_CONF and conf[j] > LEGACY_MIN_CONF:
            return (xy[i] + xy[j]) / 2.0
        return np.array([np.nan, np.nan])
    if landmark in NAME_TO_INDEX:
        i = NAME_TO_INDEX[landmark]
        if conf[i] > LEGACY_MIN_CONF:
            return xy[i].astype(float).copy()
        return np.array([np.nan, np.nan])
    raise ValueError(f"legacy pipeline cannot resolve landmark {landmark!r}")


def legacy_trajectory(detections: list[FrameDetections], landmark: str = "hips") -> np.ndarray:
    """Per-frame ``argmax``-confidence subject selection, as in ``process_video.py``."""
    out = np.full((len(detections), 2), np.nan)
    for i, frame in enumerate(detections):
        if not frame.people:
            continue
        best = max(frame.people, key=lambda p: p.score)
        out[i] = legacy_landmark(best.xy, best.conf, landmark)
    return out


@dataclass
class LegacyMetrics:
    """What the original ``compute_metrics.py`` printed."""

    flight_time_s: float
    horizontal_distance_px: float
    peak_height_px: float
    poly_x: np.ndarray
    poly_y: np.ndarray
    poly_residual_rms_px: float

    def format_table(self) -> str:
        return "\n".join(
            [
                f"Flight time                {self.flight_time_s:.3f} s",
                f"Horizontal distance        {self.horizontal_distance_px:.1f} px",
                f"Peak height above takeoff  {self.peak_height_px:.1f} px",
                f"Cubic fit residual (RMS)   {self.poly_residual_rms_px:.1f} px",
            ]
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "flight_time_s": round(self.flight_time_s, 3),
            "horizontal_distance_px": round(self.horizontal_distance_px, 1),
            "peak_height_px": round(self.peak_height_px, 1),
            "poly_residual_rms_px": round(self.poly_residual_rms_px, 2),
        }


def legacy_metrics(trajectory: np.ndarray, fps: float) -> LegacyMetrics | None:
    """Reproduce ``compute_metrics.py`` and ``polyfit.py`` on a trajectory.

    Note what "flight time" means here: the elapsed time between the first and
    last frame in which *anything* was measured. If the detector saw a person in
    frame 0 and in the final frame, this reports the length of the clip.
    """
    valid = ~np.isnan(trajectory[:, 0])
    if not valid.any():
        return None

    start = int(np.argmax(valid))
    end = len(trajectory) - 1 - int(np.argmax(valid[::-1]))

    flight_time = (end - start) / fps
    horizontal = float(abs(trajectory[end, 0] - trajectory[start, 0]))
    segment = trajectory[start : end + 1, 1]
    peak = float(trajectory[start, 1] - np.nanmin(segment))

    idx = np.where(valid)[0]
    t = idx.astype(float)
    degree = min(LEGACY_POLY_DEGREE, max(1, len(idx) - 1))
    cx = np.polyfit(t, trajectory[idx, 0], degree)
    cy = np.polyfit(t, trajectory[idx, 1], degree)
    model = np.stack([np.polyval(cx, t), np.polyval(cy, t)], axis=-1)
    residual = float(np.sqrt(np.mean(np.sum((model - trajectory[idx]) ** 2, axis=1))))

    return LegacyMetrics(
        flight_time_s=flight_time,
        horizontal_distance_px=horizontal,
        peak_height_px=peak,
        poly_x=cx,
        poly_y=cy,
        poly_residual_rms_px=residual,
    )
