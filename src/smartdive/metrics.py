"""Dive metrics in SI units.

Everything here is derived from the ballistic fit and the calibration, so every
number carries physical meaning and is comparable across clips, athletes and
camera positions.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from itertools import pairwise

import numpy as np

from smartdive.calibration import Calibration, TimeBase
from smartdive.keypoints import (
    LEFT_ANKLE,
    LEFT_HIP,
    LEFT_KNEE,
    RIGHT_ANKLE,
    RIGHT_HIP,
    RIGHT_KNEE,
)
from smartdive.physics import BallisticFit


@dataclass
class DiveMetrics:
    """Kinematic summary of one start."""

    takeoff_frame: int
    entry_frame: int
    flight_time_s: float
    takeoff_speed_ms: float
    takeoff_angle_deg: float
    horizontal_velocity_ms: float
    vertical_velocity_ms: float
    flight_distance_m: float
    peak_height_m: float
    entry_angle_deg: float
    px_per_metre: float
    calibration_method: str
    fit_r_squared: float
    fit_residual_rms_px: float
    effective_fps: float
    block_time_s: float | None = None
    set_knee_angle_deg: float | None = None
    slow_motion_factor: float | None = None

    def to_dict(self) -> dict[str, object]:
        return {k: v for k, v in asdict(self).items()}

    def format_table(self) -> str:
        """Human-readable summary."""
        rows = [
            ("Flight time", f"{self.flight_time_s:.3f} s"),
            ("Takeoff speed", f"{self.takeoff_speed_ms:.2f} m/s"),
            ("Takeoff angle", f"{self.takeoff_angle_deg:+.1f} deg"),
            ("  horizontal", f"{self.horizontal_velocity_ms:.2f} m/s"),
            ("  vertical", f"{self.vertical_velocity_ms:+.2f} m/s"),
            ("Flight distance", f"{self.flight_distance_m:.2f} m"),
            ("Peak height above takeoff", f"{self.peak_height_m:.2f} m"),
            ("Entry angle", f"{self.entry_angle_deg:.1f} deg"),
            ("Airborne frames", f"{self.takeoff_frame} to {self.entry_frame}"),
        ]
        if self.slow_motion_factor is not None:
            rows.insert(
                0,
                (
                    "Capture rate",
                    f"{self.effective_fps:g} fps"
                    f" (clip is {self.slow_motion_factor:.2g}x slow motion)",
                ),
            )
        if self.block_time_s is not None:
            rows.insert(0, ("Block time", f"{self.block_time_s:.3f} s"))
        if self.set_knee_angle_deg is not None:
            rows.append(("Front knee angle at set", f"{self.set_knee_angle_deg:.1f} deg"))
        rows += [
            ("Scale", f"{self.px_per_metre:.1f} px/m via {self.calibration_method}"),
            (
                "Fit quality",
                f"R^2 = {self.fit_r_squared:.4f}, RMS {self.fit_residual_rms_px:.1f} px",
            ),
        ]
        width = max(len(k) for k, _ in rows)
        return "\n".join(f"{k.ljust(width)}  {v}" for k, v in rows)


def integrate_metres(
    xy: np.ndarray,
    frames: np.ndarray,
    calibration: Calibration,
    axis: int | None = None,
) -> float:
    """Path length (or one axis of displacement) in metres, scale by scale.

    When the athlete's distance from the camera changes during the flight, a
    single pixels-per-metre is wrong at one end of it or the other, so each
    step is converted using the scale in force where that step happened. With a
    constant calibration this reduces exactly to dividing by it once.
    """
    good = ~np.isnan(xy[:, 0])
    xy, frames = xy[good], np.asarray(frames)[good]
    if len(xy) < 2:
        return 0.0
    steps = np.diff(xy, axis=0)
    scales = np.array([calibration.at(int((a + b) // 2)) for a, b in pairwise(frames)])
    if axis is None:
        return float(np.sum(np.linalg.norm(steps, axis=1) / scales))
    return float(abs(np.sum(steps[:, axis] / scales)))


def compute_metrics(
    fit: BallisticFit,
    calibration: Calibration,
    trajectory_px: np.ndarray | None = None,
    frames: np.ndarray | None = None,
    block_time_s: float | None = None,
    set_knee_angle_deg: float | None = None,
    time_base: TimeBase | None = None,
) -> DiveMetrics:
    """Convert a ballistic fit plus a scale into SI dive metrics."""
    ppm = calibration.at(fit.start_frame)

    vx = fit.vx / ppm
    vy = fit.vy / ppm  # image y grows downward
    speed = math.hypot(vx, vy)

    # Displacement along the modelled path, which is immune to the endpoint
    # noise that a raw first-to-last-sample difference would inherit.
    start = fit.position(np.array([0.0]))[0]
    end = fit.position(np.array([fit.duration]))[0]
    horizontal = abs(end[0] - start[0]) / ppm

    # Peak height, measured from takeoff. Zero if the athlete never rose, which
    # is normal for a flat racing start off a raised block.
    apex_t = fit.apex_time()
    apex_t = min(apex_t, fit.duration)
    apex_y = fit.position(np.array([apex_t]))[0][1]
    peak_height = max(0.0, (start[1] - apex_y) / ppm)

    # Entry angle: direction of travel at the end of the flight window.
    vy_entry = fit.vy + fit.ay * fit.duration
    entry_angle = math.degrees(math.atan2(vy_entry, abs(fit.vx))) if fit.vx else 90.0

    if trajectory_px is not None and len(trajectory_px) >= 2:
        window_frames = (
            np.arange(fit.start_frame, fit.start_frame + len(trajectory_px))
            if frames is None
            else frames
        )
        measured = integrate_metres(trajectory_px, window_frames, calibration, axis=0)
        horizontal = max(horizontal, measured)

    return DiveMetrics(
        takeoff_frame=fit.start_frame,
        entry_frame=fit.end_frame,
        flight_time_s=fit.duration,
        takeoff_speed_ms=speed,
        takeoff_angle_deg=fit.angle_deg,
        horizontal_velocity_ms=abs(vx),
        vertical_velocity_ms=-vy,  # report upward as positive
        flight_distance_m=horizontal,
        peak_height_m=peak_height,
        entry_angle_deg=entry_angle,
        px_per_metre=ppm,
        calibration_method=calibration.method,
        fit_r_squared=fit.r_squared,
        fit_residual_rms_px=fit.residual_rms,
        effective_fps=fit.fps,
        block_time_s=block_time_s,
        set_knee_angle_deg=set_knee_angle_deg,
        slow_motion_factor=(
            None if time_base is None or not time_base.corrected else 1.0 / time_base.factor
        ),
    )


def joint_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Interior angle at ``b`` formed by ``a-b-c``, in degrees."""
    ba = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    bc = np.asarray(c, dtype=float) - np.asarray(b, dtype=float)
    na = float(np.linalg.norm(ba))
    nc = float(np.linalg.norm(bc))
    if na == 0 or nc == 0:
        return float("nan")
    cosine = float(np.dot(ba, bc)) / (na * nc)
    return float(math.degrees(math.acos(max(-1.0, min(1.0, cosine)))))


def front_knee_angle(xy: np.ndarray, conf: np.ndarray, min_conf: float = 0.3) -> tuple[float, str]:
    """Knee angle of the *leading* leg, and which side that was.

    The front leg is whichever ankle is further along the direction the athlete
    is facing; facing direction is inferred from the hip-to-ankle offset, so the
    caller does not have to pass a ``facing`` flag the way the old script did.
    """
    needed = (LEFT_HIP, RIGHT_HIP, LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE)
    if any(conf[i] < min_conf for i in needed):
        return float("nan"), "unknown"

    hip_mid_x = (xy[LEFT_HIP, 0] + xy[RIGHT_HIP, 0]) / 2.0
    ankle_mid_x = (xy[LEFT_ANKLE, 0] + xy[RIGHT_ANKLE, 0]) / 2.0
    # On the blocks the athlete leans over their front foot, so the ankles sit
    # ahead of the hips in the direction of travel.
    forward = 1.0 if ankle_mid_x >= hip_mid_x else -1.0

    left_lead = forward * xy[LEFT_ANKLE, 0] > forward * xy[RIGHT_ANKLE, 0]
    if left_lead:
        angle = joint_angle(xy[LEFT_HIP], xy[LEFT_KNEE], xy[LEFT_ANKLE])
        return angle, "left"
    angle = joint_angle(xy[RIGHT_HIP], xy[RIGHT_KNEE], xy[RIGHT_ANKLE])
    return angle, "right"


#: Coaching band for the front knee in the set position. Sourced from the
#: original project's heuristic and kept for continuity, not as clinical advice.
KNEE_ANGLE_BAND = (125.0, 150.0)


def knee_angle_feedback(angle: float) -> str:
    """One-line coaching note for a set-position knee angle."""
    if math.isnan(angle):
        return "front knee not measurable in the set position"
    low, high = KNEE_ANGLE_BAND
    if angle < low:
        return f"{angle:.0f} deg: crouched below the {low:.0f}-{high:.0f} deg band"
    if angle > high:
        return f"{angle:.0f} deg: straighter than the {low:.0f}-{high:.0f} deg band"
    return f"{angle:.0f} deg: within the {low:.0f}-{high:.0f} deg band"
