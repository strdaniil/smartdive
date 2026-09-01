"""Pixel-to-metre calibration, and getting the clock right.

The original pipeline reported "horizontal distance: 431.2 px". That number is
unusable: it depends on where the camera was standing, so two clips of the same
athlete cannot be compared and no coach can act on it. Everything here exists to
turn pixels into metres, and to say honestly how well that conversion is known.

Three sources of scale are available, in decreasing order of preference:

``reference``
    A known real-world length measured in pixels, lane width, block height,
    a stick laid on the deck. Use this when accuracy matters.

``stature``
    The athlete's height, combined with how many pixels they measure. See
    :mod:`smartdive.anthro` for how the pixel measurement is made; briefly, it
    is the median of several independent bone-length estimates rather than the
    pose-dependent bounding extent the previous version used.

``gravity``
    Free flight has a known vertical acceleration, so if the ballistic fit
    measures ``a_y`` in px/s^2 then ``px_per_metre = a_y / 9.80665`` and the
    scene calibrates itself with no user input at all.

    This is the most attractive of the three and, on ordinary 30 fps footage,
    the least dependable, for a reason worth stating plainly rather than
    hiding. Over a flight of duration ``T`` the parabola departs from its own
    chord by only ``a * T**2 / 8``. A racing start filmed at 30 fps gives ``T``
    of about 0.35 s, putting that departure at roughly 25 px, the same order as
    the keypoint noise. Measured on the sample clips the free-fit gravity swings
    between 120 and 218 px/m depending on which frames the window includes: a
    factor of 1.8 on a quantity that should be a constant of the universe. So it
    is used only as a cross-check, never as the primary scale, unless the
    flight is long enough in frames for the curvature to be real information.

There is also a fourth quantity most video tools assume rather than measure: the
**time base**. A clip's container frame rate is what it will be *played* at,
which for anything shot on a phone's slow-motion mode is not what it was shot
at. Times, speeds and accelerations are all wrong by that factor, and nothing
in the image says so. :func:`resolve_time_base` recovers it, because gravity
provides an independent clock, see its docstring.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from smartdive.anthro import DEFAULT_STATURE_M, BodySize, check_stature
from smartdive.physics import BallisticFit, G

__all__ = [
    "DEFAULT_STATURE_M",
    "Calibration",
    "TimeBase",
    "check_stature",
    "from_body_size",
    "from_gravity",
    "from_reference",
    "plausibility",
    "resolve",
    "resolve_time_base",
]


@dataclass
class Calibration:
    """A pixel-to-metre scale, where it came from, and how it varies in frame.

    ``scale_series`` carries a *per-frame* pixels-per-metre for footage where
    the athlete's distance from the camera changes materially during the dive.
    A start filmed obliquely down the lane rather than square-on has the athlete
    receding as they fly, and a single scale is then wrong at one end of the
    flight or the other. Where it is None the scale is constant.
    """

    px_per_metre: float
    method: str
    note: str = ""
    stature_m: float | None = None
    scale_series: np.ndarray | None = field(default=None, repr=False)

    def at(self, frame: int) -> float:
        """Pixels per metre at a particular frame."""
        if self.scale_series is None or not 0 <= frame < len(self.scale_series):
            return self.px_per_metre
        value = self.scale_series[frame]
        return float(value) if np.isfinite(value) and value > 0 else self.px_per_metre

    def to_metres(self, px: float, frame: int | None = None) -> float:
        return px / (self.px_per_metre if frame is None else self.at(frame))

    @property
    def varies(self) -> bool:
        return self.scale_series is not None

    def depth_change(self) -> float:
        """Fractional change in scale across the frames it is defined for."""
        if self.scale_series is None:
            return 0.0
        values = self.scale_series[np.isfinite(self.scale_series)]
        if len(values) < 2 or values.min() <= 0:
            return 0.0
        return float(values.max() / values.min() - 1.0)

    def to_dict(self) -> dict[str, object]:
        return {
            "px_per_metre": round(self.px_per_metre, 2),
            "method": self.method,
            "note": self.note,
            "stature_m": self.stature_m,
            "varies_with_depth": self.varies,
            "depth_change": round(self.depth_change(), 3),
        }


@dataclass
class TimeBase:
    """The clip's real capture rate, which is not always its container rate."""

    container_fps: float
    effective_fps: float
    factor: float  # real seconds per second of container time
    method: str  # "container" or "gravity"
    note: str = ""

    @property
    def corrected(self) -> bool:
        return self.method != "container"

    def to_dict(self) -> dict[str, object]:
        return {
            "container_fps": round(self.container_fps, 3),
            "effective_fps": round(self.effective_fps, 3),
            "factor": round(self.factor, 4),
            "method": self.method,
            "note": self.note,
        }


def from_gravity(fit: BallisticFit) -> Calibration:
    """Derive the scale from the measured vertical acceleration of free flight."""
    if fit.ay <= 0:
        raise ValueError(
            "cannot calibrate from gravity: the fitted vertical acceleration "
            "is not downward, so this window is not free flight"
        )
    return Calibration(
        px_per_metre=fit.ay / G,
        method="gravity",
        note=f"from a_y = {fit.ay:.0f} px/s^2 over frames {fit.start_frame}-{fit.end_frame}",
    )


def from_reference(pixels: float, metres: float, label: str = "reference") -> Calibration:
    """Derive the scale from a known length measured in the image."""
    if pixels <= 0 or metres <= 0:
        raise ValueError("reference length must be positive in both pixels and metres")
    return Calibration(
        px_per_metre=pixels / metres,
        method="reference",
        note=f"{label}: {pixels:.1f} px = {metres:.3f} m",
    )


def from_body_size(
    size: BodySize,
    stature_m: float | None = None,
    scale_series: np.ndarray | None = None,
) -> Calibration:
    """Scale from the athlete's measured pixel stature and their real height."""
    if not np.isfinite(size.stature_px) or size.stature_px <= 0:
        raise ValueError("body size could not be measured, so stature calibration is unavailable")
    assumed = stature_m is None
    height = DEFAULT_STATURE_M if assumed else check_stature(stature_m)

    px_per_metre = size.stature_px / height
    note = f"{size.stature_px:.0f} px of stature from {len(size.per_segment)} body segments"
    note += f" (spread {size.dispersion:.0%})"
    note += f", assuming {height:.2f} m" if assumed else f" for a stature of {height:.2f} m"

    series = None
    if scale_series is not None:
        series = np.asarray(scale_series, dtype=float) * px_per_metre

    return Calibration(
        px_per_metre=px_per_metre,
        method="stature_assumed" if assumed else "stature",
        note=note,
        stature_m=height,
        scale_series=series,
    )


def resolve(
    size: BodySize,
    stature_m: float | None = None,
    reference_px: float | None = None,
    reference_m: float | None = None,
    fit: BallisticFit | None = None,
    relative_size: np.ndarray | None = None,
    gravity_agreement: float = 1.35,
) -> tuple[Calibration, list[str]]:
    """Pick the best available scale, and explain the choice.

    Gravity is never preferred over a measured length or a stature, and is
    reported as a cross-check either way: an agreement between two independent
    routes to the same number is worth having, and a disagreement is worth
    saying out loud rather than burying.
    """
    notes: list[str] = []

    if reference_px and reference_m:
        calibration = from_reference(reference_px, reference_m)
        if relative_size is not None:
            calibration.scale_series = np.asarray(relative_size, float) * calibration.px_per_metre
    else:
        calibration = from_body_size(size, stature_m, relative_size)

    if fit is not None and fit.ay > 0:
        gravity = from_gravity(fit)
        ratio = gravity.px_per_metre / calibration.px_per_metre
        if 1 / gravity_agreement <= ratio <= gravity_agreement:
            notes.append(
                f"gravity cross-check: {gravity.px_per_metre:.0f} px/m, within "
                f"{abs(1 - ratio):.0%} of the {calibration.method} scale "
                f"({calibration.px_per_metre:.0f} px/m)"
            )
        else:
            notes.append(
                f"gravity cross-check disagrees: {gravity.px_per_metre:.0f} px/m against "
                f"{calibration.px_per_metre:.0f} px/m from {calibration.method}, a "
                f"{abs(1 - ratio):.0%} difference. Over a flight this short the curvature "
                "the gravity estimate depends on is comparable to the keypoint noise, so "
                "the disagreement is expected and the body-scale figure was used."
            )

    return calibration, notes


#: Capture rates a camera plausibly shoots at. A recovered rate close to one of
#: these is snapped to it, because "shot at 120 fps" is a far more likely
#: explanation for a factor of four than "shot at 118.4 fps".
COMMON_CAPTURE_FPS: tuple[float, ...] = (24.0, 25.0, 30.0, 48.0, 50.0, 60.0, 120.0, 240.0, 480.0)


def resolve_time_base(
    container_fps: float,
    fit: BallisticFit | None,
    px_per_metre: float,
    n_points: int = 0,
    min_points: int = 12,
    trigger: float = 0.45,
    max_relative_stderr: float = 0.25,
    snap_tolerance: float = 0.15,
) -> tuple[TimeBase, list[str]]:
    """Recover the clip's real capture rate from the acceleration of gravity.

    A clip shot in a phone's slow-motion mode carries a container frame rate of
    30, the rate it will be *played* at, while having been captured at 120 or
    240. Nothing in the pixels says so, and every time-derived quantity is then
    wrong: a flight that really lasted 0.35 s is reported as 1.4 s, the speeds
    are a quarter of the truth, and the numbers are silently useless.

    Gravity gives an independent clock. If ``k`` is the number of real seconds
    per second of container time, a body in free fall shows an apparent vertical
    acceleration of ``g * px_per_metre * k**2``, because the acceleration term
    goes as time squared. Having pinned ``px_per_metre`` from the athlete's own
    body, ``k`` follows::

        k = sqrt(a_y_observed / (g * px_per_metre))

    and the true capture rate is ``container_fps / k``.

    The estimate is only taken when it can be trusted, which means all of:

    * the flight is long enough in frames to fit a curvature at all, which,
      satisfyingly, is exactly the condition slow-motion footage guarantees and
      normal footage does not;
    * the fitted acceleration is *itself* well determined. This is the criterion
      that matters, and it is checked directly, from the standard error of the
      quadratic coefficient rather than from a proxy for it. On ordinary 30 fps
      footage that standard error comes out at 40% or more of the coefficient,
      which is exactly why gravity cannot be used as a scale there, and it is
      also why such a clip must not be allowed to declare itself slow motion on
      the strength of a number it cannot measure;
    * the correction is large. A 20% discrepancy is within the ordinary error of
      a gravity estimate even when it passes the test above; a factor of four is
      not something noise produces.

    Anything smaller is left alone rather than "corrected" into the noise.
    """
    notes: list[str] = []
    base = TimeBase(container_fps, container_fps, 1.0, "container")

    if fit is None or fit.ay <= 0 or not np.isfinite(px_per_metre) or px_per_metre <= 0:
        return base, notes
    if n_points < min_points:
        return base, notes
    if not np.isfinite(fit.ay_stderr) or fit.ay_stderr > max_relative_stderr * fit.ay:
        return base, notes

    factor = float(np.sqrt(fit.ay / (G * px_per_metre)))
    if not np.isfinite(factor) or factor <= 0:
        return base, notes
    if abs(factor - 1.0) <= trigger:
        return base, notes

    effective = container_fps / factor
    snapped = min(COMMON_CAPTURE_FPS, key=lambda c: abs(c - effective))
    if abs(snapped - effective) / effective <= snap_tolerance:
        note = (
            f"gravity implies a capture rate of {effective:.0f} fps against the "
            f"container's {container_fps:g}; snapped to the standard {snapped:g} fps"
        )
        effective = snapped
        factor = container_fps / effective
    else:
        note = (
            f"gravity implies a capture rate of {effective:.0f} fps against the "
            f"container's {container_fps:g} fps"
        )

    notes.append(
        note + ". This clip is slow motion: times and speeds are reported at the "
        "capture rate, not the playback rate. Times would otherwise be wrong by "
        f"a factor of {container_fps / effective:.2f}."
    )
    return TimeBase(container_fps, effective, container_fps / effective, "gravity", note), notes


def plausibility(calibration: Calibration, stature_px: float) -> tuple[bool, str]:
    """Sanity-check a calibration against the size of a human being."""
    if not np.isfinite(stature_px) or stature_px <= 0:
        return True, "no body measurement to check the scale against"
    implied = stature_px / calibration.px_per_metre
    if 1.2 <= implied <= 2.3:
        return True, f"implied athlete stature {implied:.2f} m is plausible"
    return False, (
        f"implied athlete stature {implied:.2f} m is outside the plausible "
        "1.2-2.3 m range; treat metric values with suspicion"
    )


def plausible_start(speed_ms: float, flight_time_s: float) -> list[str]:
    """Flag metrics that fall outside what a swimming start can physically be.

    A measurement tool that cannot recognise its own nonsense is worse than one
    that produces none, because the nonsense looks exactly like a result. These
    bands are wide on purpose, they are meant to catch a broken scale or a
    mis-detected phase, not to grade an athlete.
    """
    warnings: list[str] = []
    if np.isfinite(speed_ms) and not (1.5 <= speed_ms <= 8.0):
        warnings.append(
            f"takeoff speed of {speed_ms:.1f} m/s is outside the 1.5-8.0 m/s range a "
            "swimming start can produce; the pixel-to-metre scale or the flight "
            "window is probably wrong"
        )
    if np.isfinite(flight_time_s) and not (0.15 <= flight_time_s <= 0.60):
        warnings.append(
            f"flight time of {flight_time_s:.2f} s is outside the 0.15-0.60 s range a "
            "swimming start can produce; check the detected takeoff and entry frames "
            "on the overlay, and use --flight-window to set them if they are wrong"
        )
    return warnings
