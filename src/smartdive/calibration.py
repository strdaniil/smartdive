"""Pixel-to-metre calibration.

The old pipeline reported "horizontal distance: 431.2 px" and "peak height:
88.4 px". Those numbers are unusable: they depend on where the camera was
standing, so two clips of the same athlete cannot be compared, and no coach can
act on them.

Three ways to recover the scale are offered, in decreasing order of preference:

``reference``
    A known real-world length measured in pixels, e.g. the 2.5 m lane width or
    the 0.75 m block height. Use this when accuracy matters.

``stature``
    The athlete's height, combined with how many pixels across they measure at
    full extension. Needs one number from the user and is stable.

``gravity``
    Free flight has a known vertical acceleration, so if the ballistic fit
    measures ``a_y`` in px/s^2 then ``px_per_metre = a_y / 9.80665`` and the
    scene calibrates itself with no user input at all.

    This is the most attractive of the three and the least dependable, for a
    reason worth stating plainly. Over a flight of duration ``T`` the parabola
    departs from its own chord by only ``a * T**2 / 8``. A racing start filmed
    at 30 fps gives ``T`` of about 0.35 s, which puts that departure at roughly
    25 px -- the same order as the keypoint noise. The estimate is therefore
    poorly conditioned on ordinary footage, and swings by a factor of two
    depending on exactly which frames the window includes. :func:`resolve` uses
    it only when it agrees with an independent body-scale prior, and says so in
    the report either way. It becomes genuinely good on high-frame-rate video or
    for longer flights, where ``T`` is larger and the noise is relatively
    smaller.
"""

from __future__ import annotations

from dataclasses import dataclass

from smartdive.physics import BallisticFit, G


@dataclass
class Calibration:
    """A pixel-to-metre scale and where it came from."""

    px_per_metre: float
    method: str
    note: str = ""

    def to_metres(self, px: float) -> float:
        return px / self.px_per_metre

    def to_dict(self) -> dict[str, object]:
        return {
            "px_per_metre": round(self.px_per_metre, 2),
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
        note=(f"from a_y = {fit.ay:.0f} px/s^2 over frames {fit.start_frame}-{fit.end_frame}"),
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


def from_stature(body_scale_px: float, stature_m: float) -> Calibration:
    """Derive the scale from the athlete's height and their pixel body scale."""
    if body_scale_px <= 0 or stature_m <= 0:
        raise ValueError("body scale and stature must both be positive")
    return Calibration(
        px_per_metre=body_scale_px / stature_m,
        method="stature",
        note=f"body scale {body_scale_px:.0f} px = {stature_m:.2f} m",
    )


#: Ratio of the widest keypoint separation to stature, for a swimmer at full
#: extension in flight. Wrist-to-ankle in a streamline is a little longer than
#: standing height; 1.15 is a reasonable central value.
EXTENDED_BODY_RATIO = 1.15

#: Stature assumed when the user supplies none. Roughly the mean height of adult
#: competitive swimmers; the report always states when this was used.
DEFAULT_STATURE_M = 1.75


def from_body_scale(
    extended_body_scale_px: float,
    stature_m: float = DEFAULT_STATURE_M,
    assumed: bool = False,
) -> Calibration:
    """Scale from how many pixels the athlete spans at full extension."""
    if extended_body_scale_px <= 0 or stature_m <= 0:
        raise ValueError("body scale and stature must both be positive")
    note = f"{extended_body_scale_px:.0f} px at full extension"
    if assumed:
        note += f", assuming a stature of {stature_m:.2f} m"
    else:
        note += f" for a stature of {stature_m:.2f} m"
    return Calibration(
        px_per_metre=extended_body_scale_px / (stature_m * EXTENDED_BODY_RATIO),
        method="stature_assumed" if assumed else "stature",
        note=note,
    )


def resolve(
    fit: BallisticFit | None,
    extended_body_scale_px: float,
    stature_m: float | None = None,
    reference_px: float | None = None,
    reference_m: float | None = None,
    gravity_agreement: float = 1.35,
) -> tuple[Calibration, list[str]]:
    """Pick the best available scale, and explain the choice.

    Preference order is reference, then stature, then gravity -- but gravity is
    accepted only if it lands within ``gravity_agreement`` of the body-scale
    prior, since a wildly disagreeing gravity estimate means the flight window
    was too short or too noisy to measure curvature, not that the athlete is
    four metres tall.
    """
    notes: list[str] = []

    if reference_px and reference_m:
        return from_reference(reference_px, reference_m), notes

    prior = from_body_scale(
        extended_body_scale_px,
        stature_m if stature_m else DEFAULT_STATURE_M,
        assumed=stature_m is None,
    )

    if fit is None or fit.ay <= 0:
        return prior, notes

    gravity = from_gravity(fit)
    ratio = gravity.px_per_metre / prior.px_per_metre
    if 1 / gravity_agreement <= ratio <= gravity_agreement:
        gravity.note += f"; agrees with the body-scale prior to within {abs(1 - ratio):.0%}"
        return gravity, notes

    notes.append(
        f"gravity calibration ({gravity.px_per_metre:.0f} px/m) disagrees with the "
        f"body-scale prior ({prior.px_per_metre:.0f} px/m) by {abs(1 - ratio):.0%}; "
        "the flight is too short at this frame rate to measure curvature "
        "reliably, so the prior was used. Pass --reference-px/--reference-m for "
        "an exact scale."
    )
    return prior, notes


def plausibility(calibration: Calibration, body_scale_px: float) -> tuple[bool, str]:
    """Sanity-check a calibration against the size of a human being.

    ``body_scale_px`` is the largest distance between keypoints, so it spans an
    extended athlete: roughly stature for someone standing, up to about 1.3x
    stature fingertip-to-toe in a streamline. Anything outside 1.2-2.8 m means
    the scale is wrong, and every metric derived from it should be distrusted.
    """
    implied = body_scale_px / calibration.px_per_metre
    if 1.2 <= implied <= 2.8:
        return True, f"implied athlete extent {implied:.2f} m is plausible"
    return False, (
        f"implied athlete extent {implied:.2f} m is outside the plausible "
        "1.2-2.8 m range; treat metric values with suspicion"
    )
