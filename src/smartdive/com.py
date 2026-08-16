"""Whole-body centre of mass from 2-D keypoints.

Tracking a single joint (the old pipeline tracked the hip midpoint) couples the
measured trajectory to body *rotation*: a diver who pikes or extends in flight
moves their hips relative to their true centre of mass, which shows up as a
non-ballistic wobble in the signal. Summing anthropometric segment masses
removes most of that coupling and, as a bonus, averages independent per-joint
detector noise down by roughly ``sqrt(n_segments)``.
"""

from __future__ import annotations

import numpy as np

from smartdive.keypoints import (
    HEAD_KEYPOINTS,
    HEAD_MASS_FRACTION,
    SEGMENTS,
    TOTAL_MASS_FRACTION,
)

#: Minimum fraction of total body mass that must be visible for a CoM estimate.
#: Below this the estimate is dominated by whichever limbs happen to be visible
#: and is biased towards them, so we return NaN and let the gap filler decide.
DEFAULT_MIN_MASS = 0.75


def center_of_mass(
    xy: np.ndarray,
    conf: np.ndarray,
    min_conf: float = 0.3,
    min_mass: float = DEFAULT_MIN_MASS,
) -> np.ndarray:
    """Mass-weighted centre of mass in pixels, or ``(nan, nan)``.

    Parameters
    ----------
    xy:
        ``(17, 2)`` keypoint pixel coordinates in COCO-17 order.
    conf:
        ``(17,)`` per-keypoint confidences.
    min_conf:
        Keypoints below this confidence are treated as missing.
    min_mass:
        Fraction of total body mass that must be resolvable, as a guard against
        a CoM computed from a couple of limbs.
    """
    xy = np.asarray(xy, dtype=float)
    conf = np.asarray(conf, dtype=float)

    total = np.zeros(2)
    mass = 0.0

    head = [xy[i] for i in HEAD_KEYPOINTS if conf[i] >= min_conf]
    if head:
        total += HEAD_MASS_FRACTION * np.mean(head, axis=0)
        mass += HEAD_MASS_FRACTION

    for seg in SEGMENTS:
        if any(conf[i] < min_conf for i in seg.proximal):
            continue
        if any(conf[i] < min_conf for i in seg.distal):
            continue
        prox = np.mean([xy[i] for i in seg.proximal], axis=0)
        dist = np.mean([xy[i] for i in seg.distal], axis=0)
        total += seg.mass_fraction * (prox + seg.com_ratio * (dist - prox))
        mass += seg.mass_fraction

    if mass < min_mass * TOTAL_MASS_FRACTION:
        return np.array([np.nan, np.nan])
    return total / mass


def visible_mass_fraction(conf: np.ndarray, min_conf: float = 0.3) -> float:
    """Fraction of body mass resolvable from the confident keypoints."""
    conf = np.asarray(conf, dtype=float)
    mass = HEAD_MASS_FRACTION if any(conf[i] >= min_conf for i in HEAD_KEYPOINTS) else 0.0
    for seg in SEGMENTS:
        if all(conf[i] >= min_conf for i in seg.proximal + seg.distal):
            mass += seg.mass_fraction
    return mass / TOTAL_MASS_FRACTION
