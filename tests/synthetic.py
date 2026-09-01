"""A synthetic athlete, for tests that need a whole pipeline but no weights.

The pose backend is the one part of this project that cannot run in CI, so the
tests stub it. Stubbing it well matters more than it might look: the pipeline
now measures the athlete's *stature* from their individual bone lengths in order
to calibrate the scene, so a stub that emits seventeen points on a diagonal line
does not merely fail to exercise that code, it makes the scale meaningless and
every metric derived from it untestable.

So the skeleton built here is anatomically proportioned, using the same Winter
segment ratios the calibration reads back out. A figure generated at a given
pixel stature should be measured at that pixel stature, which makes the round
trip, generate at a known scale, analyse, compare, a real test of the
calibration rather than a tautology.
"""

from __future__ import annotations

import numpy as np

from smartdive.backends.base import FrameDetections, PersonDetection, PoseBackend
from smartdive.keypoints import (
    LEFT_ANKLE,
    LEFT_EAR,
    LEFT_ELBOW,
    LEFT_EYE,
    LEFT_HIP,
    LEFT_KNEE,
    LEFT_SHOULDER,
    LEFT_WRIST,
    NOSE,
    NUM_KEYPOINTS,
    RIGHT_ANKLE,
    RIGHT_EAR,
    RIGHT_ELBOW,
    RIGHT_EYE,
    RIGHT_HIP,
    RIGHT_KNEE,
    RIGHT_SHOULDER,
    RIGHT_WRIST,
)

#: Keypoint offsets from the hip, as fractions of stature, for a body standing
#: upright and facing right. ``+x`` is forwards, ``+y`` is *up*; the caller
#: flips y into image convention. Consistent with the ratios in
#: :mod:`smartdive.anthro`: shoulders 0.27 above the hip, thigh 0.245, shank
#: 0.246, upper arm 0.186, forearm 0.146.
_SKELETON: dict[int, tuple[float, float]] = {
    NOSE: (0.045, 0.375),
    LEFT_EYE: (0.035, 0.395),
    RIGHT_EYE: (0.035, 0.395),
    LEFT_EAR: (-0.010, 0.390),
    RIGHT_EAR: (-0.010, 0.390),
    LEFT_SHOULDER: (0.0, 0.270),
    RIGHT_SHOULDER: (0.0, 0.270),
    LEFT_ELBOW: (0.010, 0.084),
    RIGHT_ELBOW: (0.010, 0.084),
    LEFT_WRIST: (0.020, -0.062),
    RIGHT_WRIST: (0.020, -0.062),
    LEFT_HIP: (0.0, 0.0),
    RIGHT_HIP: (0.0, 0.0),
    LEFT_KNEE: (0.0, -0.245),
    RIGHT_KNEE: (0.0, -0.245),
    LEFT_ANKLE: (0.0, -0.491),
    RIGHT_ANKLE: (0.0, -0.491),
}


def skeleton(
    hip: tuple[float, float],
    stature_px: float,
    rotation_deg: float = 0.0,
    facing: float = 1.0,
) -> np.ndarray:
    """``(17, 2)`` keypoints for a proportioned body, in image coordinates.

    ``rotation_deg`` tips the whole figure about its hip: 0 is upright, 90 is
    horizontal head-first in the direction it faces, which is what a diver looks
    like in flight.
    """
    theta = np.radians(rotation_deg)
    cos, sin = np.cos(theta), np.sin(theta)
    out = np.zeros((NUM_KEYPOINTS, 2))
    for index, (forward, up) in _SKELETON.items():
        fx, fy = forward * facing * stature_px, up * stature_px
        # Rotate in a right-handed frame, then flip y for image coordinates.
        rx = fx * cos - fy * sin * facing
        ry = fx * sin * facing + fy * cos
        out[index] = (hip[0] + rx, hip[1] - ry)
    return out


def detection(
    hip: tuple[float, float],
    stature_px: float,
    rotation_deg: float = 0.0,
    score: float = 0.8,
    submerged: bool = False,
    noise: float = 0.0,
    rng: np.random.Generator | None = None,
) -> PersonDetection:
    """One :class:`PersonDetection` for a proportioned synthetic body."""
    xy = skeleton(hip, stature_px, rotation_deg)
    if noise and rng is not None:
        xy = xy + rng.normal(0.0, noise, xy.shape)
    conf = np.full(NUM_KEYPOINTS, 0.9)
    if submerged:
        # Everything from the hips down disappears below the surface, which is
        # what the real footage shows: the visible mass fraction falls off a
        # cliff the moment the athlete breaks the water.
        conf[[LEFT_HIP, RIGHT_HIP, LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE]] = 0.05
    visible = xy[conf >= 0.3]
    return PersonDetection(
        box=np.array(
            [visible[:, 0].min(), visible[:, 1].min(), visible[:, 0].max(), visible[:, 1].max()]
        ),
        score=score,
        xy=xy,
        conf=conf,
    )


class ScriptedBackend(PoseBackend):
    """Replays a scripted dive plus a stationary, better-lit bystander.

    The bystander exists to reproduce the original project's headline failure:
    they are consistently the *more confident* detection, so any pipeline that
    picks the subject by per-frame confidence measures them instead of the
    athlete.
    """

    name = "scripted"

    def __init__(
        self,
        takeoff: int,
        entry: int,
        fps: float,
        px_per_m: float,
        stature_px: float,
        vx_ms: float = 5.0,
        vy_ms: float = 1.2,
        origin: tuple[float, float] = (300.0, 500.0),
        noise: float = 0.0,
        seed: int = 0,
        time_scale: float = 1.0,
    ) -> None:
        from smartdive.physics import G

        self.takeoff, self.entry = takeoff, entry
        self.fps, self.px_per_m, self.stature_px = fps, px_per_m, stature_px
        self.vx_ms, self.vy_ms = vx_ms, vy_ms
        self.origin = origin
        self.noise = noise
        self.rng = np.random.default_rng(seed)
        #: Real seconds per frame divided by ``1 / fps``: below 1, the clip is
        #: slow motion and its container frame rate is lying.
        self.time_scale = time_scale
        self._g = G

    def _athlete(self, index: int) -> PersonDetection:
        x0, y0 = self.origin
        if index <= self.takeoff:
            return detection(
                (x0, y0),
                self.stature_px,
                rotation_deg=55.0,
                score=0.78,
                noise=self.noise,
                rng=self.rng,
            )
        t = (index - self.takeoff) / self.fps * self.time_scale
        x = x0 + self.vx_ms * self.px_per_m * t
        y = y0 - self.vy_ms * self.px_per_m * t + 0.5 * self._g * self.px_per_m * t**2
        return detection(
            (x, y),
            self.stature_px,
            rotation_deg=90.0,
            score=0.78,
            submerged=index > self.entry,
            noise=self.noise,
            rng=self.rng,
        )

    def detect(self, frame: np.ndarray, index: int) -> FrameDetections:
        bystander = detection((1500.0, 520.0), self.stature_px * 0.92, score=0.95)
        return FrameDetections(index=index, people=[self._athlete(index), bystander])

    def describe(self) -> dict[str, object]:
        return {"backend": self.name, "scripted": True}
