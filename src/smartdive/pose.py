"""Whole-skeleton time series: completion, conditioning, and landmark extraction.

The previous version of this pipeline conditioned the *landmark*, one ``(N, 2)``
trajectory, and computed the centre of mass first, frame by frame, from
whichever keypoints happened to be confident in that frame. That ordering has a
failure mode which is easy to miss and expensive to live with.

The centre of mass is a weighted sum over body segments. If a segment drops out
of one frame, the sum is taken over a *different set of segments* in that frame,
so the estimate moves even when the athlete does not. Losing one shank shifts
the computed centre of mass by around three percent of body length towards the
head, which at typical framing is eight pixels: comparable to the entire
curvature signal a 0.35 s flight contains. The trajectory then looks noisy, and
worse, the noise is correlated with body configuration rather than random, so it
does not average out over the fit. The obvious defence, refusing to compute a
centre of mass unless most of the body is visible, trades the bias for
dropouts, and on the sample clips it threw away nearly half of every clip.

So the order is inverted here. Each of the 17 keypoints is treated as its own
time series and completed and conditioned independently, and only then are the
segment masses summed. The set of segments contributing to the centre of mass is
then the same in every frame, which removes the composition bias entirely, and
each keypoint's own dropouts are filled from its own neighbours in time, which
is the right prior, since a knee that was in a particular place two frames ago
and two frames hence was somewhere between the two in between.

Two completion mechanisms run before conditioning:

**Contralateral mirroring.** In a lateral view, which is how a racing start is
almost always filmed, the two legs and the two arms project to nearly the same
place. When one side's keypoint is missing and its mirror is confident, the
mirror stands in for it. This is what rescues the frames where the detector sees
only the near half of a body.

**Temporal interpolation.** Short gaps in a single keypoint are filled by
monotone cubic interpolation, exactly as for the landmark trajectory.

Long gaps are left as gaps. When a swimmer is half-submerged, their legs are not
missing because the detector had a bad frame; they are missing because they are
underwater, and inventing them would erase the single clearest signal of water
entry the video contains.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from smartdive.keypoints import (
    HEAD_KEYPOINTS,
    HEAD_MASS_FRACTION,
    MIRROR,
    NUM_KEYPOINTS,
    SEGMENTS,
    TOTAL_MASS_FRACTION,
    body_scale,
)
from smartdive.signal import SmoothingConfig, condition
from smartdive.tracking import Track


@dataclass
class PoseSeries:
    """Every keypoint of one athlete, over every frame of a clip.

    ``xy`` is NaN wherever a keypoint has no value. ``measured`` records which
    entries came from the detector rather than from mirroring or interpolation,
    so downstream code can tell a real observation from a reconstruction.
    """

    frames: np.ndarray  # (N,)
    xy: np.ndarray  # (N, 17, 2)
    conf: np.ndarray  # (N, 17)
    measured: np.ndarray  # (N, 17) bool
    mirrored: np.ndarray  # (N, 17) bool
    interpolated: np.ndarray  # (N, 17) bool
    fps: float

    @property
    def n_frames(self) -> int:
        return len(self.frames)

    def present(self) -> np.ndarray:
        """``(N, 17)`` mask of keypoints that have a value at all."""
        return ~np.isnan(self.xy[..., 0])

    def body_scale(self, min_conf: float = 0.0) -> np.ndarray:
        """Per-frame widest keypoint separation, NaN where undeterminable."""
        out = np.full(self.n_frames, np.nan)
        for i in range(self.n_frames):
            conf = np.where(self.present()[i], 1.0, 0.0)
            out[i] = body_scale(np.nan_to_num(self.xy[i]), conf, 0.5)
        return out

    def visible_mass(self, min_conf: float = 0.3) -> np.ndarray:
        """Per-frame fraction of body mass the *detector* actually resolved.

        Deliberately computed from the raw confidences, not from the completed
        skeleton: this is the water-entry cue, and completing the skeleton is
        precisely the operation that would hide it.
        """
        out = np.zeros(self.n_frames)
        for i in range(self.n_frames):
            conf = self.conf[i]
            mass = HEAD_MASS_FRACTION if np.any(conf[list(HEAD_KEYPOINTS)] >= min_conf) else 0.0
            for seg in SEGMENTS:
                if all(conf[k] >= min_conf for k in seg.proximal + seg.distal):
                    mass += seg.mass_fraction
            out[i] = mass / TOTAL_MASS_FRACTION
        return out


def build_pose_series(
    subject: Track,
    n_frames: int,
    fps: float,
    min_conf: float = 0.3,
) -> PoseSeries:
    """Assemble a subject's detections into dense per-keypoint arrays."""
    xy = np.full((n_frames, NUM_KEYPOINTS, 2), np.nan)
    conf = np.zeros((n_frames, NUM_KEYPOINTS))
    for frame, det in subject.observations.items():
        if not 0 <= frame < n_frames:
            continue
        conf[frame] = det.conf
        good = det.conf >= min_conf
        xy[frame, good] = np.asarray(det.xy, dtype=float)[good]
    measured = ~np.isnan(xy[..., 0])
    return PoseSeries(
        frames=np.arange(n_frames),
        xy=xy,
        conf=conf,
        measured=measured,
        mirrored=np.zeros_like(measured),
        interpolated=np.zeros_like(measured),
        fps=fps,
    )


def mirror_missing(series: PoseSeries) -> PoseSeries:
    """Fill a missing keypoint from its contralateral partner, where present.

    Valid because a racing start is filmed from the side: the left and right
    knee project to within a few percent of body length of each other, far
    inside the error this replaces. It is *not* valid for footage shot down the
    lane, and :func:`smartdive.analysis.analyze` warns when the view looks
    frontal enough for that to matter.
    """
    xy = series.xy.copy()
    mirrored = series.mirrored.copy()
    present = ~np.isnan(xy[..., 0])
    for left, right in MIRROR:
        for a, b in ((left, right), (right, left)):
            gap = ~present[:, a] & present[:, b]
            xy[gap, a] = xy[gap, b]
            mirrored[gap, a] = True
    return PoseSeries(
        frames=series.frames,
        xy=xy,
        conf=series.conf,
        measured=series.measured,
        mirrored=mirrored,
        interpolated=series.interpolated,
        fps=series.fps,
    )


def condition_pose(series: PoseSeries, config: SmoothingConfig | None = None) -> PoseSeries:
    """Reject spikes, fill short gaps and smooth, one keypoint at a time."""
    cfg = config or SmoothingConfig()
    xy = series.xy.copy()
    interpolated = series.interpolated.copy()

    for k in range(NUM_KEYPOINTS):
        conditioned = condition(series.frames, series.xy[:, k, :], series.fps, cfg)
        xy[:, k, :] = conditioned.xy
        interpolated[:, k] |= conditioned.interpolated | conditioned.outliers

    return PoseSeries(
        frames=series.frames,
        xy=xy,
        conf=series.conf,
        measured=series.measured,
        mirrored=series.mirrored,
        interpolated=interpolated,
        fps=series.fps,
    )


def prepare(
    subject: Track,
    n_frames: int,
    fps: float,
    min_conf: float = 0.3,
    smoothing: SmoothingConfig | None = None,
    mirror: bool = True,
) -> PoseSeries:
    """Full pipeline: assemble, mirror, condition."""
    series = build_pose_series(subject, n_frames, fps, min_conf)
    if mirror:
        series = mirror_missing(series)
    return condition_pose(series, smoothing)


def centre_of_mass(series: PoseSeries, min_mass: float = 0.75) -> np.ndarray:
    """``(N, 2)`` whole-body centre of mass from a conditioned pose series.

    Because the series has already been completed, the same segments contribute
    in almost every frame, so the estimate no longer jumps when the detector's
    view of a limb changes. Frames where too little of the body survives are
    still NaN, that is water entry, and it should be visible.
    """
    present = series.present()
    out = np.full((series.n_frames, 2), np.nan)

    for i in range(series.n_frames):
        xy = series.xy[i]
        total = np.zeros(2)
        mass = 0.0

        head = [xy[k] for k in HEAD_KEYPOINTS if present[i, k]]
        if head:
            total += HEAD_MASS_FRACTION * np.mean(head, axis=0)
            mass += HEAD_MASS_FRACTION

        for seg in SEGMENTS:
            if not all(present[i, k] for k in seg.proximal + seg.distal):
                continue
            proximal = np.mean([xy[k] for k in seg.proximal], axis=0)
            distal = np.mean([xy[k] for k in seg.distal], axis=0)
            total += seg.mass_fraction * (proximal + seg.com_ratio * (distal - proximal))
            mass += seg.mass_fraction

        if mass >= min_mass * TOTAL_MASS_FRACTION:
            out[i] = total / mass

    return out


def landmark_series(series: PoseSeries, landmark: str, min_mass: float = 0.75) -> np.ndarray:
    """``(N, 2)`` trajectory of any supported landmark from a conditioned series."""
    from smartdive.keypoints import NAME_TO_INDEX, PAIR_LANDMARKS

    if landmark == "com":
        return centre_of_mass(series, min_mass)

    present = series.present()
    if landmark in PAIR_LANDMARKS:
        i, j = PAIR_LANDMARKS[landmark]
        out = (series.xy[:, i, :] + series.xy[:, j, :]) / 2.0
        out[~(present[:, i] & present[:, j])] = np.nan
        return out

    if landmark in NAME_TO_INDEX:
        k = NAME_TO_INDEX[landmark]
        out = series.xy[:, k, :].copy()
        out[~present[:, k]] = np.nan
        return out

    raise ValueError(f"unknown landmark {landmark!r}")


def lateral_view_score(series: PoseSeries, min_conf: float = 0.3) -> float:
    """How side-on the camera is, from 0 (straight down the lane) to 1 (side-on).

    Measured as the median ratio of shoulder separation to trunk length. Filmed
    from the side the shoulders project onto each other and the ratio collapses;
    filmed head-on it approaches the athlete's real biacromial-to-trunk ratio of
    about 0.9. The mirroring in :func:`mirror_missing` and the planar projectile
    model both assume the first case, so this is worth reporting.
    """
    from smartdive.keypoints import LEFT_HIP, LEFT_SHOULDER, RIGHT_HIP, RIGHT_SHOULDER

    present = series.present()
    ratios = []
    for i in range(series.n_frames):
        needed = (LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP)
        if not all(present[i, k] for k in needed) or series.mirrored[i, RIGHT_SHOULDER]:
            continue
        shoulders = np.linalg.norm(series.xy[i, LEFT_SHOULDER] - series.xy[i, RIGHT_SHOULDER])
        trunk = np.linalg.norm(
            (series.xy[i, LEFT_SHOULDER] + series.xy[i, RIGHT_SHOULDER]) / 2
            - (series.xy[i, LEFT_HIP] + series.xy[i, RIGHT_HIP]) / 2
        )
        if trunk > 1:
            ratios.append(float(shoulders / trunk))
    if not ratios:
        return float("nan")
    # 0.9 is roughly biacromial width over trunk length for an adult; anything
    # at or above that is a frontal view.
    return float(np.clip(1.0 - np.median(ratios) / 0.9, 0.0, 1.0))
