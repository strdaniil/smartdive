"""Estimating the athlete's size in pixels from their own skeleton.

Every absolute number this project reports, speed in m/s, distance in metres,
needs a pixels-per-metre scale, and on ordinary poolside footage there is nothing
in the frame whose real size is known. The athlete is therefore the ruler, and
this module is about measuring the ruler well.

The previous approach was to take the widest separation between any two
confident keypoints and call it 1.15 times the athlete's stature. That quantity
is *pose dependent*: it is roughly stature for someone standing, a third of that
for someone crouched on a block, and about 1.2 times stature for someone
streamlined in flight. Since the pose changes continuously through the clip, so
does the estimate, and the correction factor is a single constant that can only
be right for one of those poses.

What is measured here instead is the length of individual body *segments*,
trunk, thigh, shank, upper arm, forearm, each of which is a fixed fraction of
stature that does not depend on what the athlete is doing. Two properties make
this work:

**Projection only ever shortens.** A segment of true pixel length ``L`` seen at
an angle ``theta`` out of the image plane projects to ``L * cos(theta)``, which
is at most ``L``. So the *upper* end of the distribution of a segment's observed
length over a clip estimates its true length, and every foreshortened frame is
simply discarded by taking a high percentile rather than a mean. A high
percentile rather than the maximum, because detector noise has no such one-sided
bound.

**Several segments disagree independently.** Each segment gives its own stature
estimate. Combining them with a weighted median, weighted towards the longer
segments, whose relative measurement error is smaller, means one badly tracked
limb cannot move the answer.

The same machinery gives a second thing for free: a *per-frame* size, which is
how the pipeline sees an athlete travelling away from an obliquely placed camera.
See :func:`relative_size_series`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from smartdive.keypoints import (
    LEFT_ANKLE,
    LEFT_ELBOW,
    LEFT_HIP,
    LEFT_KNEE,
    LEFT_SHOULDER,
    LEFT_WRIST,
    RIGHT_ANKLE,
    RIGHT_ELBOW,
    RIGHT_HIP,
    RIGHT_KNEE,
    RIGHT_SHOULDER,
    RIGHT_WRIST,
)


@dataclass(frozen=True)
class SegmentRatio:
    """A measurable bone-to-bone distance and its length as a fraction of stature.

    ``a`` and ``b`` are tuples of keypoint indices; each end is the mean of its
    tuple, so ``(LEFT_HIP, RIGHT_HIP)`` means "the mid-hip".
    """

    name: str
    a: tuple[int, ...]
    b: tuple[int, ...]
    stature_fraction: float
    #: Relative confidence in this segment as a stature estimator. Long segments
    #: score better because a fixed keypoint error is a smaller fraction of
    #: them; the trunk scores best because it is rigid and always visible.
    weight: float


# Fractions of stature from Winter, *Biomechanics and Motor Control of Human
# Movement* (4th ed., Fig. 4.1, after Drillis & Contini), with one adjustment.
#
# Winter's shoulder height (0.818) minus hip height (0.530) gives a trunk of
# 0.288 stature, but those are *surface* heights, and COCO's shoulder keypoint
# is annotated at the glenohumeral joint, inside and below the top of the
# shoulder, while its hip keypoint sits at the joint centre. Measured on the
# sample clips the resulting shoulder-to-hip distance sits between 0.85 and 1.15
# of what 0.288 predicts depending on how side-on the athlete is, because a
# trunk pointing at the camera, which is what a track start on the blocks
# looks like, foreshortens more than a limb does. So the trunk keeps a
# slightly reduced ratio and, more importantly, a low weight: the legs are the
# reliable ruler here, being long, near-planar in flight, and bounded at both
# ends by joints the detector localises well.
SEGMENT_RATIOS: tuple[SegmentRatio, ...] = (
    SegmentRatio("trunk", (LEFT_SHOULDER, RIGHT_SHOULDER), (LEFT_HIP, RIGHT_HIP), 0.270, 0.5),
    SegmentRatio("left_thigh", (LEFT_HIP,), (LEFT_KNEE,), 0.245, 0.8),
    SegmentRatio("right_thigh", (RIGHT_HIP,), (RIGHT_KNEE,), 0.245, 0.8),
    SegmentRatio("left_shank", (LEFT_KNEE,), (LEFT_ANKLE,), 0.246, 0.8),
    SegmentRatio("right_shank", (RIGHT_KNEE,), (RIGHT_ANKLE,), 0.246, 0.8),
    SegmentRatio("left_upper_arm", (LEFT_SHOULDER,), (LEFT_ELBOW,), 0.186, 0.35),
    SegmentRatio("right_upper_arm", (RIGHT_SHOULDER,), (RIGHT_ELBOW,), 0.186, 0.35),
    SegmentRatio("left_forearm", (LEFT_ELBOW,), (LEFT_WRIST,), 0.146, 0.25),
    SegmentRatio("right_forearm", (RIGHT_ELBOW,), (RIGHT_WRIST,), 0.146, 0.25),
    # Composite spans. Longer than any single bone and therefore less sensitive
    # to keypoint noise, at the cost of being exactly straight only when the
    # joint between them is extended, which, for a diver in flight, it is.
    SegmentRatio("left_leg", (LEFT_HIP,), (LEFT_ANKLE,), 0.491, 1.0),
    SegmentRatio("right_leg", (RIGHT_HIP,), (RIGHT_ANKLE,), 0.491, 1.0),
)

#: Stature assumed when the user supplies none. Roughly the mean height of adult
#: competitive swimmers; every report states when this was used.
DEFAULT_STATURE_M = 1.78

#: Range of athlete heights the pipeline will accept, in metres. Wide enough for
#: a small age-grouper and a very tall senior, narrow enough to catch the one
#: mistake that would otherwise pass silently, a height given in centimetres.
MIN_STATURE_M = 1.20
MAX_STATURE_M = 2.30


def check_stature(stature_m: float) -> float:
    """Validate a user-supplied athlete height, in metres.

    Every distance and speed the pipeline reports is inversely proportional to
    this number, so a units slip here does not produce a slightly wrong answer,
    it produces one wrong by a factor of a hundred, and nothing downstream can
    tell, because a scale is exactly the thing a single camera cannot check. The
    error message names the likely mistake rather than just the bound.
    """
    value = float(stature_m)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"athlete height must be a positive number of metres, not {stature_m!r}")
    if MIN_STATURE_M <= value <= MAX_STATURE_M:
        return value
    if MIN_STATURE_M * 100 <= value <= MAX_STATURE_M * 100:
        raise ValueError(
            f"athlete height {value:g} looks like centimetres; it is expected in "
            f"metres, so pass {value / 100:.2f}"
        )
    if MIN_STATURE_M / 100 <= value <= MAX_STATURE_M / 100:
        raise ValueError(
            f"athlete height {value:g} looks like hundredths of a metre; it is "
            f"expected in metres, so pass {value * 100:.2f}"
        )
    raise ValueError(
        f"athlete height {value:g} m is outside the plausible "
        f"{MIN_STATURE_M:.2f}-{MAX_STATURE_M:.2f} m range"
    )


@dataclass
class BodySize:
    """The athlete's measured size in pixels, and how well it is pinned down."""

    #: Estimated pixel length of the athlete's stature.
    stature_px: float
    #: Per-segment stature estimates, in pixels, keyed by segment name.
    per_segment: dict[str, float]
    #: Spread of those estimates as a fraction of the median. A tight spread
    #: means the segments agree; a wide one means the pose or the detector was
    #: not cooperating and the scale should be treated with more caution.
    dispersion: float
    #: Frames that contributed at least one segment measurement.
    n_frames: int

    def to_dict(self) -> dict[str, object]:
        return {
            "stature_px": round(self.stature_px, 1),
            "dispersion": round(self.dispersion, 3),
            "n_frames": self.n_frames,
            "per_segment_px": {k: round(v, 1) for k, v in sorted(self.per_segment.items())},
        }


def _segment_lengths(
    xy: np.ndarray, present: np.ndarray, ratio: SegmentRatio
) -> tuple[np.ndarray, np.ndarray]:
    """``(frames, lengths)`` for one segment, over the frames where it is visible."""
    needed = list(dict.fromkeys(ratio.a + ratio.b))
    ok = present[:, needed].all(axis=1)
    if not ok.any():
        return np.array([], dtype=int), np.array([])
    a = xy[np.ix_(np.where(ok)[0], list(ratio.a))].mean(axis=1)
    b = xy[np.ix_(np.where(ok)[0], list(ratio.b))].mean(axis=1)
    return np.where(ok)[0], np.linalg.norm(a - b, axis=1)


def measure(
    xy: np.ndarray,
    present: np.ndarray,
    frames: np.ndarray | None = None,
    percentile: float = 85.0,
    min_observations: int = 4,
    trim: float = 0.35,
) -> BodySize:
    """Estimate the athlete's stature in pixels from a conditioned pose series.

    Parameters
    ----------
    xy:
        ``(N, 17, 2)`` keypoint positions.
    present:
        ``(N, 17)`` mask of which entries have a value.
    frames:
        Optional subset of frame indices to measure over. Restricting this to
        the flight window is what makes the scale correct for an athlete who is
        closer to (or further from) the camera than they were on the block.
    percentile:
        Where in each segment's length distribution to read off its true length.
        High, because projection foreshortens and never lengthens.
    trim:
        Segments whose implied stature differs from the consensus by more than
        this fraction are dropped and the consensus recomputed. A mislocated
        wrist can double the apparent length of a forearm, and a forearm is
        short enough that doubling it is not obviously absurd until it is
        compared with the thigh.
    """
    xy = np.asarray(xy, dtype=float)
    present = np.asarray(present, dtype=bool)
    if frames is not None:
        frames = np.asarray(frames, dtype=int)
        frames = frames[(frames >= 0) & (frames < len(xy))]
        xy, present = xy[frames], present[frames]

    estimates: dict[str, float] = {}
    weights: dict[str, float] = {}
    contributing: set[int] = set()

    for ratio in SEGMENT_RATIOS:
        indices, lengths = _segment_lengths(xy, present, ratio)
        lengths = lengths[np.isfinite(lengths) & (lengths > 0)]
        if len(lengths) < min_observations:
            continue
        contributing.update(indices.tolist())
        estimates[ratio.name] = float(np.percentile(lengths, percentile)) / ratio.stature_fraction
        weights[ratio.name] = ratio.weight

    if not estimates:
        return BodySize(float("nan"), {}, float("nan"), 0)

    keys = list(estimates)
    values = np.array([estimates[k] for k in keys])
    w = np.array([weights[k] for k in keys])

    stature = _weighted_median(values, w)
    for _ in range(3):
        keep = np.abs(values - stature) <= trim * stature
        if keep.sum() < 2 or keep.all():
            break
        keys = [k for k, ok in zip(keys, keep, strict=True) if ok]
        values, w = values[keep], w[keep]
        refined = _weighted_median(values, w)
        if abs(refined - stature) < 1e-6:
            break
        stature = refined

    dispersion = (
        float(np.median(np.abs(values - stature)) / stature) if stature > 0 else float("nan")
    )

    return BodySize(
        stature_px=float(stature),
        per_segment={k: estimates[k] for k in keys},
        dispersion=dispersion,
        n_frames=len(contributing),
    )


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    cumulative = np.cumsum(weights)
    if cumulative[-1] <= 0:
        return float(np.median(values))
    return float(values[np.searchsorted(cumulative, cumulative[-1] / 2.0)])


def relative_size_series(
    xy: np.ndarray,
    present: np.ndarray,
    reference: BodySize,
    smooth: int = 5,
) -> np.ndarray:
    """Per-frame apparent size, as a multiple of ``reference.stature_px``.

    Each segment visible in a frame votes with its own length divided by the
    length that segment would have at the reference scale, and the frame's size
    is the median of those votes. Normalising every segment by its *own*
    reference removes the pose dependence that made raw body extent useless:
    when the athlete pikes, every segment foreshortens somewhat, but the median
    vote moves far less than any single one.

    This is what a camera placed obliquely to the lane needs. An athlete diving
    away from the camera shrinks, and a single pixels-per-metre for the whole
    clip is then wrong at both ends of the flight. NaN where nothing was
    measurable.
    """
    xy = np.asarray(xy, dtype=float)
    present = np.asarray(present, dtype=bool)
    n = len(xy)
    votes: list[np.ndarray] = []

    for ratio in SEGMENT_RATIOS:
        expected = reference.per_segment.get(ratio.name)
        if not expected or not np.isfinite(expected):
            continue
        column = np.full(n, np.nan)
        indices, lengths = _segment_lengths(xy, present, ratio)
        if len(indices):
            column[indices] = lengths / (expected * ratio.stature_fraction)
        votes.append(column)

    if not votes:
        return np.full(n, np.nan)

    stacked = np.vstack(votes)
    size = np.full(len(xy), np.nan)
    measured = ~np.all(np.isnan(stacked), axis=0)
    if measured.any():
        size[measured] = np.nanmedian(stacked[:, measured], axis=0)

    if smooth and smooth > 1:
        size = _nan_moving_median(size, smooth)
    return size


def _nan_moving_median(values: np.ndarray, window: int) -> np.ndarray:
    half = max(window // 2, 1)
    out = np.full(len(values), np.nan)
    for i in range(len(values)):
        chunk = values[max(0, i - half) : i + half + 1]
        chunk = chunk[~np.isnan(chunk)]
        if len(chunk):
            out[i] = np.median(chunk)
    return out
