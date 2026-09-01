"""Trajectory conditioning: outlier rejection, gap filling, smoothing.

A raw landmark trajectory straight out of a pose detector is not usable for
kinematics. It contains dropouts (NaN wherever the joint was occluded), spikes
(a left/right limb swap costs a body width in one frame), and per-frame jitter
that a naive finite difference amplifies into meaningless velocities.

The pipeline is deliberately ordered: reject, then fill, then smooth. Smoothing
first would smear a spike across its neighbours instead of removing it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.signal import savgol_filter


@dataclass
class SmoothingConfig:
    """Conditioning hyper-parameters, all in subject-relative units."""

    #: Half-width of the Hampel window, in frames.
    hampel_window: int = 5
    #: Spike threshold as a multiple of the local robust deviation.
    hampel_sigma: float = 3.0
    #: Longest run of missing frames that may be interpolated across. Longer
    #: gaps stay NaN rather than inventing a trajectory nobody measured.
    max_gap: int = 8
    #: Savitzky-Golay window length in frames (forced odd).
    savgol_window: int = 7
    #: Savitzky-Golay polynomial order. Order 2 is not arbitrary: it reproduces
    #: constant-acceleration motion *exactly*, so smoothing a projectile
    #: trajectory cannot bias the gravity estimate the physics stage depends on.
    savgol_polyorder: int = 2


@dataclass
class ConditionedTrajectory:
    """A conditioned trajectory plus the provenance of every sample."""

    frames: np.ndarray  # (N,) frame indices, contiguous
    raw: np.ndarray  # (N, 2) as measured, NaN where missing
    xy: np.ndarray  # (N, 2) conditioned, NaN only in unfilled gaps
    valid: np.ndarray  # (N,) bool, True where a real measurement exists
    outliers: np.ndarray  # (N,) bool, True where a spike was rejected
    interpolated: np.ndarray  # (N,) bool, True where the value was filled in
    fps: float
    noise_sigma: float = float("nan")  # px, estimated measurement noise

    @property
    def coverage(self) -> float:
        """Fraction of frames carrying an original measurement."""
        return float(self.valid.mean())

    @property
    def usable(self) -> np.ndarray:
        """Frames where the conditioned trajectory is defined."""
        return ~np.isnan(self.xy[:, 0])

    def velocity(self) -> np.ndarray:
        """Per-frame velocity in px/s via a Savitzky-Golay derivative."""
        return _derivative(self.xy, self.fps, order=1)

    def acceleration(self) -> np.ndarray:
        """Per-frame acceleration in px/s^2."""
        return _derivative(self.xy, self.fps, order=2)


def hampel_mask(values: np.ndarray, window: int = 5, sigma: float = 3.0) -> np.ndarray:
    """Boolean mask of spikes, via a rolling median absolute deviation.

    The Hampel filter compares each sample with the median of its neighbours,
    scaled by the median absolute deviation of that same window. Unlike a
    mean/standard-deviation test it is not dragged around by the very outlier it
    is trying to find.
    """
    values = np.asarray(values, dtype=float)
    n = len(values)
    out = np.zeros(n, dtype=bool)
    if n < 2 * window + 1:
        return out

    # 1.4826 rescales the MAD to be a consistent estimator of the standard
    # deviation for normally distributed data.
    scale = 1.4826
    for i in range(n):
        lo = max(0, i - window)
        hi = min(n, i + window + 1)
        neighbourhood = values[lo:hi]
        neighbourhood = neighbourhood[~np.isnan(neighbourhood)]
        if len(neighbourhood) < 3 or np.isnan(values[i]):
            continue
        median = np.median(neighbourhood)
        mad = scale * np.median(np.abs(neighbourhood - median))
        if mad <= 0:
            continue
        if abs(values[i] - median) > sigma * mad:
            out[i] = True
    return out


def fill_gaps(t: np.ndarray, values: np.ndarray, max_gap: int) -> tuple[np.ndarray, np.ndarray]:
    """Monotone cubic interpolation across gaps no longer than ``max_gap``.

    PCHIP is used rather than a plain cubic spline because it does not overshoot:
    a spline through a dropout either side of the trajectory apex can invent a
    bump that never happened.

    Returns the filled series and a mask marking which samples were invented.
    """
    values = np.asarray(values, dtype=float)
    known = ~np.isnan(values)
    filled = values.copy()
    invented = np.zeros(len(values), dtype=bool)

    if known.sum() < 2:
        return filled, invented

    interpolator = PchipInterpolator(t[known], values[known], extrapolate=False)

    # Walk each run of missing samples and only fill the short ones.
    i = 0
    n = len(values)
    while i < n:
        if known[i]:
            i += 1
            continue
        j = i
        while j < n and not known[j]:
            j += 1
        gap = j - i
        bounded = i > 0 and j < n  # interior gap, so interpolation is defined
        if bounded and gap <= max_gap:
            filled[i:j] = interpolator(t[i:j])
            invented[i:j] = True
        i = j

    return filled, invented


def _odd_window(window: int, n: int, polyorder: int) -> int:
    """Largest usable odd Savitzky-Golay window for a series of length ``n``."""
    window = min(window, n if n % 2 else n - 1)
    if window % 2 == 0:
        window -= 1
    return max(window, polyorder + 1 + (polyorder % 2 == 0))


def _derivative(xy: np.ndarray, fps: float, order: int) -> np.ndarray:
    """Savitzky-Golay derivative, NaN-safe, in px/s^order."""
    out = np.full_like(xy, np.nan, dtype=float)
    good = ~np.isnan(xy[:, 0])
    if good.sum() < 5:
        return out
    idx = np.where(good)[0]
    # Only differentiate across contiguous runs so a gap does not create a
    # phantom acceleration spike at its edges.
    for start, stop in _contiguous_runs(idx):
        seg = xy[start:stop]
        n = len(seg)
        polyorder = max(order + 1, 2)
        if n < polyorder + 2:
            continue
        window = _odd_window(max(polyorder + 2, 7), n, polyorder)
        for axis in (0, 1):
            out[start:stop, axis] = savgol_filter(
                seg[:, axis], window, polyorder, deriv=order, delta=1.0 / fps
            )
    return out


def _contiguous_runs(idx: np.ndarray) -> list[tuple[int, int]]:
    """Split sorted indices into ``(start, stop)`` half-open contiguous runs."""
    if len(idx) == 0:
        return []
    runs: list[tuple[int, int]] = []
    start = prev = idx[0]
    for i in idx[1:]:
        if i != prev + 1:
            runs.append((int(start), int(prev) + 1))
            start = i
        prev = i
    runs.append((int(start), int(prev) + 1))
    return runs


def condition(
    frames: np.ndarray,
    xy: np.ndarray,
    fps: float,
    config: SmoothingConfig | None = None,
) -> ConditionedTrajectory:
    """Run the full reject -> fill -> smooth pipeline on a raw trajectory."""
    cfg = config or SmoothingConfig()
    frames = np.asarray(frames)
    raw = np.asarray(xy, dtype=float).copy()
    work = raw.copy()

    valid = ~np.isnan(work[:, 0])

    # 1. Reject spikes, judging each axis independently: a limb swap usually
    #    displaces one axis far more than the other.
    outliers = np.zeros(len(work), dtype=bool)
    for axis in (0, 1):
        outliers |= hampel_mask(work[:, axis], cfg.hampel_window, cfg.hampel_sigma)
    work[outliers] = np.nan

    # 2. Fill short gaps (both genuine dropouts and the spikes just removed).
    t = frames.astype(float)
    interpolated = np.zeros(len(work), dtype=bool)
    for axis in (0, 1):
        work[:, axis], invented = fill_gaps(t, work[:, axis], cfg.max_gap)
        interpolated |= invented

    # 3. Smooth. Order-2 Savitzky-Golay is exact for ballistic motion.
    good = ~np.isnan(work[:, 0])
    if good.sum() >= cfg.savgol_polyorder + 2:
        for start, stop in _contiguous_runs(np.where(good)[0]):
            n = stop - start
            if n < cfg.savgol_polyorder + 2:
                continue
            window = _odd_window(cfg.savgol_window, n, cfg.savgol_polyorder)
            for axis in (0, 1):
                work[start:stop, axis] = savgol_filter(
                    work[start:stop, axis], window, cfg.savgol_polyorder
                )

    return ConditionedTrajectory(
        frames=frames,
        raw=raw,
        xy=work,
        valid=valid,
        outliers=outliers,
        interpolated=interpolated,
        fps=fps,
        # Estimated from the raw signal: conditioning correlates neighbouring
        # samples, which would make the smoothed series look quieter than the
        # measurement actually was.
        noise_sigma=noise_scale(raw),
    )


def noise_scale(xy: np.ndarray) -> float:
    """Estimate per-sample measurement noise, in pixels, from second differences.

    For a smooth signal sampled with independent noise of standard deviation
    ``sigma``, the second difference ``y[i-1] - 2*y[i] + y[i+1]`` has variance
    ``6 * sigma**2``, and the underlying smooth motion contributes almost
    nothing because a second difference annihilates anything locally linear.
    Dividing the observed second-difference RMS by ``sqrt(6)`` therefore
    recovers the noise without needing to know the trajectory.

    This matters because the obvious alternative, taking the residual of the
    best-fitting window as the noise estimate, is biased low precisely
    *because* that window was chosen for having the smallest residual.
    """
    j = jitter(xy)
    return float(j / np.sqrt(6.0)) if np.isfinite(j) else float("nan")


def jitter(xy: np.ndarray) -> float:
    """RMS of the second difference, a scale-free measure of path roughness.

    A physically real trajectory is smooth, so after removing genuine motion
    (which a second difference does for anything moving at constant velocity)
    what is left is measurement noise. Lower is better.
    """
    good = ~np.isnan(xy[:, 0])
    if good.sum() < 4:
        return float("nan")
    runs = _contiguous_runs(np.where(good)[0])
    terms = [np.diff(xy[a:b], n=2, axis=0) for a, b in runs if b - a >= 3]
    terms = [t for t in terms if len(t)]
    if not terms:
        return float("nan")
    stacked = np.vstack(terms)
    return float(np.sqrt(np.mean(np.sum(stacked**2, axis=1))))
