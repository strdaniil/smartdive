"""Ballistic flight modelling and automatic phase detection.

The old pipeline fit a degree-3 polynomial to ``x(t)`` and ``y(t)`` over the
whole clip. That has two problems. A cubic has no physical meaning for a body in
free flight, so its coefficients cannot be turned into a velocity, an angle, or a
scale; and fitting over the whole clip mixes three regimes -- feet on the block,
free flight, and deceleration in the water -- into one curve that describes none
of them.

A body in free flight obeys, exactly::

    x(t) = x0 + vx * t
    y(t) = y0 + vy * t + 0.5 * g * t^2      (y grows downward in image space)

That model has real consequences. Its coefficients *are* the takeoff velocity.
Its quality tells you which frames were actually airborne. And because the
quadratic coefficient must equal gravity, the fit measures the pixels-per-metre
scale of the scene for free -- see :mod:`smartdive.calibration`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Standard gravity, m/s^2.
G = 9.80665


@dataclass
class BallisticFit:
    """A constant-acceleration fit to one contiguous stretch of trajectory."""

    start_frame: int
    end_frame: int
    fps: float
    x0: float  # px
    y0: float  # px
    vx: float  # px/s (positive = rightward in image space)
    vy: float  # px/s (positive = downward in image space)
    ay: float  # px/s^2, downward positive; equals g * px_per_metre
    n_points: int
    residual_rms: float  # px
    residual_max: float  # px
    r_squared: float

    @property
    def duration(self) -> float:
        return (self.end_frame - self.start_frame) / self.fps

    @property
    def speed(self) -> float:
        """Takeoff speed in px/s."""
        return float(np.hypot(self.vx, self.vy))

    @property
    def angle_deg(self) -> float:
        """Takeoff angle above horizontal, in degrees.

        Positive is upward, regardless of which way the athlete faces.
        """
        return float(np.degrees(np.arctan2(-self.vy, abs(self.vx))))

    def position(self, t: np.ndarray) -> np.ndarray:
        """Model position, ``t`` in seconds from ``start_frame``."""
        t = np.asarray(t, dtype=float)
        return np.stack(
            [self.x0 + self.vx * t, self.y0 + self.vy * t + 0.5 * self.ay * t**2],
            axis=-1,
        )

    def sample(self, n: int = 200) -> np.ndarray:
        """``n`` points along the modelled flight path, in pixels."""
        return self.position(np.linspace(0.0, self.duration, n))

    def apex_time(self) -> float:
        """Time from takeoff to the highest point, in seconds (0 if never rising)."""
        if self.ay <= 0:
            return 0.0
        return float(max(0.0, -self.vy / self.ay))

    def to_dict(self) -> dict[str, object]:
        return {
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "duration_s": round(self.duration, 3),
            "n_points": self.n_points,
            "vx_px_s": round(self.vx, 1),
            "vy_px_s": round(self.vy, 1),
            "ay_px_s2": round(self.ay, 1),
            "takeoff_angle_deg": round(self.angle_deg, 2),
            "residual_rms_px": round(self.residual_rms, 2),
            "residual_max_px": round(self.residual_max, 2),
            "r_squared": round(self.r_squared, 4),
        }


def fit_ballistic(
    frames: np.ndarray, xy: np.ndarray, fps: float, ay: float | None = None
) -> BallisticFit | None:
    """Least-squares fit of the projectile model to one stretch of trajectory.

    If ``ay`` is given the vertical acceleration is held fixed and only the four
    velocity/offset terms are estimated. That constrained form is much better
    conditioned: over a 0.3 s flight sampled at 30 fps, the curvature a free fit
    has to isolate is comparable in size to the keypoint noise, so the
    unconstrained acceleration estimate swings by a factor of two or more
    depending on exactly where the window ends. Fixing ``ay`` from a scale prior
    removes that degeneracy and, as a side effect, turns the residual into a
    sharp test of whether a frame really was in free flight.
    """
    frames = np.asarray(frames)
    xy = np.asarray(xy, dtype=float)
    good = ~np.isnan(xy[:, 0])
    if good.sum() < (3 if ay is not None else 4):
        return None

    f = frames[good]
    x = xy[good, 0]
    y = xy[good, 1]
    t = (f - f[0]) / fps

    cx = np.polyfit(t, x, 1)
    if ay is None:
        cy = np.polyfit(t, y, 2)
    else:
        # Remove the known gravitational term, then fit what is left as a line.
        linear = np.polyfit(t, y - 0.5 * ay * t**2, 1)
        cy = np.array([0.5 * ay, linear[0], linear[1]])

    model = np.stack([np.polyval(cx, t), np.polyval(cy, t)], axis=-1)
    residual = np.linalg.norm(model - np.stack([x, y], axis=-1), axis=1)

    observed = np.stack([x, y], axis=-1)
    ss_res = float(np.sum((model - observed) ** 2))
    ss_tot = float(np.sum((observed - observed.mean(axis=0)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return BallisticFit(
        start_frame=int(f[0]),
        end_frame=int(f[-1]),
        fps=fps,
        x0=float(cx[1]),
        y0=float(cy[2]),
        vx=float(cx[0]),
        vy=float(cy[1]),
        ay=float(2 * cy[0]),
        n_points=int(good.sum()),
        residual_rms=float(np.sqrt(np.mean(residual**2))),
        residual_max=float(residual.max()),
        r_squared=float(r2),
    )


@dataclass
class FlightSearchConfig:
    """Constraints on what counts as a plausible flight phase."""

    #: Shortest acceptable seed window, in frames.
    seed_frames: int = 6
    #: Fit residual budget for a seed, as a fraction of the distance travelled
    #: within it. Normalising by travel rather than by an absolute pixel count
    #: is what stops a near-motionless crouch on the block from winning purely
    #: by having nothing to be wrong about.
    max_seed_relative_residual: float = 0.08
    #: A seed must cover at least this many body lengths of travel.
    min_seed_travel_bodies: float = 0.5
    #: The full window must cover at least this many body lengths.
    min_travel_bodies: float = 0.8
    #: A frame is an inlier if its residual is within this multiple of the
    #: estimated measurement noise...
    inlier_sigma: float = 1.5
    #: ...or within this fraction of a body length, whichever is larger. The
    #: floor stops an unusually clean seed from refusing to grow at all.
    inlier_floor_bodies: float = 0.04
    #: How many consecutive outliers the window may bridge. One badly localised
    #: joint should not truncate a flight that plainly continues past it.
    max_outlier_run: int = 2
    #: Consensus refinement passes.
    refine_iterations: int = 4
    #: Frames whose residual exceeds this multiple of the inlier tolerance are
    #: dropped from the fit entirely, rather than merely bounding the window.
    outlier_tolerance_factor: float = 2.0
    #: Implied athlete extent must stay human, given the fitted gravity. Body
    #: scale is the widest keypoint separation, so it runs from roughly stature
    #: (standing) to about 1.3x stature (streamlined), which for adult athletes
    #: bounds it at 1.4-2.6 m.
    min_implied_stature_m: float = 1.4
    max_implied_stature_m: float = 2.6


@dataclass
class FlightPhase:
    """The detected airborne window, and the fit that identified it."""

    fit: BallisticFit
    body_scale_px: float
    implied_stature_m: float
    searched_windows: int
    n_inliers: int = 0
    n_frames: int = 0

    @property
    def takeoff_frame(self) -> int:
        return self.fit.start_frame

    @property
    def entry_frame(self) -> int:
        return self.fit.end_frame

    def to_dict(self) -> dict[str, object]:
        return {
            "takeoff_frame": self.takeoff_frame,
            "entry_frame": self.entry_frame,
            "body_scale_px": round(self.body_scale_px, 1),
            "implied_stature_m": round(self.implied_stature_m, 2),
            "searched_windows": self.searched_windows,
            "n_inliers": self.n_inliers,
            "n_frames": self.n_frames,
            "fit": self.fit.to_dict(),
        }


def _travel(window: np.ndarray) -> float:
    """Path length through a stretch of trajectory, in pixels."""
    if len(window) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(window, axis=0), axis=1)))


def _residual_at(fit: BallisticFit, frame: int, point: np.ndarray, fps: float) -> float:
    """Distance from a single sample to the model, extrapolating if needed."""
    t = (frame - fit.start_frame) / fps
    predicted = fit.position(np.array([t]))[0]
    return float(np.linalg.norm(predicted - point))


def detect_flight(
    frames: np.ndarray,
    xy: np.ndarray,
    fps: float,
    body_scale_px: float,
    config: FlightSearchConfig | None = None,
    airborne: np.ndarray | None = None,
    noise_sigma: float | None = None,
    ay_prior: float | None = None,
) -> FlightPhase | None:
    """Find the airborne window by seeding a projectile model and growing it.

    Taking the longest window whose residual stays under a fixed budget does not
    work: the budget has to be loose enough to tolerate keypoint noise during
    flight, and once it is that loose the window happily swallows the last few
    frames on the block and the first few in the water, both of which bias the
    fitted gravity and therefore every metric derived from it.

    Instead this is change-point detection. A short seed window is chosen for
    fitting the projectile model best *relative to how far the athlete moved
    inside it*, then the window grows one frame at a time in each direction for
    as long as the new frame is an inlier to the model fit so far. A foot still
    on the block is not an inlier, so growth stops where the physics starts.

    The far end needs help, and ``airborne`` provides it. A streamlined swimmer
    in the first tenth of a second underwater is still travelling on very nearly
    a parabola, so the residual alone cannot say where the water begins -- but
    the pose detector can only see a whole body while that body is out of the
    water. Passing a mask of frames in which most of the athlete's mass is
    visible turns entry into a direct observation instead of an inference.
    """
    cfg = config or FlightSearchConfig()
    frames = np.asarray(frames)
    xy = np.asarray(xy, dtype=float)
    n = len(frames)

    if not np.isfinite(body_scale_px) or body_scale_px <= 0:
        return None

    airborne_mask = None if airborne is None else np.asarray(airborne, dtype=bool)
    usable = ~np.isnan(xy[:, 0])
    if airborne_mask is not None:
        usable = usable & airborne_mask

    seed_travel_budget = cfg.min_seed_travel_bodies * body_scale_px
    searched = 0
    best_seed: tuple[float, int, int] | None = None  # (relative residual, i, j)

    for i in range(n):
        if not usable[i]:
            continue
        for j in range(i + cfg.seed_frames, n + 1):
            if not usable[j - 1]:
                break  # an unusable frame ends every window that would start here
            window = xy[i:j]
            searched += 1

            travel = _travel(window)
            if travel < seed_travel_budget:
                continue

            fit = fit_ballistic(frames[i:j], window, fps, ay_prior)
            if fit is None or fit.ay <= 0:
                continue

            relative = fit.residual_rms / travel
            if relative > cfg.max_seed_relative_residual:
                continue

            if ay_prior is None:
                stature = body_scale_px / (fit.ay / G)
                if not (cfg.min_implied_stature_m <= stature <= cfg.max_implied_stature_m):
                    continue

            if best_seed is None or relative < best_seed[0]:
                best_seed = (relative, i, j)

    if best_seed is None:
        return None

    _, seed_start, seed_stop = best_seed
    fit = fit_ballistic(frames[seed_start:seed_stop], xy[seed_start:seed_stop], fps, ay_prior)
    if fit is None:  # pragma: no cover - the seed was fit successfully above
        return None

    # The noise scale is frozen at the seed. Re-deriving it from the growing
    # window would be a positive feedback loop: each marginal frame raises the
    # residual, which raises the tolerance, which admits the next marginal
    # frame, until the window has eaten the whole clip.
    # Two error sources have to fit inside the tolerance and either can dominate.
    # ``noise_sigma`` measures per-frame detector jitter; the seed's own residual
    # measures how well the projectile model describes the landmark at all,
    # which for a joint rather than the true centre of mass also contains the
    # body's rotation about it. Taking the larger keeps the tolerance honest on
    # clips where either one is the limiting factor.
    sigma = fit.residual_rms
    if noise_sigma is not None and np.isfinite(noise_sigma):
        sigma = max(sigma, noise_sigma)
    tolerance = max(
        cfg.inlier_sigma * sigma,
        cfg.inlier_floor_bodies * body_scale_px,
    )

    # Expand the seed into a consensus set, RANSAC style. Every usable frame is
    # scored against the current model; the window is the contiguous stretch of
    # inliers containing the seed, and the model is then refit on those inliers
    # alone. A single badly localised joint is bridged rather than treated as
    # the end of the flight, but a sustained departure from the model -- a foot
    # still on the block, a body decelerating in water -- stops the expansion.
    #
    # The plausibility gate is deliberately not applied per iteration: a short
    # seed measures gravity imprecisely, so intermediate windows swing through
    # implausible scales on the way to the right answer. It is checked once, at
    # the end, where the estimate is worth trusting.
    start, stop = seed_start, seed_stop
    inliers = np.zeros(n, dtype=bool)

    # Entry is fixed by ``airborne`` rather than by the fit. A streamlined
    # swimmer in the first tenth of a second underwater is still travelling on
    # very nearly a parabola, so the residual cannot see the water -- but the
    # visibility collapse can, and does, within a frame or two.
    right = seed_stop - 1
    for k in range(seed_stop, n):
        if not (airborne_mask[k] if airborne_mask is not None else True):
            break
        if usable[k]:
            right = k

    # Takeoff is fixed by the fit. While a foot is still on the block the
    # athlete is being accelerated by their legs, which a constant-acceleration
    # model cannot express, so the residual rises steeply and unambiguously --
    # typically by an order of magnitude within three frames of the break.
    left = seed_start
    for _ in range(cfg.refine_iterations):
        residuals = np.full(n, np.inf)
        for k in range(n):
            if usable[k]:
                residuals[k] = _residual_at(fit, frames[k], xy[k], fps)
        is_inlier = residuals <= tolerance

        new_left, run = left, 0
        for k in range(seed_start - 1, -1, -1):
            if is_inlier[k]:
                new_left, run = k, 0
            else:
                run += 1
                if run > cfg.max_outlier_run:
                    break

        window = np.zeros(n, dtype=bool)
        window[new_left : right + 1] = usable[new_left : right + 1]
        # Drop only gross outliers from the fit; ordinary noise stays in so the
        # estimate uses every frame that was actually measured.
        window &= residuals <= cfg.outlier_tolerance_factor * tolerance
        window[seed_start:seed_stop] = usable[seed_start:seed_stop]
        if window.sum() < 4:
            break

        refit = fit_ballistic(frames, np.where(window[:, None], xy, np.nan), fps, ay_prior)
        if refit is None or refit.ay <= 0:
            break

        converged = new_left == left
        left, fit, inliers = new_left, refit, window
        if converged:
            break

    start, stop = left, right + 1

    if _travel(xy[start:stop]) < cfg.min_travel_bodies * body_scale_px:
        return None

    stature = body_scale_px / (fit.ay / G)
    if ay_prior is None and not (cfg.min_implied_stature_m <= stature <= cfg.max_implied_stature_m):
        return None

    return FlightPhase(
        fit=fit,
        body_scale_px=body_scale_px,
        implied_stature_m=stature,
        searched_windows=searched,
        n_inliers=int(inliers.sum()),
        n_frames=stop - start,
    )


def bootstrap_uncertainty(
    frames: np.ndarray,
    xy: np.ndarray,
    fps: float,
    n_resamples: int = 400,
    seed: int = 0,
    ay: float | None = None,
) -> dict[str, tuple[float, float]]:
    """Residual-bootstrap 95% intervals for the fitted flight parameters.

    Reported numbers without an error bar invite over-reading. Resampling the
    fit residuals gives an honest spread for the quantities the metrics are
    built from.

    ``ay`` must match whatever the reported metrics were computed under: an
    interval from a free fit does not bound a constrained estimate, and would
    quietly fail to contain the very number it is printed next to.
    """
    base = fit_ballistic(frames, xy, fps, ay)
    if base is None:
        return {}

    frames = np.asarray(frames)
    xy = np.asarray(xy, dtype=float)
    good = ~np.isnan(xy[:, 0])
    f = frames[good]
    t = (f - f[0]) / fps
    observed = xy[good]
    model = base.position(t)
    residual = observed - model

    rng = np.random.default_rng(seed)
    samples: dict[str, list[float]] = {"vx": [], "vy": [], "ay": [], "angle_deg": [], "speed": []}
    for _ in range(n_resamples):
        draw = residual[rng.integers(0, len(residual), len(residual))]
        fit = fit_ballistic(f, model + draw, fps, ay)
        if fit is None:
            continue
        samples["vx"].append(fit.vx)
        samples["vy"].append(fit.vy)
        samples["ay"].append(fit.ay)
        samples["angle_deg"].append(fit.angle_deg)
        samples["speed"].append(fit.speed)

    return {
        key: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
        for key, v in samples.items()
        if v
    }
