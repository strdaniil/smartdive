"""Ballistic flight modelling and automatic phase detection.

The old pipeline fit a degree-3 polynomial to ``x(t)`` and ``y(t)`` over the
whole clip. That has two problems. A cubic has no physical meaning for a body in
free flight, so its coefficients cannot be turned into a velocity, an angle, or a
scale; and fitting over the whole clip mixes three regimes, feet on the block,
free flight, and deceleration in the water, into one curve that describes none
of them.

A body in free flight obeys, exactly::

    x(t) = x0 + vx * t
    y(t) = y0 + vy * t + 0.5 * g * t^2      (y grows downward in image space)

That model has real consequences. Its coefficients *are* the takeoff velocity.
Its quality tells you which frames were actually airborne. And because the
quadratic coefficient must equal gravity, the fit measures the pixels-per-metre
scale of the scene for free, see :mod:`smartdive.calibration`.
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
    ay_stderr: float  # px/s^2, standard error of ay; inf when ay was fixed
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
            "ay_stderr_px_s2": (
                None if not np.isfinite(self.ay_stderr) else round(self.ay_stderr, 1)
            ),
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
    ay_stderr = float("inf")
    if ay is None:
        cy = np.polyfit(t, y, 2)
        # The standard error of the quadratic term is the whole story for
        # whether this window can measure gravity at all. Over a third of a
        # second at 30 fps it is routinely as large as the term itself, which
        # is why the pipeline never trusts a free gravity estimate without
        # checking this first.
        if len(t) > 3:
            try:
                _, covariance = np.polyfit(t, y, 2, cov=True)
                ay_stderr = float(2.0 * np.sqrt(max(covariance[0, 0], 0.0)))
            except (ValueError, np.linalg.LinAlgError):  # pragma: no cover - degenerate window
                ay_stderr = float("inf")
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
        ay_stderr=ay_stderr,
        n_points=int(good.sum()),
        residual_rms=float(np.sqrt(np.mean(residual**2))),
        residual_max=float(residual.max()),
        r_squared=float(r2),
    )


@dataclass
class FlightSearchConfig:
    """Constraints on what counts as a plausible flight phase."""

    #: Shortest acceptable flight, in frames.
    min_flight_frames: int = 5
    #: Minimum travel, in body lengths. A distance gate, not a speed gate, so
    #: slow-motion footage is not thrown away before the clock is recovered.
    min_travel_bodies: float = 0.8
    #: A frame fits if its residual is within this multiple of the measured
    #: noise, or within the floor/ceiling body-length band below, whichever
    #: applies.
    inlier_sigma: float = 1.6
    #: Floor of the tolerance band: stops a quiet clip demanding impossible
    #: precision.
    inlier_floor_bodies: float = 0.05
    #: Ceiling of the tolerance band: stops a noisy clip's tolerance from
    #: swallowing the block phase.
    inlier_ceiling_bodies: float = 0.08
    #: Residual quantile the window is judged at, so one bad joint cannot veto a
    #: clean flight.
    outlier_quantile: float = 0.8
    #: Speed the athlete must already have at the takeoff frame, as a fraction of
    #: takeoff speed. Rules out a takeoff placed while still in the set position.
    min_start_speed_fraction: float = 0.55
    #: How far the first frame may sit from the model, as a multiple of the
    #: tolerance. Stops the window reaching back into the block phase.
    start_tolerance_factor: float = 1.8
    #: Residual multiple above which a frame is dropped from the final fit.
    outlier_tolerance_factor: float = 2.0
    #: Failures allowed before the window stops growing: the greater of this many
    #: frames and outlier_budget_fraction of the frames scanned so far.
    max_outlier_run: int = 2
    outlier_budget_fraction: float = 0.3
    #: Consecutive sub-threshold frames needed to call water entry. One is a
    #: hiccup; two in a row is a swimmer.
    entry_confirm_frames: int = 2
    #: Frames the window may extend past the visibility collapse while still
    #: fitting the flight model.
    max_entry_extend: int = 4
    #: Frames the entry may be pulled back if the last frames do not fit.
    max_entry_trim: int = 3
    #: Implied athlete extent must stay human, given the fitted gravity. Checked
    #: only when gravity was fitted freely (never in the slow-motion scout).
    min_implied_stature_m: float = 1.3
    max_implied_stature_m: float = 2.4


@dataclass
class FlightPhase:
    """The detected airborne window, and the fit that identified it."""

    fit: BallisticFit
    body_scale_px: float
    implied_stature_m: float
    searched_windows: int
    tolerance_px: float = float("nan")
    n_inliers: int = 0
    n_frames: int = 0
    entry_source: str = "visibility"

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
            "tolerance_px": round(self.tolerance_px, 2),
            "n_inliers": self.n_inliers,
            "n_frames": self.n_frames,
            "entry_source": self.entry_source,
            "fit": self.fit.to_dict(),
        }


def _travel(window: np.ndarray) -> float:
    """Path length through a stretch of trajectory, in pixels."""
    window = window[~np.isnan(window[:, 0])]
    if len(window) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(window, axis=0), axis=1)))


def _residuals(fit: BallisticFit, frames: np.ndarray, xy: np.ndarray, fps: float) -> np.ndarray:
    """Per-sample distance from the model, NaN where there is no sample."""
    t = (np.asarray(frames, dtype=float) - fit.start_frame) / fps
    predicted = fit.position(t)
    return np.linalg.norm(predicted - xy, axis=1)


def find_entry(
    usable: np.ndarray,
    airborne: np.ndarray | None,
    peak_frame: int,
    confirm: int = 2,
) -> tuple[int, str]:
    """Locate water entry, and say what evidence found it.

    Entry is read off the *visibility* of the athlete rather than off their
    trajectory, and the reason is worth stating: a streamlined swimmer in the
    first tenth of a second underwater is still travelling on very nearly a
    parabola, so the residual of a ballistic fit cannot see the water at all. A
    pose detector, on the other hand, can only label the joints it can see, so
    the fraction of body mass it resolves falls off a cliff, typically from
    1.0 to about 0.35, the trunk and one arm being the last things above the
    surface, within a frame of the athlete breaking through.

    Searching forward from ``peak_frame`` rather than from the start of the clip
    matters, because the athlete is also partly hidden while crouched on the
    block behind a lane rope, and that earlier dip is not water.
    """
    n = len(usable)
    if airborne is None:
        candidates = np.where(usable)[0]
        return (int(candidates[-1]) if len(candidates) else peak_frame), "last measurement"

    run = 0
    for k in range(max(peak_frame, 0), n):
        if not airborne[k]:
            run += 1
            if run >= confirm:
                collapse = k - run + 1
                before = np.where(usable[:collapse])[0]
                if len(before):
                    return int(before[-1]), "visibility"
                break
        else:
            run = 0

    candidates = np.where(usable)[0]
    return (int(candidates[-1]) if len(candidates) else peak_frame), "last measurement"


def airborne_episodes(
    usable: np.ndarray,
    airborne: np.ndarray | None,
    min_frames: int,
    confirm: int = 2,
) -> list[tuple[int, int]]:
    """Contiguous stretches in which the athlete is measurable and out of the water.

    A clip does not contain one interval of interest, it contains several: the
    dive, and then the swimmer surfacing at the far end of the lane, and on a
    long clip the next swimmer's breakout after that. Each of those is a run of
    frames in which most of the body is visible and moving, and any of them can
    be the fastest thing in the clip, on a 120 fps recording the flight
    advances a dozen pixels a frame while a swimmer breaking the surface thirty
    frames later advances thirty.

    So the search does not begin from a single global maximum. It enumerates the
    episodes and evaluates each one, which costs a handful of extra fits and
    removes an entire class of silently-wrong answer.

    Short interruptions are bridged: ``confirm`` consecutive frames below the
    visibility threshold are needed to end an episode, since a single frame in
    which the detector loses a leg is not water.
    """
    n = len(usable)
    mask = usable if airborne is None else (usable & np.asarray(airborne, dtype=bool))
    episodes: list[tuple[int, int]] = []
    start, run = None, 0

    for k in range(n):
        if mask[k]:
            if start is None:
                start = k
            run = 0
            continue
        if start is None:
            continue
        run += 1
        if run >= confirm:
            stop = k - run
            if stop - start + 1 >= min_frames:
                episodes.append((start, stop))
            start, run = None, 0

    if start is not None and n - 1 - start + 1 >= min_frames:
        episodes.append((start, n - 1))
    return episodes


def _peak_speed_frame(
    frames: np.ndarray, xy: np.ndarray, airborne: np.ndarray | None = None
) -> int:
    """Frame of greatest speed among airborne frames, which is inside the flight.

    Restricting the search to frames where the athlete is actually visible above
    the water is not a refinement, it is load-bearing. On a clip shot at 120 fps
    the flight advances only a dozen or so pixels per frame, while the swimmer
    surfacing thirty frames later, half-hidden, so their apparent centre of
    mass lurches, advances thirty. Without the mask the "peak speed" lands in
    the water and everything anchored to it follows.
    """
    usable = ~np.isnan(xy[:, 0])
    if airborne is not None:
        usable = usable & np.asarray(airborne, dtype=bool)
    good = np.where(usable)[0]
    if len(good) < 3:
        good = np.where(~np.isnan(xy[:, 0]))[0]
    if len(good) < 3:
        return int(good[0]) if len(good) else 0
    steps = np.diff(good)
    speed = np.linalg.norm(np.diff(xy[good], axis=0), axis=1) / steps
    speed[steps > 2] = 0.0
    return int(good[int(np.argmax(speed))])


def detect_flight(
    frames: np.ndarray,
    xy: np.ndarray,
    fps: float,
    body_scale_px: float,
    config: FlightSearchConfig | None = None,
    airborne: np.ndarray | None = None,
    noise_sigma: float | None = None,
    ay_prior: float | None = None,
    require_plausible_scale: bool = True,
) -> FlightPhase | None:
    """Find the airborne window: fix the end by visibility, then grow the start.

    The two ends of a flight are found by different evidence, because each cue
    is sharp at one end and blind at the other.

    **Entry** comes from visibility, as described in :func:`find_entry`.

    **Takeoff** comes from the physics. While a foot is still on the block the
    athlete is being accelerated by their legs, and a constant-acceleration
    model cannot express that, so the fit residual rises steeply and
    unambiguously, typically by an order of magnitude within three frames of
    the break. So the search fixes the entry frame and walks the takeoff frame
    *backwards*, accepting the earliest start for which the whole window still
    fits the projectile model within the measurement noise.

    That direction is the important detail. The previous version of this
    function searched for the short window that fit *best*, and short windows
    always fit better than long ones, a fit has the same number of parameters
    either way, so shrinking the window can only reduce the residual. It
    therefore systematically clipped the flight at both ends: on the sample
    footage it reported 0.27 s for a flight that visibly lasts 0.40 s, and every
    velocity derived from that window inherited the error. Asking for the
    *longest* window consistent with the noise removes the bias, and the noise
    threshold is estimated independently of the window so it cannot be gamed.
    """
    cfg = config or FlightSearchConfig()
    frames = np.asarray(frames)
    xy = np.asarray(xy, dtype=float)
    n = len(frames)

    if not np.isfinite(body_scale_px) or body_scale_px <= 0 or n < cfg.min_flight_frames:
        return None

    usable = ~np.isnan(xy[:, 0])
    if usable.sum() < cfg.min_flight_frames:
        return None

    airborne_mask = None if airborne is None else np.asarray(airborne, dtype=bool)

    sigma = noise_sigma if noise_sigma is not None and np.isfinite(noise_sigma) else 0.0
    tolerance = float(
        np.clip(
            cfg.inlier_sigma * sigma,
            cfg.inlier_floor_bodies * body_scale_px,
            cfg.inlier_ceiling_bodies * body_scale_px,
        )
    )

    episodes = airborne_episodes(
        usable, airborne_mask, cfg.min_flight_frames, cfg.entry_confirm_frames
    )
    if not episodes:
        episodes = [(int(np.argmax(usable)), int(n - 1 - np.argmax(usable[::-1])))]

    # Score every (start, stop) candidate across every episode before judging
    # any of them: the acceptance bar is relative to the best residual the whole
    # search achieved, so it cannot be computed candidate by candidate.
    candidates: list[tuple[int, int, list[tuple[int, float, float, BallisticFit]]]] = []
    searched = 0
    entry_source = "visibility" if airborne_mask is not None else "last measurement"

    for episode_start, episode_stop in episodes:
        window_usable = np.zeros(n, dtype=bool)
        window_usable[episode_start : episode_stop + 1] = usable[episode_start : episode_stop + 1]
        if window_usable.sum() < cfg.min_flight_frames:
            continue

        peak = _peak_speed_frame(frames, xy, window_usable)
        entry = int(np.where(window_usable)[0][-1])

        # The last frames may be given up if that lets a longer flight be found.
        # How far the entry may be pulled back depends on how well it is known: a
        # visibility collapse is sharp and trusted to a frame or two; without one
        # the entry has to be searched for properly.
        trim_budget = (
            cfg.max_entry_trim
            if airborne_mask is not None
            else max(cfg.max_entry_trim, entry - peak - cfg.min_flight_frames)
        )
        for trim in range(max(trim_budget, 0) + 1):
            stop = entry - trim
            if stop - peak < 1 and trim:
                break
            scored = _score_starts(frames, xy, window_usable, fps, stop, peak, ay_prior, cfg)
            searched += len(scored)
            if scored:
                candidates.append((episode_start, stop, scored))

    if not candidates:
        return None

    # Rank surviving windows by residual per unit of travel, not by travel.
    # Travel alone is maximised by running on into the water, where the athlete
    # is still moving but no longer ballistic; a scale-free residual is not.
    best: tuple[int, int, BallisticFit] | None = None
    best_score = np.inf
    for _episode_start, stop, scored in candidates:
        accepted = _accept_earliest(scored, tolerance, cfg)
        if accepted is None:
            continue
        start, quantile, fit = accepted
        travel = _travel(xy[start : stop + 1])
        if travel < cfg.min_travel_bodies * body_scale_px:
            continue
        score = quantile / travel
        if score < best_score:
            best, best_score = (start, stop, fit), score

    if best is None:
        return None

    start, stop, fit = best
    stop = _extend_entry(frames, xy, usable, fps, start, stop, tolerance, ay_prior, cfg)
    if airborne_mask is not None and stop > int(np.where(usable & airborne_mask)[0][-1]):
        entry_source = "visibility, extended by fit"

    # Refit without the gross outliers, keeping the window itself unchanged.
    window = np.zeros(n, dtype=bool)
    window[start : stop + 1] = usable[start : stop + 1]
    residuals = _residuals(fit, frames, xy, fps)
    keep = window & (residuals <= cfg.outlier_tolerance_factor * tolerance)
    if keep.sum() >= max(4, cfg.min_flight_frames - 1):
        # The endpoints define the window and are never dropped: losing one
        # would silently shorten the flight the search just worked to find.
        keep[start] = usable[start]
        keep[stop] = usable[stop]
        refit = fit_ballistic(frames, np.where(keep[:, None], xy, np.nan), fps, ay_prior)
        if refit is not None and refit.ay > 0:
            fit, window = refit, keep

    if _travel(xy[start : stop + 1]) < cfg.min_travel_bodies * body_scale_px:
        return None

    stature = body_scale_px / (fit.ay / G) if fit.ay > 0 else float("nan")
    if (
        require_plausible_scale
        and ay_prior is None
        and not (cfg.min_implied_stature_m <= stature <= cfg.max_implied_stature_m)
    ):
        return None

    return FlightPhase(
        fit=fit,
        body_scale_px=body_scale_px,
        implied_stature_m=stature,
        searched_windows=searched,
        tolerance_px=tolerance,
        n_inliers=int(window.sum()),
        n_frames=stop - start + 1,
        entry_source=entry_source,
    )


def _extend_entry(
    frames: np.ndarray,
    xy: np.ndarray,
    usable: np.ndarray,
    fps: float,
    start: int,
    stop: int,
    tolerance: float,
    ay_prior: float | None,
    cfg: FlightSearchConfig,
) -> int:
    """Push the entry frame forward while the athlete is still on the parabola.

    The visibility cue that fixes entry fires when the detector stops resolving
    most of the body, and a diver fully extended a few centimetres above the
    water loses their legs to a pose detector slightly before they lose them to
    the water. Those frames are flight, they fit the flight model, and dropping
    them costs a tenth of a second of a third-of-a-second measurement.

    The extension is safe in the direction that matters because water is not
    gravity: a body decelerating in it leaves the fixed-acceleration model
    immediately and conspicuously, so the residual test that admits the extra
    airborne frames refuses the submerged ones.
    """
    fit = fit_ballistic(frames[start : stop + 1], xy[start : stop + 1], fps, ay_prior)
    if fit is None:
        return stop

    n = len(frames)
    extended, run = stop, 0
    for k in range(stop + 1, min(n, stop + cfg.max_entry_extend + 1)):
        if not usable[k]:
            run += 1
            if run > cfg.max_outlier_run:
                break
            continue
        t = np.array([(frames[k] - fit.start_frame) / fps])
        residual = float(np.linalg.norm(fit.position(t)[0] - xy[k]))
        if residual <= tolerance:
            extended, run = k, 0
        else:
            run += 1
            if run > cfg.max_outlier_run:
                break
    return extended


def _score_starts(
    frames: np.ndarray,
    xy: np.ndarray,
    usable: np.ndarray,
    fps: float,
    stop: int,
    peak: int,
    ay_prior: float | None,
    cfg: FlightSearchConfig,
) -> list[tuple[int, float, float, BallisticFit]]:
    """Fit ``[start, stop]`` for every plausible start, without judging any of them.

    Scoring is separated from accepting because the acceptance bar depends on
    the best residual anywhere in the search, and that is not known until the
    whole search has been scored. Each entry is
    ``(start, residual quantile, residual at the start frame, fit)``.
    """
    latest_start = min(peak, stop - cfg.min_flight_frames + 1)
    scored: list[tuple[int, float, float, BallisticFit]] = []

    for start in range(latest_start, -1, -1):
        if not usable[start]:
            continue
        window = slice(start, stop + 1)
        if usable[window].sum() < max(4, cfg.min_flight_frames - 1):
            continue

        fit = fit_ballistic(frames[window], xy[window], fps, ay_prior)
        if fit is None or fit.ay <= 0:
            continue

        if _local_speed(xy, start, stop) < cfg.min_start_speed_fraction * abs(fit.speed) / fps:
            continue

        residuals = _residuals(fit, frames[window], xy[window], fps)
        first = residuals[0]
        finite = residuals[np.isfinite(residuals)]
        if not len(finite) or not np.isfinite(first):
            continue
        scored.append((start, float(np.quantile(finite, cfg.outlier_quantile)), float(first), fit))

    return scored


def _local_speed(xy: np.ndarray, index: int, limit: int) -> float:
    """Measured speed at ``index`` in px per frame, from the next usable sample."""
    for k in range(index + 1, min(index + 3, limit + 1)):
        if not np.isnan(xy[k, 0]) and not np.isnan(xy[index, 0]):
            return float(np.linalg.norm(xy[k] - xy[index]) / (k - index))
    return float("inf")


def _accept_earliest(
    scored: list[tuple[int, float, float, BallisticFit]],
    tolerance: float,
    cfg: FlightSearchConfig,
) -> tuple[int, float, BallisticFit] | None:
    """Walk back through scored starts and stop where the fit gives out.

    The test is on the residual *at the proposed takeoff frame*, and only there.
    While a foot is still on the block the athlete is being accelerated by their
    legs, which a constant-acceleration model cannot express, so that one
    frame's residual climbs steeply and unambiguously, on the sample footage
    it runs 15, 21, 30, 41, 56 px over five consecutive candidate frames.

    Judging the *window* instead, by some summary of all its residuals, does not
    work, and not only because a summary dilutes a signal carried by one frame.
    Any threshold on a whole-window statistic has to scale with the window, and
    a window that reaches one frame further back into the block phase is also
    one frame longer, so the threshold relaxes exactly as the evidence against
    the window mounts. That feedback is not a tuning problem, it is a wrong
    shape of test: it made the reported flight time swing by a quarter between
    two runs that differed only in where the entry frame landed.
    """
    accepted: tuple[int, float, BallisticFit] | None = None
    failures = 0
    for seen, (start, quantile, first, fit) in enumerate(scored, start=1):
        if first > cfg.start_tolerance_factor * tolerance:
            # Once the window has eaten into the block phase the residual only
            # gets worse, so there is nothing to gain by looking further back,
            # but a badly localised frame or two should not end the search.
            failures += 1
            budget = max(cfg.max_outlier_run, cfg.outlier_budget_fraction * seen)
            if accepted is not None and failures > budget:
                break
            continue
        accepted = (start, quantile, fit)
    return accepted


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
