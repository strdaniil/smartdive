"""End-to-end analysis: video in, calibrated dive metrics out.

The pipeline runs in one direction with one loop in it. Detection, camera-motion
estimation, association, subject choice and skeleton conditioning all happen
once; the physics then runs twice, because two of its inputs are things it has
to find the flight in order to measure.

The first pass fits the flight with gravity left free. That fit is a poor
*measurement* of gravity, over a third of a second it is barely determined,
but it is enough to bracket the flight, and its acceleration, compared against
the scale read off the athlete's own body, is what reveals a slow-motion clip
whose container frame rate is lying about when things happened. The second pass
re-runs the search with gravity pinned and the clock corrected, which sharpens
the residual into a much better test of which frames were airborne.
"""

from __future__ import annotations

import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from smartdive import anthro
from smartdive import calibration as calib
from smartdive.backends import FrameDetections, PoseBackend, get_backend
from smartdive.camera import CameraConfig, CameraMotion, estimate_motion, identity_motion
from smartdive.camera import stabilise_detections as stabilise
from smartdive.metrics import (
    DiveMetrics,
    compute_metrics,
    front_knee_angle,
    knee_angle_feedback,
)
from smartdive.physics import (
    BallisticFit,
    FlightPhase,
    FlightSearchConfig,
    G,
    bootstrap_uncertainty,
    detect_flight,
    fit_ballistic,
)
from smartdive.pose import (
    PoseSeries,
    build_pose_series,
    landmark_series,
    lateral_view_score,
    prepare,
)
from smartdive.refine import (
    RefineConfig,
    RefineReport,
    apply_refinement,
    refine_frame_detections,
    refine_subject,
    reject_inconsistent,
)
from smartdive.signal import (
    ConditionedTrajectory,
    SmoothingConfig,
    jitter,
    noise_scale,
)
from smartdive.tracking import SubjectScore, Track, TrackerConfig, select_subject, track_all
from smartdive.video import VideoInfo, probe
from smartdive.video import frames as iter_frames


@dataclass
class AnalysisConfig:
    """Everything the pipeline can be told to do differently."""

    landmark: str = "com"
    backend: str = "yolo"
    backend_options: dict[str, object] = field(default_factory=dict)
    min_keypoint_conf: float = 0.3
    smoothing: SmoothingConfig = field(default_factory=SmoothingConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    flight: FlightSearchConfig = field(default_factory=FlightSearchConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    refinement: RefineConfig = field(default_factory=RefineConfig)

    #: Re-detect the athlete on a crop that follows them, at full resolution.
    #: This is where the flight frames come from: a whole-frame pass loses a
    #: motion-blurred body over the water, and a flight that is not detected
    #: cannot be measured. See :mod:`smartdive.refine`.
    refine: bool = True

    #: Estimate and subtract camera motion before doing any physics. Off only
    #: makes sense for genuinely locked-off footage, and costs nothing there.
    stabilise: bool = True
    #: Fill a missing keypoint from its contralateral partner. Valid for the
    #: side-on view a racing start is normally filmed from; see
    #: :func:`smartdive.pose.mirror_missing`.
    mirror_keypoints: bool = True
    #: Allow the pixels-per-metre scale to vary through the clip as the athlete's
    #: distance from the camera changes. Only takes effect when the change is
    #: big enough to matter.
    depth_scale: bool = True
    #: Fractional change in apparent size across the flight below which the
    #: depth correction is not worth the noise it introduces.
    depth_scale_trigger: float = 0.12
    #: Recover the real capture rate from gravity for slow-motion clips.
    detect_slow_motion: bool = True

    #: Manual calibration overrides. ``reference_px``/``reference_m`` together
    #: pin the scale to a measured object; ``stature_m`` uses athlete height.
    reference_px: float | None = None
    reference_m: float | None = None
    stature_m: float | None = None
    #: Override the container frame rate, for footage whose metadata is wrong
    #: and which the automatic detector cannot resolve.
    capture_fps: float | None = None

    #: Bootstrap resample count for the uncertainty report (0 disables).
    bootstrap: int = 400

    #: Override automatic phase detection with an explicit ``(takeoff, entry)``
    #: frame range. Automatic detection is right most of the time, not all of
    #: the time, and this is the escape hatch for the rest.
    flight_window: tuple[int, int] | None = None

    #: Fraction of body mass that must be visible for a frame to count as
    #: airborne. Once the athlete breaks the surface the submerged half stops
    #: being detectable, so this doubles as a water-entry detector.
    airborne_min_mass: float = 0.6

    def to_dict(self) -> dict[str, object]:
        return {
            "landmark": self.landmark,
            "backend": self.backend,
            "backend_options": dict(self.backend_options),
            "min_keypoint_conf": self.min_keypoint_conf,
            "stabilise": self.stabilise,
            "refine": self.refine,
            "mirror_keypoints": self.mirror_keypoints,
            "depth_scale": self.depth_scale,
            "detect_slow_motion": self.detect_slow_motion,
            "reference_px": self.reference_px,
            "reference_m": self.reference_m,
            "stature_m": self.stature_m,
            "capture_fps": self.capture_fps,
            "flight_window": list(self.flight_window) if self.flight_window else None,
        }


@dataclass
class DiveAnalysis:
    """The complete result of analysing one clip."""

    video: VideoInfo
    config: AnalysisConfig
    detections: list[FrameDetections]
    motion: CameraMotion
    tracks: list[Track]
    subject: Track
    subject_ranking: list[SubjectScore]
    refinement: RefineReport | None
    pose: PoseSeries
    trajectory: ConditionedTrajectory
    body: anthro.BodySize
    body_scale_px: float
    lateral_score: float
    airborne: np.ndarray
    flight: FlightPhase | None
    calibration: calib.Calibration | None
    time_base: calib.TimeBase
    metrics: DiveMetrics | None
    uncertainty: dict[str, tuple[float, float]]
    warnings: list[str]
    elapsed_s: float
    backend_info: dict[str, object]

    @property
    def fit(self) -> BallisticFit | None:
        return self.flight.fit if self.flight else None

    @property
    def fps(self) -> float:
        """Frames per real second, after any slow-motion correction."""
        return self.time_base.effective_fps

    def quality(self) -> dict[str, float]:
        """Signal-quality numbers, useful for regression testing a change."""
        return {
            "coverage": self.trajectory.coverage,
            "jitter_raw_px": jitter(self.trajectory.raw),
            "jitter_conditioned_px": jitter(self.trajectory.xy),
            "outliers_rejected": float(self.trajectory.outliers.sum()),
            "gaps_filled": float(self.trajectory.interpolated.sum()),
            "subject_switches": float(subject_switches(self.trajectory.raw, self.body_scale_px)),
            "keypoints_mirrored": float(self.pose.mirrored.sum()),
            "lateral_view": self.lateral_score,
            "camera_drift_px": float(np.max(self.motion.displacement_px()))
            if len(self.motion)
            else 0.0,
        }

    def to_dict(self) -> dict[str, object]:
        return {
            "video": self.video.to_dict(),
            "config": self.config.to_dict(),
            "backend": self.backend_info,
            "elapsed_s": round(self.elapsed_s, 2),
            "camera": self.motion.to_dict(),
            "refinement": self.refinement.to_dict() if self.refinement else None,
            "time_base": self.time_base.to_dict(),
            "subject": {
                "track_id": self.subject.track_id,
                "frames": [int(self.subject.frames[0]), int(self.subject.frames[-1])],
                "n_observations": len(self.subject.observations),
                "ranking": [s.to_dict() for s in self.subject_ranking],
            },
            "body": self.body.to_dict(),
            "body_scale_px": round(self.body_scale_px, 1),
            "quality": {
                k: (None if not np.isfinite(v) else round(v, 4)) for k, v in self.quality().items()
            },
            "flight": self.flight.to_dict() if self.flight else None,
            "calibration": self.calibration.to_dict() if self.calibration else None,
            "metrics": self.metrics.to_dict() if self.metrics else None,
            "uncertainty_95": {
                k: [round(lo, 3), round(hi, 3)] for k, (lo, hi) in self.uncertainty.items()
            },
            "warnings": list(self.warnings),
        }


def subject_switches(raw_xy: np.ndarray, body_scale_px: float) -> int:
    """Count frame-to-frame jumps larger than one body length.

    A person cannot move their centre of mass a whole body length in one frame
    at 30 fps. Every such jump is the tracker having swapped to somebody else,
    which makes this a direct, assumption-free measure of identity errors.
    """
    if not np.isfinite(body_scale_px) or body_scale_px <= 0:
        return 0
    good = ~np.isnan(raw_xy[:, 0])
    idx = np.where(good)[0]
    if len(idx) < 2:
        return 0
    steps = np.linalg.norm(np.diff(raw_xy[idx], axis=0), axis=1)
    contiguous = np.diff(idx) == 1
    return int(np.sum((steps > body_scale_px) & contiguous))


def _cache_path(cache: Path, video: Path, config: AnalysisConfig, suffix: str = "") -> Path:
    """Cache key covers everything that changes what the detector produces."""
    options = "_".join(f"{k}{v}" for k, v in sorted(config.backend_options.items()))
    stem = f"{video.stem}__{config.backend}__{options or 'default'}{suffix}"
    return Path(cache) / f"{stem}.pkl"


def _detect(
    video_path: Path,
    info: VideoInfo,
    cfg: AnalysisConfig,
    backend: PoseBackend | None,
    progress: bool,
    cache: Path | None,
) -> tuple[list[FrameDetections], dict[str, object]]:
    backend_info: dict[str, object] = {"backend": cfg.backend, **cfg.backend_options}
    cached_at = _cache_path(cache, video_path, cfg) if cache is not None else None
    if cached_at is not None and cached_at.exists():
        payload = pickle.loads(cached_at.read_bytes())
        return payload["detections"], payload["backend_info"]

    owns_backend = backend is None
    if backend is None:
        backend = get_backend(cfg.backend, **cfg.backend_options)
    try:
        detections = []
        for index, frame in iter_frames(video_path):
            detections.append(backend.detect(frame, index))
            if progress and index % 25 == 0:
                print(f"  detect frame {index}/{info.frame_count}", flush=True)
        backend_info = backend.describe()
    finally:
        if owns_backend:
            backend.close()

    if cached_at is not None:
        cached_at.parent.mkdir(parents=True, exist_ok=True)
        payload = {"detections": detections, "backend_info": backend_info}
        cached_at.write_bytes(pickle.dumps(payload))
    return detections, backend_info


def _camera_motion(
    video_path: Path,
    detections: list[FrameDetections],
    cfg: AnalysisConfig,
    cache: Path | None,
) -> CameraMotion:
    if not cfg.stabilise:
        return identity_motion(len(detections))
    cached_at = _cache_path(cache, video_path, cfg, "__motion") if cache is not None else None
    if cached_at is not None and cached_at.exists():
        payload = pickle.loads(cached_at.read_bytes())
        if len(payload.to_reference) == len(detections):
            return payload
    motion = estimate_motion(video_path, detections, cfg.camera)
    if cached_at is not None:
        cached_at.parent.mkdir(parents=True, exist_ok=True)
        cached_at.write_bytes(pickle.dumps(motion))
    return motion


def _trajectory(
    raw: np.ndarray, conditioned: np.ndarray, fps: float, pose: PoseSeries
) -> ConditionedTrajectory:
    """Package the landmark before and after conditioning, for reporting."""
    valid = ~np.isnan(raw[:, 0])
    filled = ~np.isnan(conditioned[:, 0]) & ~valid
    return ConditionedTrajectory(
        frames=pose.frames,
        raw=raw,
        xy=conditioned,
        valid=valid,
        outliers=np.zeros(len(raw), dtype=bool),
        interpolated=filled,
        fps=fps,
        noise_sigma=noise_scale(raw),
    )


def analyze(
    video_path: str | Path,
    config: AnalysisConfig | None = None,
    backend: PoseBackend | None = None,
    progress: bool = False,
    cache: str | Path | None = None,
) -> DiveAnalysis:
    """Analyse a dive video end to end.

    Parameters
    ----------
    video_path:
        Clip containing a single start. Extra people in frame are fine.
    config:
        Pipeline settings; the defaults are tuned for poolside 30 fps footage.
    backend:
        A pre-built pose backend, to amortise model loading across many clips.
    progress:
        Print a one-line-per-stage progress trace.
    cache:
        Directory in which to memoise raw detections and the camera-motion
        estimate. Inference dominates the runtime, so caching it makes iterating
        on everything downstream effectively free.
    """
    cfg = config or AnalysisConfig()
    started = time.perf_counter()
    warnings: list[str] = []

    video_path = Path(video_path)
    info = probe(video_path)
    cache_dir = Path(cache) if cache is not None else None

    detections, backend_info = _detect(video_path, info, cfg, backend, progress, cache_dir)
    n_frames = len(detections)
    if n_frames == 0:
        raise ValueError(f"no frames decoded from {info.path}")

    motion = _camera_motion(video_path, detections, cfg, cache_dir)
    if cfg.stabilise and len(motion) != n_frames:
        motion = identity_motion(n_frames)
        warnings.append("camera motion estimation failed; the clip is assumed to be locked off")
    stabilised = stabilise(detections, motion) if cfg.stabilise else detections

    container_fps = cfg.capture_fps or info.fps

    tracks = track_all(stabilised, cfg.tracker)
    if not tracks:
        raise ValueError(f"no people detected in {info.path}")

    subject, ranking = select_subject(tracks, cfg.tracker, container_fps)
    contenders = [s for s in ranking if not s.rejected]
    if len(contenders) > 1 and contenders[0].total - contenders[1].total < 0.6:
        warnings.append(
            "subject selection was close between tracks "
            f"{contenders[0].track_id} and {contenders[1].track_id}; check the overlay"
        )

    refinement: RefineReport | None = None
    if cfg.refine:
        refined, refinement = _refine(
            video_path, subject, cfg, backend, motion, n_frames, cache_dir, progress
        )
        if refined:
            refined = reject_inconsistent(subject, refined)
            subject = apply_refinement(subject, refined)
            detections = refine_frame_detections(detections, refined)

    # --- skeleton ---------------------------------------------------------
    # Kept unconditioned on purpose: the noise estimate that sets the flight
    # detector's tolerance has to come from the signal as measured, because
    # conditioning correlates neighbouring samples and would make the
    # measurement look quieter than it was.
    raw_pose = build_pose_series(subject, n_frames, container_fps, cfg.min_keypoint_conf)
    pose = prepare(
        subject,
        n_frames,
        container_fps,
        cfg.min_keypoint_conf,
        cfg.smoothing,
        mirror=cfg.mirror_keypoints,
    )
    lateral = lateral_view_score(pose)
    if np.isfinite(lateral) and lateral < 0.35:
        warnings.append(
            f"the camera looks close to head-on (lateral score {lateral:.2f}); the planar "
            "projectile model assumes the dive runs across the frame, so distances and "
            "speeds will be underestimated. Film square-on to the lane where possible."
        )

    landmark = landmark_series(pose, cfg.landmark)
    raw_landmark = landmark_series(raw_pose, cfg.landmark)
    trajectory = _trajectory(raw_landmark, landmark, container_fps, pose)
    body_scale_px = float(np.nanmedian(pose.body_scale()))

    if trajectory.coverage < 0.25:
        warnings.append(
            f"only {trajectory.coverage:.0%} of frames yielded a {cfg.landmark} "
            "measurement; try --landmark hips or a larger --imgsz"
        )

    visible_mass = pose.visible_mass(cfg.min_keypoint_conf)
    airborne = visible_mass >= cfg.airborne_min_mass
    # A frame with no detection at all is not evidence of being underwater, so
    # it must not end the flight; whether it is usable is decided separately by
    # whether the trajectory has a value there.
    airborne |= ~pose.measured.any(axis=1)

    measured = pose.measured & ~pose.mirrored
    body = anthro.measure(pose.xy, measured)
    if not np.isfinite(body.stature_px):
        raise ValueError(
            "could not measure the athlete's size from any body segment; the pose "
            "detections are too sparse to calibrate this clip"
        )

    prior, _ = calib.resolve(
        body,
        stature_m=cfg.stature_m,
        reference_px=cfg.reference_px,
        reference_m=cfg.reference_m,
    )

    # --- flight, pass 1: gravity free, clock as-is -------------------------
    if cfg.flight_window is not None:
        flight = _manual_flight(cfg, trajectory, container_fps, body_scale_px, prior, warnings)
        time_base = calib.TimeBase(container_fps, container_fps, 1.0, "container")
    else:
        scout = detect_flight(
            trajectory.frames,
            trajectory.xy,
            container_fps,
            body_scale_px,
            cfg.flight,
            airborne,
            trajectory.noise_sigma,
            ay_prior=None,
            require_plausible_scale=not cfg.detect_slow_motion,
        )
        time_base, time_notes = _resolve_time_base(
            cfg, scout, container_fps, prior, body, pose, trajectory
        )
        warnings.extend(time_notes)

        ay_prior = G * prior.px_per_metre * time_base.factor**2
        constrained = detect_flight(
            trajectory.frames,
            trajectory.xy,
            container_fps,
            body_scale_px,
            cfg.flight,
            airborne,
            trajectory.noise_sigma,
            ay_prior=ay_prior,
        )
        flight = _better_flight(constrained, scout)

    calibration: calib.Calibration | None = None
    metrics: DiveMetrics | None = None
    uncertainty: dict[str, tuple[float, float]] = {}

    if flight is None:
        warnings.append(
            "no ballistic flight phase found: the clip may not contain a dive, the "
            "athlete may leave frame, or the pose detections may be too sparse "
            "through the flight. Use --flight-window to name the frames directly."
        )
        if cfg.flight_window is None:
            time_base = calib.TimeBase(container_fps, container_fps, 1.0, "container")
    else:
        window = np.arange(flight.fit.start_frame, flight.fit.end_frame + 1)

        # Re-measure the athlete over the flight itself. On an obliquely placed
        # camera they are a different size there than they were on the block,
        # and it is the flight that is being measured.
        flight_body = anthro.measure(pose.xy, measured, window)
        if not np.isfinite(flight_body.stature_px):
            flight_body = body

        relative = None
        if cfg.depth_scale:
            trend, change = _depth_trend(
                anthro.relative_size_series(pose.xy, measured, flight_body), window
            )
            if trend is not None and np.isfinite(change) and change >= cfg.depth_scale_trigger:
                relative = trend
                warnings.append(
                    f"the athlete's apparent size changes by {change:.0%} across the flight, "
                    "so they are moving towards or away from the camera; distances are "
                    "integrated with a per-frame scale, but out-of-plane travel is still "
                    "invisible to a single camera and speeds remain a lower bound"
                )

        free_fit = fit_ballistic(
            trajectory.frames[window], trajectory.xy[window], time_base.effective_fps
        )
        calibration, notes = calib.resolve(
            flight_body,
            stature_m=cfg.stature_m,
            reference_px=cfg.reference_px,
            reference_m=cfg.reference_m,
            fit=free_fit,
            relative_size=relative,
        )
        warnings.extend(notes)
        if calibration.method == "stature_assumed":
            warnings.append(
                f"absolute distances assume a {anthro.DEFAULT_STATURE_M:.2f} m athlete; "
                "pass --stature-m or --reference-px/--reference-m to make them exact. "
                "Times and angles are unaffected."
            )
        ok, note = calib.plausibility(calibration, flight_body.stature_px)
        if not ok:
            warnings.append(note)

        # Restate the flight in real seconds. Everything downstream reads the
        # fit's velocities directly, so the correction belongs here rather than
        # sprinkled through the metrics.
        final_fit = fit_ballistic(
            trajectory.frames[window],
            trajectory.xy[window],
            time_base.effective_fps,
            G * calibration.px_per_metre,
        )
        if final_fit is not None:
            flight = FlightPhase(
                fit=final_fit,
                body_scale_px=flight.body_scale_px,
                implied_stature_m=flight_body.stature_px / calibration.px_per_metre,
                searched_windows=flight.searched_windows,
                tolerance_px=flight.tolerance_px,
                n_inliers=flight.n_inliers,
                n_frames=flight.n_frames,
                entry_source=flight.entry_source,
            )

        knee_angle = None
        set_pose = _set_position(subject, flight.fit.start_frame)
        if set_pose is not None:
            angle, _side = front_knee_angle(*set_pose, cfg.min_keypoint_conf)
            if np.isfinite(angle):
                knee_angle = angle
                warnings.append("set position: " + knee_angle_feedback(angle))

        block_time = None
        first_frame = int(subject.frames[0])
        if flight.fit.start_frame > first_frame:
            block_time = (flight.fit.start_frame - first_frame) / time_base.effective_fps

        metrics = compute_metrics(
            flight.fit,
            calibration,
            trajectory_px=trajectory.xy[window],
            frames=window,
            block_time_s=block_time,
            set_knee_angle_deg=knee_angle,
            time_base=time_base,
        )
        warnings.extend(calib.plausible_start(metrics.takeoff_speed_ms, metrics.flight_time_s))

        if cfg.bootstrap:
            uncertainty = bootstrap_uncertainty(
                trajectory.frames[window],
                trajectory.xy[window],
                time_base.effective_fps,
                n_resamples=cfg.bootstrap,
                ay=flight.fit.ay,
            )

    return DiveAnalysis(
        video=info,
        config=cfg,
        detections=detections,
        motion=motion,
        tracks=tracks,
        subject=subject,
        subject_ranking=ranking,
        refinement=refinement,
        pose=pose,
        trajectory=trajectory,
        body=body,
        body_scale_px=body_scale_px,
        lateral_score=lateral,
        airborne=airborne,
        flight=flight,
        calibration=calibration,
        time_base=time_base,
        metrics=metrics,
        uncertainty=uncertainty,
        warnings=warnings,
        elapsed_s=time.perf_counter() - started,
        backend_info=backend_info,
    )


def _refine(
    video_path: Path,
    subject: Track,
    cfg: AnalysisConfig,
    backend: PoseBackend | None,
    motion: CameraMotion,
    n_frames: int,
    cache: Path | None,
    progress: bool,
):
    """Run (or reload) the follow-crop second detection pass for the athlete."""
    span = (int(subject.frames[0]), int(subject.frames[-1])) if len(subject.frames) else (0, 0)
    cached_at = _cache_path(cache, video_path, cfg, "__refine") if cache is not None else None
    if cached_at is not None and cached_at.exists():
        payload = pickle.loads(cached_at.read_bytes())
        if payload.get("span") == span:
            return payload["refined"], payload["report"]

    owns_backend = backend is None
    if backend is None:
        options = dict(cfg.backend_options)
        if cfg.backend == "yolo":
            options.setdefault("imgsz", cfg.refinement.imgsz)
        try:
            backend = get_backend(cfg.backend, **options)
        except ImportError:
            return {}, None
    try:
        refined, report = refine_subject(
            video_path, subject, backend, motion, n_frames, cfg.refinement, progress
        )
    finally:
        if owns_backend:
            backend.close()

    if cached_at is not None:
        cached_at.parent.mkdir(parents=True, exist_ok=True)
        cached_at.write_bytes(pickle.dumps({"span": span, "refined": refined, "report": report}))
    return refined, report


def _better_flight(
    constrained: FlightPhase | None, scout: FlightPhase | None
) -> FlightPhase | None:
    """Choose between the pinned-gravity search and the free one.

    Pinning gravity sharpens the residual into a better test of which frames
    were airborne, so its answer is preferred, but only when it found one.
    Pinning it to a *prior* also means that when the prior and the footage
    disagree, the constrained model fits worse everywhere and the search can
    come back empty on a clip where the flight is perfectly visible. Falling
    back to the free window there is not a failure, it is the reason both
    searches are run.
    """
    if constrained is None:
        return scout
    if scout is None:
        return constrained
    return constrained if constrained.n_frames + 2 >= scout.n_frames else scout


#: Fraction of the scouted flight window dropped from the start before gravity
#: is re-measured for the time base. See :func:`_resolve_time_base`.
_TIME_BASE_TRIM = 0.3


def _resolve_time_base(
    cfg: AnalysisConfig,
    scout: FlightPhase | None,
    container_fps: float,
    prior: calib.Calibration,
    body: anthro.BodySize,
    pose: PoseSeries,
    trajectory: ConditionedTrajectory,
) -> tuple[calib.TimeBase, list[str]]:
    if not cfg.detect_slow_motion or scout is None:
        return calib.TimeBase(container_fps, container_fps, 1.0, "container"), []

    # Re-fit gravity over the *later* part of the scouted window. The scout runs
    # with gravity free, which is what lets it bracket a clip whose clock is
    # unknown, but it also means the fit can flatten its parabola to absorb a
    # few frames of block push at the start, and the flatter the parabola, the
    # slower the clip appears to be. Measured on the sample footage, dropping
    # the first third of the window moves the recovered capture rate from 220 to
    # 138 fps on a clip that was actually shot at 120, and from 143 to 118 on
    # one shot at 120. The tail of a flight is unambiguously airborne, so
    # nothing is risked by reading the curvature there.
    first, last = scout.fit.start_frame, scout.fit.end_frame
    inner = round(first + _TIME_BASE_TRIM * (last - first))
    fit = fit_ballistic(np.arange(inner, last + 1), trajectory.xy[inner : last + 1], container_fps)
    if fit is None:
        fit = scout.fit
        inner = first

    window = np.arange(inner, last + 1)
    measured = pose.measured & ~pose.mirrored
    local = anthro.measure(pose.xy, measured, window)
    px_per_metre = prior.px_per_metre
    if np.isfinite(local.stature_px) and prior.stature_m:
        px_per_metre = local.stature_px / prior.stature_m
    return calib.resolve_time_base(container_fps, fit, px_per_metre, n_points=fit.n_points)


def _depth_trend(
    relative: np.ndarray | None, window: np.ndarray
) -> tuple[np.ndarray | None, float]:
    """Fit a smooth depth trend to the athlete's apparent size over the flight.

    The per-frame size estimate is noisy, it is a median over a handful of
    projected bone lengths, each of which foreshortens as the athlete rotates,
    and feeding that noise straight into the scale would add error rather than
    remove it. What the correction actually needs is the *trend*, because depth
    changes smoothly: an athlete cannot get closer to the camera and back again
    within a tenth of a second.

    So a straight line is fitted to log-size over the flight, and it is only
    accepted if its slope is several times its own standard error. That
    distinguishes a real recession from noise, which a max-over-min ratio, the
    obvious thing to compute, cannot: on ``adive_wide`` the raw ratio reads
    31% across eight frames of a dive that barely changes depth at all.
    """
    if relative is None:
        return None, float("nan")
    values = relative[window]
    good = np.isfinite(values) & (values > 0)
    if good.sum() < 6:
        return None, float("nan")

    t = window[good].astype(float)
    y = np.log(values[good])
    coefficients, residuals, rank, *_ = np.linalg.lstsq(
        np.vstack([t, np.ones_like(t)]).T, y, rcond=None
    )
    slope, intercept = coefficients
    n = len(t)
    if rank < 2 or n <= 2:
        return None, float("nan")
    sigma2 = float(residuals[0]) / (n - 2) if len(residuals) else 0.0
    variance = sigma2 / max(float(np.sum((t - t.mean()) ** 2)), 1e-9)
    stderr = float(np.sqrt(variance))
    if stderr <= 0 or abs(slope) < 3.0 * stderr:
        return None, float("nan")

    change = float(np.exp(abs(slope) * (window[-1] - window[0])) - 1.0)
    trend = np.full(len(relative), np.nan)
    trend[window] = np.exp(slope * window.astype(float) + intercept)
    return trend, change


def _manual_flight(
    cfg: AnalysisConfig,
    trajectory: ConditionedTrajectory,
    fps: float,
    body_scale_px: float,
    prior: calib.Calibration,
    warnings: list[str],
) -> FlightPhase | None:
    first, last = cfg.flight_window
    manual = fit_ballistic(
        trajectory.frames[first : last + 1],
        trajectory.xy[first : last + 1],
        fps,
        G * prior.px_per_metre,
    )
    if manual is None:
        warnings.append(
            f"the requested window {first}-{last} contains too few measured frames to fit"
        )
        return None
    return FlightPhase(
        fit=manual,
        body_scale_px=body_scale_px,
        implied_stature_m=body_scale_px / prior.px_per_metre,
        searched_windows=0,
        n_inliers=manual.n_points,
        n_frames=last - first + 1,
        entry_source="manual",
    )


def _set_position(subject: Track, takeoff_frame: int):
    """Keypoints from the last stable frame before takeoff, for the knee angle."""
    candidates = [f for f in subject.observations if f < takeoff_frame]
    if not candidates:
        return None
    det = subject.observations[max(candidates)]
    return det.xy, det.conf


def refit_window(analysis: DiveAnalysis, start_frame: int, end_frame: int) -> BallisticFit | None:
    """Refit the ballistic model to a manually chosen window.

    Automatic phase detection is right most of the time, not all of the time.
    This is the escape hatch for the rest.
    """
    window = slice(start_frame, end_frame + 1)
    return fit_ballistic(
        analysis.trajectory.frames[window],
        analysis.trajectory.xy[window],
        analysis.fps,
    )
