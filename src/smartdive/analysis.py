"""End-to-end analysis: video in, calibrated dive metrics out."""

from __future__ import annotations

import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from smartdive import calibration as calib
from smartdive.backends import FrameDetections, PoseBackend, get_backend
from smartdive.com import visible_mass_fraction
from smartdive.keypoints import body_scale, resolve_landmark
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
from smartdive.signal import (
    ConditionedTrajectory,
    SmoothingConfig,
    condition,
    jitter,
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

    #: Manual calibration overrides. ``reference_px``/``reference_m`` together
    #: pin the scale to a measured object; ``stature_m`` uses athlete height.
    reference_px: float | None = None
    reference_m: float | None = None
    stature_m: float | None = None

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
            "reference_px": self.reference_px,
            "reference_m": self.reference_m,
            "stature_m": self.stature_m,
            "flight_window": list(self.flight_window) if self.flight_window else None,
        }


@dataclass
class DiveAnalysis:
    """The complete result of analysing one clip."""

    video: VideoInfo
    config: AnalysisConfig
    detections: list[FrameDetections]
    tracks: list[Track]
    subject: Track
    subject_ranking: list[SubjectScore]
    trajectory: ConditionedTrajectory
    body_scale_px: float
    extended_body_scale_px: float
    airborne: np.ndarray
    flight: FlightPhase | None
    calibration: calib.Calibration | None
    metrics: DiveMetrics | None
    uncertainty: dict[str, tuple[float, float]]
    warnings: list[str]
    elapsed_s: float
    backend_info: dict[str, object]

    @property
    def fit(self) -> BallisticFit | None:
        return self.flight.fit if self.flight else None

    def quality(self) -> dict[str, float]:
        """Signal-quality numbers, useful for regression testing a change."""
        return {
            "coverage": self.trajectory.coverage,
            "jitter_raw_px": jitter(self.trajectory.raw),
            "jitter_conditioned_px": jitter(self.trajectory.xy),
            "outliers_rejected": float(self.trajectory.outliers.sum()),
            "gaps_filled": float(self.trajectory.interpolated.sum()),
            "subject_switches": float(subject_switches(self.trajectory.raw, self.body_scale_px)),
        }

    def to_dict(self) -> dict[str, object]:
        return {
            "video": self.video.to_dict(),
            "config": self.config.to_dict(),
            "backend": self.backend_info,
            "elapsed_s": round(self.elapsed_s, 2),
            "subject": {
                "track_id": self.subject.track_id,
                "frames": [int(self.subject.frames[0]), int(self.subject.frames[-1])],
                "n_observations": len(self.subject.observations),
                "ranking": [s.to_dict() for s in self.subject_ranking],
            },
            "body_scale_px": round(self.body_scale_px, 1),
            "extended_body_scale_px": round(self.extended_body_scale_px, 1),
            "quality": {
                k: (None if np.isnan(v) else round(v, 4)) for k, v in self.quality().items()
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


def _landmark_series(
    subject: Track, n_frames: int, landmark: str, min_conf: float
) -> tuple[np.ndarray, np.ndarray]:
    """Dense ``(frames, xy)`` arrays for the subject's landmark over the clip."""
    frames = np.arange(n_frames)
    xy = np.full((n_frames, 2), np.nan)
    for f, det in subject.observations.items():
        if 0 <= f < n_frames:
            xy[f] = resolve_landmark(det.xy, det.conf, landmark, min_conf)
    return frames, xy


def _airborne_mask(subject: Track, n_frames: int, min_conf: float, min_mass: float) -> np.ndarray:
    """Frames in which enough of the athlete is visible to be out of the water.

    A pose detector cannot label the joints it cannot see, so once a swimmer is
    part-submerged the visible mass fraction falls off a cliff -- typically from
    1.0 to about 0.35, since the trunk and one arm stay above the surface while
    the legs vanish. That step is a far more direct signal of water entry than
    anything recoverable from the trajectory itself.
    """
    # Frames with no detection at all are left True: not seeing the athlete is
    # not evidence that they are underwater, and treating it as such would end
    # the flight at the first dropped frame. Whether such a frame is usable is
    # decided separately, by whether the trajectory has a value there.
    mask = np.ones(n_frames, dtype=bool)
    for f, det in subject.observations.items():
        if 0 <= f < n_frames:
            mask[f] = visible_mass_fraction(det.conf, min_conf) >= min_mass
    return mask


def _body_scales(subject: Track, min_conf: float) -> tuple[float, float]:
    """Median and extended pixel body scale for the subject.

    The median is a stable length unit for thresholds. The extended scale -- the
    90th percentile -- is what calibration wants, because it corresponds to the
    athlete stretched out in flight rather than folded up on the block, and it
    is the pose whose real-world length we can actually put a number on.
    """
    scales = [body_scale(det.xy, det.conf, min_conf) for det in subject.observations.values()]
    scales = [s for s in scales if np.isfinite(s)]
    if not scales:
        return float("nan"), float("nan")
    return float(np.median(scales)), float(np.percentile(scales, 90))


def _set_position(subject: Track, takeoff_frame: int, min_conf: float):
    """Keypoints from the last stable frame before takeoff, for the knee angle."""
    candidates = [f for f in subject.observations if f < takeoff_frame]
    if not candidates:
        return None
    det = subject.observations[max(candidates)]
    return det.xy, det.conf


def _cache_path(cache: Path, video: Path, config: AnalysisConfig) -> Path:
    """Cache key covers everything that changes what the detector produces."""
    options = "_".join(f"{k}{v}" for k, v in sorted(config.backend_options.items()))
    return Path(cache) / f"{video.stem}__{config.backend}__{options or 'default'}.pkl"


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
        Directory in which to memoise raw detections. Inference dominates the
        runtime, so caching it makes iterating on everything downstream --
        conditioning, phase detection, calibration -- effectively free.
    """
    cfg = config or AnalysisConfig()
    started = time.perf_counter()
    warnings: list[str] = []

    info = probe(video_path)
    detections: list[FrameDetections] | None = None
    backend_info: dict[str, object] = {"backend": cfg.backend, **cfg.backend_options}

    cached_at = None
    if cache is not None:
        cached_at = _cache_path(Path(cache), Path(video_path), cfg)
        if cached_at.exists():
            payload = pickle.loads(cached_at.read_bytes())
            detections = payload["detections"]
            backend_info = payload["backend_info"]

    if detections is None:
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
            cached_at.write_bytes(
                pickle.dumps({"detections": detections, "backend_info": backend_info})
            )

    n_frames = len(detections)
    tracks = track_all(detections, cfg.tracker)
    if not tracks:
        raise ValueError(f"no people detected in {info.path}")

    subject, ranking = select_subject(tracks, cfg.tracker)
    if len(ranking) > 1 and ranking[0].total - ranking[1].total < 0.25:
        warnings.append(
            "subject selection was close between tracks "
            f"{ranking[0].track_id} and {ranking[1].track_id}; check the overlay"
        )

    raw_frames, raw_xy = _landmark_series(subject, n_frames, cfg.landmark, cfg.min_keypoint_conf)
    trajectory = condition(raw_frames, raw_xy, info.fps, cfg.smoothing)
    scale_px, extended_px = _body_scales(subject, cfg.min_keypoint_conf)

    if trajectory.coverage < 0.3:
        warnings.append(
            f"only {trajectory.coverage:.0%} of frames yielded a {cfg.landmark} "
            "measurement; try --landmark hips or a larger --imgsz"
        )

    airborne = _airborne_mask(subject, n_frames, cfg.min_keypoint_conf, cfg.airborne_min_mass)
    # Fix the vertical acceleration from an independent scale prior before
    # searching for the flight. See fit_ballistic() for why a free fit cannot
    # measure it reliably from a third of a second of 30 fps video.
    prior = calib.resolve(
        None,
        extended_px,
        stature_m=cfg.stature_m,
        reference_px=cfg.reference_px,
        reference_m=cfg.reference_m,
    )[0]
    if cfg.flight_window is not None:
        first, last = cfg.flight_window
        manual = fit_ballistic(
            trajectory.frames[first : last + 1],
            trajectory.xy[first : last + 1],
            info.fps,
            G * prior.px_per_metre,
        )
        flight = (
            None
            if manual is None
            else FlightPhase(
                fit=manual,
                body_scale_px=scale_px,
                implied_stature_m=scale_px / prior.px_per_metre,
                searched_windows=0,
                n_inliers=manual.n_points,
                n_frames=last - first + 1,
            )
        )
        if flight is None:
            warnings.append(
                f"the requested window {first}-{last} contains too few measured frames to fit"
            )
    else:
        flight = detect_flight(
            trajectory.frames,
            trajectory.xy,
            info.fps,
            scale_px,
            cfg.flight,
            airborne,
            trajectory.noise_sigma,
            ay_prior=G * prior.px_per_metre,
        )

    calibration = None
    metrics = None
    uncertainty: dict[str, tuple[float, float]] = {}

    if flight is None:
        warnings.append(
            "no ballistic flight phase found: the clip may not contain a dive, "
            "the athlete may leave frame, or the camera may be panning"
        )
    else:
        # Re-estimate the scale from a free fit over the window that has now
        # been identified, and take it only if it corroborates the prior.
        window_slice = slice(flight.fit.start_frame, flight.fit.end_frame + 1)
        free_fit = fit_ballistic(
            trajectory.frames[window_slice], trajectory.xy[window_slice], info.fps
        )
        calibration, notes = calib.resolve(
            free_fit,
            extended_px,
            stature_m=cfg.stature_m,
            reference_px=cfg.reference_px,
            reference_m=cfg.reference_m,
        )
        warnings.extend(notes)
        if calibration.method == "stature_assumed":
            warnings.append(
                "absolute distances assume a "
                f"{calib.DEFAULT_STATURE_M:.2f} m athlete; pass --stature-m or "
                "--reference-px/--reference-m to make them exact. Times and "
                "angles are unaffected."
            )

        window = slice(flight.fit.start_frame, flight.fit.end_frame + 1)
        knee_angle = None
        set_pose = _set_position(subject, flight.fit.start_frame, cfg.min_keypoint_conf)
        if set_pose is not None:
            angle, _side = front_knee_angle(*set_pose, cfg.min_keypoint_conf)
            if np.isfinite(angle):
                knee_angle = angle
                warnings.append("set position: " + knee_angle_feedback(angle))

        block_time = None
        first_frame = int(subject.frames[0])
        if flight.fit.start_frame > first_frame:
            block_time = (flight.fit.start_frame - first_frame) / info.fps

        metrics = compute_metrics(
            flight.fit,
            calibration,
            trajectory_px=trajectory.xy[window],
            block_time_s=block_time,
            set_knee_angle_deg=knee_angle,
        )

        if cfg.bootstrap:
            # Resample under the same model the metrics were computed with.
            uncertainty = bootstrap_uncertainty(
                trajectory.frames[window],
                trajectory.xy[window],
                info.fps,
                n_resamples=cfg.bootstrap,
                ay=flight.fit.ay,
            )

    return DiveAnalysis(
        video=info,
        config=cfg,
        detections=detections,
        tracks=tracks,
        subject=subject,
        subject_ranking=ranking,
        trajectory=trajectory,
        body_scale_px=scale_px,
        extended_body_scale_px=extended_px,
        airborne=airborne,
        flight=flight,
        calibration=calibration,
        metrics=metrics,
        uncertainty=uncertainty,
        warnings=warnings,
        elapsed_s=time.perf_counter() - started,
        backend_info=backend_info,
    )


def refit_window(analysis: DiveAnalysis, start_frame: int, end_frame: int) -> BallisticFit | None:
    """Refit the ballistic model to a manually chosen window.

    Automatic phase detection is right most of the time, not all of the time.
    This is the escape hatch for the rest.
    """
    window = slice(start_frame, end_frame + 1)
    return fit_ballistic(
        analysis.trajectory.frames[window],
        analysis.trajectory.xy[window],
        analysis.video.fps,
    )
