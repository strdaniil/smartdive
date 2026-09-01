"""Run the original SmartDive pipeline and the current one over the same clips.

Both pipelines consume the *same* pose detections, so nothing in the comparison
turns on one of them having had a better detector. What is compared is what each
does with the detections: who it decides the athlete is, how it conditions the
trajectory, which frames it thinks were airborne, and what it reports.

    python benchmarks/run_benchmark.py videos/*.mp4 -o benchmarks/results

Outputs a Markdown table, a JSON dump of every number, and per-clip comparison
figures.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from smartdive.analysis import AnalysisConfig, analyze, subject_switches
from smartdive.backends import get_backend
from smartdive.legacy import legacy_metrics, legacy_trajectory
from smartdive.signal import jitter


@dataclass
class Comparison:
    """Old versus new, for one clip."""

    clip: str
    frames: int
    fps: float
    # legacy
    old_coverage: float
    old_jitter: float
    old_switches: int
    old_travel_bodies: float
    subject_agreement: float
    old_flight_time: float | None
    old_distance_px: float | None
    old_fit_rms: float | None
    # current
    capture_fps: float
    slow_motion: bool
    new_coverage: float
    new_travel_bodies: float
    new_jitter_raw: float
    new_jitter: float
    new_switches: int
    new_flight: tuple[int, int] | None
    new_flight_time: float | None
    new_speed_ms: float | None
    new_angle_deg: float | None
    new_distance_m: float | None
    new_fit_rms: float | None
    new_r2: float | None
    seconds: float
    warnings: list[str]

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


def travel_bodies(xy: np.ndarray, body_scale_px: float) -> float:
    """Total straight-line displacement of a trajectory, in body lengths.

    This is the single number that exposes a subject-selection failure. A diver
    covers two to three body lengths during a start; a spectator on the deck
    covers essentially none. A pipeline that reports a dive of 0.01 body lengths
    has not measured a bad dive, it has measured the wrong person.
    """
    good = xy[~np.isnan(xy[:, 0])]
    if len(good) < 2 or not np.isfinite(body_scale_px) or body_scale_px <= 0:
        return float("nan")
    return float(np.linalg.norm(good.max(axis=0) - good.min(axis=0)) / body_scale_px)


def subject_agreement(result, min_conf: float) -> float:
    """Fraction of frames where per-frame argmax picks the tracked athlete.

    The original pipeline chose the subject independently in every frame, by
    detector confidence. This measures how often that choice was the person the
    tracker follows through the whole clip, i.e. how often the old approach
    was looking at the diver at all.
    """
    agree = total = 0
    for frame in result.detections:
        if not frame.people:
            continue
        subject_det = result.subject.observations.get(frame.index)
        if subject_det is None:
            continue
        total += 1
        best = max(frame.people, key=lambda p: p.score)
        if best is subject_det:
            agree += 1
    return agree / total if total else float("nan")


def compare(video: Path, config: AnalysisConfig, backend, cache: Path | None = None) -> Comparison:
    result = analyze(video, config, backend=backend, cache=cache)

    # --- legacy pipeline, on the identical detections ---------------------
    legacy_xy = legacy_trajectory(result.detections, landmark="hips")
    legacy = legacy_metrics(legacy_xy, result.video.fps)
    scale_px = result.body_scale_px

    quality = result.quality()
    m = result.metrics

    return (
        Comparison(
            clip=video.name,
            frames=result.video.frame_count,
            fps=result.video.fps,
            old_coverage=float(np.mean(~np.isnan(legacy_xy[:, 0]))),
            old_jitter=jitter(legacy_xy),
            old_switches=subject_switches(legacy_xy, scale_px),
            old_travel_bodies=travel_bodies(legacy_xy, scale_px),
            subject_agreement=subject_agreement(result, config.min_keypoint_conf),
            old_flight_time=legacy.flight_time_s if legacy else None,
            old_distance_px=legacy.horizontal_distance_px if legacy else None,
            old_fit_rms=legacy.poly_residual_rms_px if legacy else None,
            capture_fps=result.fps,
            slow_motion=result.time_base.corrected,
            new_coverage=quality["coverage"],
            new_travel_bodies=travel_bodies(result.trajectory.xy, scale_px),
            new_jitter_raw=quality["jitter_raw_px"],
            new_jitter=quality["jitter_conditioned_px"],
            new_switches=int(quality["subject_switches"]),
            new_flight=(m.takeoff_frame, m.entry_frame) if m else None,
            new_flight_time=m.flight_time_s if m else None,
            new_speed_ms=m.takeoff_speed_ms if m else None,
            new_angle_deg=m.takeoff_angle_deg if m else None,
            new_distance_m=m.flight_distance_m if m else None,
            new_fit_rms=m.fit_residual_rms_px if m else None,
            new_r2=m.fit_r_squared if m else None,
            seconds=result.elapsed_s,
            warnings=result.warnings,
        ),
        result,
        legacy_xy,
    )


def plot_clip(result, legacy_xy: np.ndarray, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    traj = result.trajectory
    fit = result.fit
    t = traj.frames / traj.fps

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    fig.suptitle(f"{result.video.path.name} - original pipeline vs SmartDive 3", fontsize=13)

    ax = axes[0]
    ax.plot(legacy_xy[:, 0], legacy_xy[:, 1], "o-", ms=3, lw=1, color="#c92a2a", label="original")
    ax.plot(traj.xy[:, 0], traj.xy[:, 1], "-", lw=2, color="#2f9e44", label="SmartDive 3")
    if fit is not None:
        s = fit.sample(150)
        ax.plot(s[:, 0], s[:, 1], "--", lw=2, color="#f08c00", label="ballistic model")
    ax.invert_yaxis()
    ax.set_title("tracked path")
    ax.set_xlabel("x (px)")
    ax.set_ylabel("y (px)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    ax = axes[1]
    ax.plot(t, legacy_xy[:, 0], "o-", ms=3, lw=1, color="#c92a2a", label="original")
    ax.plot(t, traj.xy[:, 0], "-", lw=2, color="#2f9e44", label="SmartDive 3")
    if fit is not None:
        ax.axvspan(
            fit.start_frame / traj.fps,
            fit.end_frame / traj.fps,
            color="#f08c00",
            alpha=0.15,
            label="detected flight",
        )
    ax.set_title("horizontal position over time")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("x (px)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    ax = axes[2]
    steps_old = np.linalg.norm(np.diff(legacy_xy, axis=0), axis=1)
    steps_new = np.linalg.norm(np.diff(traj.xy, axis=0), axis=1)
    ax.plot(t[1:], steps_old, "o-", ms=3, lw=1, color="#c92a2a", label="original")
    ax.plot(t[1:], steps_new, "-", lw=2, color="#2f9e44", label="SmartDive 3")
    ax.axhline(
        result.body_scale_px,
        color="#495057",
        ls=":",
        label="one body length (impossible in 1 frame)",
    )
    ax.set_title("frame-to-frame displacement")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("px per frame")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def markdown_table(rows: list[Comparison]) -> str:
    def fmt(value, spec="{:.2f}", dash="--"):
        return dash if value is None else spec.format(value)

    lines = [
        "| clip | | looking at the diver | path travelled | jitter | flight phase | result |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        new_window = (
            "**not found**"
            if r.new_flight is None
            else f"frames {r.new_flight[0]}-{r.new_flight[1]}"
        )
        old_result = (
            "--"
            if r.old_flight_time is None
            else f"{r.old_flight_time:.2f} s, {r.old_distance_px:.0f} px"
        )
        new_result = (
            "**not found**"
            if r.new_flight_time is None
            else f"{r.new_flight_time:.2f} s, {r.new_speed_ms:.1f} m/s, {r.new_distance_m:.1f} m"
        )
        if r.slow_motion:
            new_result += f"<br><sub>at {r.capture_fps:g} fps capture</sub>"
        lines.append(
            f"| **`{r.clip}`**<br><sub>{r.frames} frames</sub> | original "
            f"| {r.subject_agreement:.0%} of frames "
            f"| {fmt(r.old_travel_bodies)} body lengths "
            f"| {fmt(r.old_jitter, '{:.1f}')} px "
            f"| whole clip (0-{r.frames - 1}) | {old_result} |"
        )
        lines.append(
            "| | **SmartDive 3** | by construction "
            f"| **{fmt(r.new_travel_bodies)}** body lengths "
            f"| **{fmt(r.new_jitter, '{:.1f}')}** px "
            f"| {new_window} | {new_result} |"
        )
    return "\n".join(lines)


#: Clips known to be the same dive exported at different playback rates. Their
#: results should agree, which is the only ground truth available here that does
#: not require a tape measure at the poolside.
DUPLICATE_PAIRS: tuple[tuple[str, str], ...] = (("adive.mp4", "adive2.mp4"),)


def duplicate_check(rows: list[Comparison]) -> str:
    """Compare clips that are the same dive at different playback rates.

    There is no measured ground truth for this footage, but there is something
    almost as useful: two exports of the same start, one real time and one four
    times slow. Nothing about the athlete differs between them, so any
    disagreement in the reported metrics is the pipeline's own error, and it
    exercises the slow-motion detection, which has to recover a capture rate of
    120 fps from a container claiming 30, or the two will not agree at all.
    """
    index = {r.clip: r for r in rows}
    lines: list[str] = []
    for a, b in DUPLICATE_PAIRS:
        first, second = index.get(a), index.get(b)
        if not first or not second or not first.new_flight_time or not second.new_flight_time:
            continue
        lines.append(
            f"**Self-consistency: `{a}` vs `{b}`**, the same dive, "
            f"`{b}` exported at four times slow motion. Nothing about the athlete "
            "differs, so every disagreement below is the pipeline's own error."
        )
        lines.append("")
        lines.append(f"| quantity | {a} | {b} | agreement |")
        lines.append("|---|---|---|---|")
        lines.append(
            f"| capture rate recovered (fps) | {first.capture_fps:.0f} | "
            f"{second.capture_fps:.0f} | should differ by 4x |"
        )
        for label, x, y, spec in (
            ("flight time (s)", first.new_flight_time, second.new_flight_time, "{:.3f}"),
            ("takeoff speed (m/s)", first.new_speed_ms, second.new_speed_ms, "{:.2f}"),
            ("flight distance (m)", first.new_distance_m, second.new_distance_m, "{:.2f}"),
        ):
            delta = abs(y - x) / x if x else float("nan")
            lines.append(f"| {label} | {spec.format(x)} | {spec.format(y)} | {delta:.0%} |")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("videos", nargs="+", type=Path)
    parser.add_argument("-o", "--outdir", type=Path, default=Path("benchmarks/results"))
    parser.add_argument(
        "--landmark",
        default="com",
        help="what the current pipeline tracks; the legacy re-implementation "
        "always uses the hip midpoint, because that is what it tracked",
    )
    parser.add_argument("--weights", default="yolo11m-pose.pt")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--no-refine", action="store_true")
    parser.add_argument(
        "--cache",
        type=Path,
        default=None,
        help="directory for cached detections, so reruns skip inference",
    )
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    config = AnalysisConfig(landmark=args.landmark, bootstrap=200, refine=not args.no_refine)
    backend = get_backend("yolo", weights=args.weights, imgsz=args.imgsz, conf=0.15)

    rows: list[Comparison] = []
    started = time.perf_counter()
    try:
        for video in args.videos:
            print(f"[{video.name}] ...", flush=True)
            row, result, legacy_xy = compare(video, config, backend, args.cache)
            rows.append(row)
            plot_clip(result, legacy_xy, args.outdir / f"{video.stem}_comparison.png")
            print(
                f"  old: {row.old_switches} subject jumps, "
                f"reports {row.old_flight_time:.2f} s over the whole clip\n"
                f"  new: {row.new_switches} subject jumps, "
                f"flight {row.new_flight}, {row.new_flight_time or float('nan'):.2f} s",
                flush=True,
            )
    finally:
        backend.close()

    (args.outdir / "benchmark.json").write_text(
        json.dumps([r.to_dict() for r in rows], indent=2), encoding="utf-8"
    )
    table = markdown_table(rows)
    consistency = duplicate_check(rows)
    if consistency:
        table += "\n\n" + consistency
    (args.outdir / "benchmark.md").write_text(table + "\n", encoding="utf-8")
    print("\n" + table)
    print(f"\ntotal {time.perf_counter() - started:.0f} s -> {args.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
