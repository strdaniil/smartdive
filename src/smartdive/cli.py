"""Command line interface."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from smartdive import __version__
from smartdive.analysis import AnalysisConfig, analyze
from smartdive.backends import BACKENDS
from smartdive.keypoints import LANDMARKS
from smartdive.report import plot_diagnostics, write_csv, write_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="smartdive",
        description=(
            "Measure a swim start from video: automatic athlete selection, "
            "automatic flight-phase detection, and metrics in SI units."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"smartdive {__version__}")

    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("analyze", help="analyse one video")
    run.add_argument("video", type=Path, help="input video")
    run.add_argument(
        "-o",
        "--outdir",
        type=Path,
        default=Path("output"),
        help="directory for the report, CSV, figure and overlay",
    )
    run.add_argument("--name", default=None, help="basename for outputs (default: video stem)")
    run.add_argument(
        "--landmark",
        default="com",
        choices=LANDMARKS,
        help="what to track; 'com' is the whole-body centre of mass",
    )
    run.add_argument("--backend", default="yolo", choices=BACKENDS, help="pose estimator")
    run.add_argument("--weights", default=None, help="backend checkpoint (yolo only)")
    run.add_argument(
        "--imgsz",
        type=int,
        default=None,
        help="inference resolution (yolo only); higher is slower and more precise",
    )
    run.add_argument("--device", default=None, help="torch device, e.g. cpu or 0")
    run.add_argument(
        "--flight-window",
        default=None,
        metavar="TAKEOFF:ENTRY",
        help="override automatic phase detection with explicit frame numbers",
    )
    run.add_argument(
        "--min-conf",
        type=float,
        default=0.3,
        dest="min_conf",
        help="keypoints below this confidence are treated as missing",
    )

    scale = run.add_argument_group(
        "calibration",
        "By default the scale is recovered from gravity during free flight. "
        "Supplying a reference or a stature overrides that.",
    )
    scale.add_argument("--reference-px", type=float, default=None, help="known length, in pixels")
    scale.add_argument("--reference-m", type=float, default=None, help="the same length, in metres")
    scale.add_argument("--stature-m", type=float, default=None, help="athlete height in metres")

    out = run.add_argument_group("outputs")
    out.add_argument("--overlay", action="store_true", help="render an annotated video")
    out.add_argument("--no-plot", action="store_true", help="skip the diagnostic figure")
    out.add_argument("--no-csv", action="store_true", help="skip the per-frame CSV")
    out.add_argument(
        "--bootstrap",
        type=int,
        default=400,
        help="uncertainty resamples (0 disables)",
    )
    out.add_argument("--quiet", action="store_true", help="print JSON only")
    out.add_argument("--progress", action="store_true", help="print per-frame detection progress")

    info = sub.add_parser("info", help="print video metadata")
    info.add_argument("video", type=Path)

    return parser


def _config_from_args(args: argparse.Namespace) -> AnalysisConfig:
    backend_options: dict[str, object] = {}
    if args.backend == "yolo":
        if args.weights:
            backend_options["weights"] = args.weights
        if args.imgsz:
            backend_options["imgsz"] = args.imgsz
        if args.device:
            backend_options["device"] = args.device

    if bool(args.reference_px) != bool(args.reference_m):
        raise SystemExit("--reference-px and --reference-m must be given together")

    window = None
    if args.flight_window:
        try:
            first, last = (int(part) for part in args.flight_window.split(":"))
        except ValueError:
            raise SystemExit("--flight-window must look like TAKEOFF:ENTRY, e.g. 14:26") from None
        if last <= first:
            raise SystemExit("--flight-window entry frame must be after the takeoff frame")
        window = (first, last)

    return AnalysisConfig(
        landmark=args.landmark,
        backend=args.backend,
        backend_options=backend_options,
        min_keypoint_conf=args.min_conf,
        reference_px=args.reference_px,
        reference_m=args.reference_m,
        stature_m=args.stature_m,
        flight_window=window,
        bootstrap=args.bootstrap,
    )


def _run_analyze(args: argparse.Namespace) -> int:
    config = _config_from_args(args)
    name = args.name or args.video.stem
    outdir: Path = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    result = analyze(args.video, config, progress=args.progress)

    report = write_json(result, outdir / f"{name}.json")
    written = [report]
    if not args.no_csv:
        written.append(write_csv(result, outdir / f"{name}_trajectory.csv"))
    if not args.no_plot:
        written.append(plot_diagnostics(result, outdir / f"{name}_diagnostics.png"))
    if args.overlay:
        from smartdive.render import render_overlay

        written.append(render_overlay(result, outdir / f"{name}_overlay.mp4"))

    if args.quiet:
        print(json.dumps(result.to_dict(), indent=2))
        return 0

    print(f"\nSmartDive {__version__}  |  {args.video.name}")
    print(
        f"{result.video.width}x{result.video.height} @ {result.video.fps:g} fps, "
        f"{result.video.frame_count} frames  |  analysed in {result.elapsed_s:.1f} s"
    )
    print(
        f"subject: track {result.subject.track_id} of {len(result.tracks)} "
        f"({len(result.subject.observations)} detections, "
        f"{result.subject_ranking[0].displacement_bodies:.1f} body lengths travelled)"
    )

    quality = result.quality()
    print(
        f"signal: {quality['coverage']:.0%} coverage, "
        f"jitter {quality['jitter_raw_px']:.1f} -> {quality['jitter_conditioned_px']:.1f} px, "
        f"{int(quality['outliers_rejected'])} spikes rejected, "
        f"{int(quality['gaps_filled'])} frames interpolated"
    )

    if result.metrics is None:
        print("\nNo ballistic flight phase was found in this clip.")
    else:
        print()
        print(result.metrics.format_table())
        if result.uncertainty:
            lo, hi = result.uncertainty.get("angle_deg", (float("nan"),) * 2)
            print(f"\n95% CI on takeoff angle: {lo:+.1f} to {hi:+.1f} deg")

    for warning in result.warnings:
        print(f"  ! {warning}")

    print("\nwrote:")
    for path in written:
        print(f"  {path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "info":
        from smartdive.video import probe

        print(json.dumps(probe(args.video).to_dict(), indent=2))
        return 0

    return _run_analyze(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
