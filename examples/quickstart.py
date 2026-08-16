"""Analyse one clip and print everything SmartDive knows about it.

python examples/quickstart.py videos/adive.mp4
"""

from __future__ import annotations

import sys
from pathlib import Path

from smartdive import AnalysisConfig, analyze
from smartdive.report import plot_diagnostics, write_csv, write_json


def main(video: str) -> int:
    # Caching the detections makes re-running this on the same clip instant,
    # which matters because inference dominates the runtime.
    config = AnalysisConfig(landmark="hips", bootstrap=400)
    result = analyze(video, config, cache=".cache")

    print(
        f"{result.video.path.name}: {result.video.frame_count} frames "
        f"@ {result.video.fps:g} fps, analysed in {result.elapsed_s:.1f} s"
    )

    # Which person did it decide to measure, and why?
    print(f"\nchose track {result.subject.track_id} of {len(result.tracks)}:")
    for score in result.subject_ranking[:3]:
        print(
            f"  track {score.track_id:2d}  "
            f"{score.displacement_bodies:5.2f} body lengths  "
            f"{score.n_observations:3d} detections  "
            f"confidence {score.mean_score:.2f}"
        )

    # How good is the signal underneath the numbers?
    quality = result.quality()
    print(
        f"\ncoverage {quality['coverage']:.0%}, "
        f"jitter {quality['jitter_raw_px']:.1f} -> "
        f"{quality['jitter_conditioned_px']:.1f} px, "
        f"{int(quality['subject_switches'])} identity errors"
    )

    if result.metrics is None:
        print("\nno flight phase found -- try --flight-window, or a longer clip")
    else:
        print()
        print(result.metrics.format_table())
        print("\n95% confidence intervals:")
        for name, (low, high) in result.uncertainty.items():
            print(f"  {name:10s} {low:9.2f} to {high:9.2f}")

    for warning in result.warnings:
        print(f"\n! {warning}")

    out = Path("output")
    write_json(result, out / "quickstart.json")
    write_csv(result, out / "quickstart_trajectory.csv")
    plot_diagnostics(result, out / "quickstart_diagnostics.png")
    print(f"\nwrote reports to {out}/")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    raise SystemExit(main(sys.argv[1]))
