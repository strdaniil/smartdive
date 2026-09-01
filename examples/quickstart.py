"""Analyse one clip and print everything SmartDive knows about it.

python examples/quickstart.py videos/adive.mp4
"""

from __future__ import annotations

import sys
from pathlib import Path

from smartdive import AnalysisConfig, analyze
from smartdive.report import plot_diagnostics, write_csv, write_json


def main(video: str) -> int:
    # Caching memoises the pose detections, the camera-motion estimate and the
    # refinement pass, which makes re-running this on the same clip instant.
    # Inference dominates the runtime and none of it depends on the analysis.
    config = AnalysisConfig(bootstrap=400)
    result = analyze(video, config, cache=".cache")

    rate = f"{result.video.fps:g} fps"
    if result.time_base.corrected:
        # The container frame rate was lying: this is slow-motion footage and
        # every time and speed would otherwise be wrong by the same factor.
        rate += f" container / {result.fps:g} fps capture"
    print(
        f"{result.video.path.name}: {result.video.frame_count} frames "
        f"@ {rate}, analysed in {result.elapsed_s:.1f} s"
    )

    # Which person did it decide to measure, and why?
    print(f"\nchose track {result.subject.track_id} of {len(result.tracks)}:")
    for score in result.subject_ranking[:3]:
        print(
            f"  track {score.track_id:2d}  "
            f"{score.peak_speed_bodies_s:5.2f} body lengths/s peak  "
            f"{score.displacement_bodies:5.2f} travelled  "
            f"{score.n_observations:3d} detections  "
            f"confidence {score.mean_score:.2f}"
        )

    # How good is the signal underneath the numbers?
    quality = result.quality()
    print(
        f"\ncoverage {quality['coverage']:.0%}, "
        f"jitter {quality['jitter_raw_px']:.1f} -> "
        f"{quality['jitter_conditioned_px']:.1f} px, "
        f"{int(quality['subject_switches'])} identity errors, "
        f"{quality['camera_drift_px']:.0f} px of camera motion removed"
    )
    if result.refinement is not None:
        print(
            f"second detection pass: {result.refinement.frames_recovered} frames "
            f"recovered, {result.refinement.frames_improved} improved"
        )
    print(
        f"scale: {result.body.stature_px:.0f} px of stature from "
        f"{len(result.body.per_segment)} body segments "
        f"(spread {result.body.dispersion:.0%})"
    )

    if result.metrics is None:
        print("\nno flight phase found, try --flight-window, or a longer clip")
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
