"""Exports: JSON, per-frame CSV, and diagnostic figures."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from smartdive.analysis import DiveAnalysis


def write_json(analysis: DiveAnalysis, path: str | Path) -> Path:
    """Machine-readable summary, including provenance and warnings."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(analysis.to_dict(), indent=2), encoding="utf-8")
    return path


def write_csv(analysis: DiveAnalysis, path: str | Path) -> Path:
    """Per-frame trajectory, in both pixels and metres where calibrated."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    traj = analysis.trajectory
    velocity = traj.velocity()
    ppm = analysis.calibration.px_per_metre if analysis.calibration else None
    fit = analysis.fit

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "frame",
                "time_s",
                "raw_x_px",
                "raw_y_px",
                "x_px",
                "y_px",
                "vx_px_s",
                "vy_px_s",
                "x_m",
                "y_m",
                "speed_m_s",
                "measured",
                "outlier",
                "interpolated",
                "phase",
            ]
        )
        for i, frame in enumerate(traj.frames):
            t = frame / traj.fps
            x, y = traj.xy[i]
            vx, vy = velocity[i]
            if ppm:
                xm, ym = x / ppm, y / ppm
                speed = float(np.hypot(vx, vy) / ppm) if np.isfinite(vx) else ""
            else:
                xm = ym = speed = ""
            if fit is None:
                phase = "unknown"
            elif frame < fit.start_frame:
                phase = "block"
            elif frame <= fit.end_frame:
                phase = "flight"
            else:
                phase = "water"
            writer.writerow(
                [
                    int(frame),
                    round(t, 4),
                    _num(traj.raw[i, 0]),
                    _num(traj.raw[i, 1]),
                    _num(x),
                    _num(y),
                    _num(vx),
                    _num(vy),
                    _num(xm),
                    _num(ym),
                    _num(speed),
                    int(traj.valid[i]),
                    int(traj.outliers[i]),
                    int(traj.interpolated[i]),
                    phase,
                ]
            )
    return path


def _num(value: object) -> object:
    if value == "" or value is None:
        return ""
    v = float(value)
    return "" if np.isnan(v) else round(v, 3)


def plot_diagnostics(analysis: DiveAnalysis, path: str | Path) -> Path:
    """Four-panel diagnostic figure: path, y(t), x(t), and fit residuals."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    traj = analysis.trajectory
    fit = analysis.fit
    t = traj.frames / traj.fps

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle(
        f"SmartDive diagnostics - {analysis.video.path.name} "
        f"({analysis.config.landmark}, {analysis.config.backend})",
        fontsize=13,
    )

    ax = axes[0][0]
    ax.plot(traj.raw[:, 0], traj.raw[:, 1], ".", ms=4, color="#bbbbbb", label="raw")
    ax.plot(traj.xy[:, 0], traj.xy[:, 1], "-", lw=1.8, color="#2f9e44", label="conditioned")
    if fit is not None:
        model = fit.sample(200)
        ax.plot(model[:, 0], model[:, 1], "--", lw=2, color="#f08c00", label="ballistic fit")
    ax.invert_yaxis()
    ax.set_xlabel("x (px)")
    ax.set_ylabel("y (px)")
    ax.set_title("path in image space")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    ax = axes[0][1]
    ax.plot(t, traj.raw[:, 1], ".", ms=4, color="#bbbbbb")
    ax.plot(t, traj.xy[:, 1], "-", lw=1.8, color="#2f9e44")
    if fit is not None:
        ft = np.linspace(0, fit.duration, 100)
        ax.plot(
            fit.start_frame / traj.fps + ft,
            fit.position(ft)[:, 1],
            "--",
            lw=2,
            color="#f08c00",
        )
        ax.axvspan(
            fit.start_frame / traj.fps,
            fit.end_frame / traj.fps,
            color="#f08c00",
            alpha=0.10,
            label="flight",
        )
        ax.legend(fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("time (s)")
    ax.set_ylabel("y (px, down)")
    ax.set_title("vertical position: parabolic in free flight")
    ax.grid(alpha=0.25)

    ax = axes[1][0]
    ax.plot(t, traj.raw[:, 0], ".", ms=4, color="#bbbbbb")
    ax.plot(t, traj.xy[:, 0], "-", lw=1.8, color="#2f9e44")
    if fit is not None:
        ft = np.linspace(0, fit.duration, 100)
        ax.plot(
            fit.start_frame / traj.fps + ft,
            fit.position(ft)[:, 0],
            "--",
            lw=2,
            color="#f08c00",
        )
        ax.axvspan(
            fit.start_frame / traj.fps,
            fit.end_frame / traj.fps,
            color="#f08c00",
            alpha=0.10,
        )
    ax.set_xlabel("time (s)")
    ax.set_ylabel("x (px)")
    ax.set_title("horizontal position: linear in free flight")
    ax.grid(alpha=0.25)

    ax = axes[1][1]
    if fit is not None:
        window = slice(fit.start_frame, fit.end_frame + 1)
        wt = (traj.frames[window] - fit.start_frame) / traj.fps
        observed = traj.xy[window]
        model = fit.position(wt)
        ax.axhline(0, color="#888888", lw=0.8)
        ax.plot(wt, observed[:, 0] - model[:, 0], "o-", ms=4, label="x residual")
        ax.plot(wt, observed[:, 1] - model[:, 1], "s-", ms=4, label="y residual")
        ax.set_title(
            f"fit residuals (RMS {fit.residual_rms:.1f} px, "
            f"{fit.residual_rms / analysis.body_scale_px:.1%} of body length)"
        )
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "no flight phase detected", ha="center", va="center")
        ax.set_title("fit residuals")
    ax.set_xlabel("time since takeoff (s)")
    ax.set_ylabel("residual (px)")
    ax.grid(alpha=0.25)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path
