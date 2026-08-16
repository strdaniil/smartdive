"""SmartDive - physics-grounded swim start analysis from ordinary video.

Typical use::

    from smartdive import analyze

    result = analyze("videos/start.mp4")
    print(result.metrics.format_table())

See :func:`smartdive.analysis.analyze` for the options.
"""

from __future__ import annotations

from smartdive.analysis import AnalysisConfig, DiveAnalysis, analyze
from smartdive.calibration import Calibration
from smartdive.metrics import DiveMetrics
from smartdive.physics import BallisticFit, FlightPhase
from smartdive.signal import ConditionedTrajectory, SmoothingConfig
from smartdive.tracking import Track, TrackerConfig

__version__ = "3.0.0"

__all__ = [
    "AnalysisConfig",
    "BallisticFit",
    "Calibration",
    "ConditionedTrajectory",
    "DiveAnalysis",
    "DiveMetrics",
    "FlightPhase",
    "SmoothingConfig",
    "Track",
    "TrackerConfig",
    "__version__",
    "analyze",
]
