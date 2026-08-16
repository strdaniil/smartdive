# Changelog

## 3.0.0

Complete rewrite. The previous versions were a set of scripts that shared
constants by copy-paste across two repositories; this is a package with an API,
a CLI, tests and a benchmark against the old behaviour.

### Fixed

- **The subject was chosen per frame by detector confidence**, which locks onto
  whichever bystander is best lit the moment the athlete blurs. On the sample
  footage this picked the diver in as little as 1% of frames, and reported a
  dive of 3 px. Subject selection is now a whole-clip decision based on tracked
  motion.
- **"Flight time" was the span between the first and last frame containing any
  detection** -- i.e. the length of the clip. Flight is now detected.
- **A degree-3 polynomial was fit to x(t) and y(t) over the whole clip.** A
  cubic has no physical meaning for a body in free flight and mixes the block,
  flight and water phases into one curve. Replaced with the projectile model.
- **Results were reported in pixels**, so no two clips were comparable. Results
  are now in SI units with a stated calibration source.
- **The orientation flag was applied inconsistently** across `process_video.py`,
  `analyze_json.py`, `overlay_video.py` and `compute_metrics.py`, so whether a
  trajectory came out mirrored depended on which scripts you ran and in what
  order. Nothing is mirrored anywhere now; the fitted signs carry direction.
- Missing keypoints produced NaN with no gap filling, outlier rejection or
  smoothing. There is now a conditioning stage, and it reports what it did.
- The overlay redrew the entire trail segment-by-segment on every frame, making
  rendering quadratic in clip length.

### Added

- Whole-body centre of mass from anthropometric segment parameters.
- Automatic takeoff detection from the ballistic residual, and automatic water
  entry detection from the collapse in visible body mass.
- Gravity-based scale calibration, with an honest agreement check against a
  body-scale prior and a documented account of when it is trustworthy.
- Bootstrap confidence intervals on the fitted flight parameters.
- Pluggable pose backends (YOLO, MediaPipe) behind one COCO-17 schema.
- JSON, per-frame CSV, diagnostic figures and annotated video output.
- `smartdive.legacy`, a faithful re-implementation of the old pipeline, so the
  comparison in the README is reproducible rather than asserted.
