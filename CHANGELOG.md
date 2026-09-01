# Changelog

## 3.1.0

The 3.0 rewrite fixed what the *original* pipeline got wrong. This release
fixes what 3.0 got wrong, which turned out to be enough to leave three of the
five sample clips with no result at all and the other two with a flight window
clipped at both ends.

### Fixed

- **The camera was assumed to be locked off.** It is not, every sample clip
  pans, by 40 to 260 px. The vertical component of that lands directly on the
  curvature the physics depends on, and because a pan is smooth it biases the
  fit rather than averaging out. Camera motion is now measured per frame from
  the background and subtracted before anything else happens (`camera.py`).
- **The container frame rate was taken as the capture rate.** Two of the five
  sample clips are 120 fps footage in a 30 fps container, so every time and
  speed reported for them was wrong by a factor of four with nothing in the
  image to say so. The capture rate is now recovered from gravity, guarded by
  the standard error of the fitted acceleration so ordinary footage cannot
  declare itself slow motion (`calibration.resolve_time_base`).
- **The flight search looked for the window that fit *best*.** Short windows
  always fit better than long ones, so it clipped the flight at both ends,
  0.27 s for a flight that visibly lasts 0.40 s. Takeoff is now a change point
  read off the residual *at the candidate takeoff frame*, and entry is pushed
  forward while the trajectory still fits.
- **Subject selection ranked on total displacement**, which a diver whose track
  fragments at takeoff loses to a swimmer walking the deck. It now ranks on peak
  speed in body lengths per second, which is a property of an instant and
  therefore survives fragmentation. On `ginz_dive.mp4` this is the difference
  between measuring the diver and measuring a bystander behind the blocks.
- **The tracker fragmented the athlete** exactly where it mattered: three
  fragments of one swimmer on `test1.mp4`. Association now uses the mid-hip and
  keypoint agreement rather than box geometry, with a speed-dependent gate, and
  a separate stitching pass reconnects fragments after the fact.
- **The centre of mass was computed before conditioning**, so it was a sum over
  a different set of segments in every frame and moved when the athlete did not.
  All 17 keypoints are now completed and conditioned individually first
  (`pose.py`), which removes the composition bias and roughly halves the
  conditioned jitter on every sample clip.
- **The pixel scale came from the widest keypoint separation**, which is
  pose-dependent by a factor of three between a crouch and a streamline. It now
  comes from individual bone lengths against Winter's stature ratios
  (`anthro.py`); checked against a standing bystander whose height is directly
  measurable in the image, it agrees to 2%.
- **A single scale was applied to the whole clip**, which is wrong for a start
  filmed obliquely down the lane. Distances are now integrated with a per-frame
  scale when the athlete's apparent size shows a statistically real trend.
- The overlay drew stabilised coordinates onto unstabilised frames, so the trail
  drifted off the athlete by however far the camera had panned.

### Added

- **A second detection pass on a crop that follows the athlete** (`refine.py`).
  A whole-frame pass loses a motion-blurred body over the water, on
  `ginz_dive.mp4` at frame 40 of a flight ending at frame 50, and a measurement
  that was never made cannot be recovered downstream. Recovers those frames and
  improves the keypoint count on about half the rest, with consistency checks
  that never let it leave a track worse than it found it.
- Camera-motion, water-entry-cue and per-frame-scale panels in the diagnostic
  figure; camera drift, refinement counts and the recovered capture rate in the
  CLI summary and the JSON report.
- Joint trails (hips, shoulders, head) in the overlay, alongside the centre of
  mass, what the original project drew, kept as context rather than as the
  measurement.
- Plausibility warnings for a takeoff speed or flight time outside what a
  swimming start can physically produce.
- A self-consistency benchmark: `adive.mp4` and `adive2.mp4` are the same dive
  at different playback rates, so their disagreement is the pipeline's own
  error. Speeds and distances agree to 5%.
- `--capture-fps`, `--no-refine`, `--no-stabilise`, `--no-depth-scale`,
  `--no-slow-motion` and `--cache` on the CLI.
- Tests for camera motion, skeleton conditioning, anthropometry, stitching and
  refinement, on an anatomically proportioned synthetic athlete, so
  "generate at a known stature, analyse, check the calibration finds it again"
  is a real test rather than a tautology.

- Validation on `--stature-m`. A height in centimetres (`--stature-m 185`) is
  now refused with a message naming the mistake, and checked at argument-parse
  time rather than after minutes of inference. Every distance and speed is
  inversely proportional to this number, so a units slip is a factor of a
  hundred and nothing downstream can notice it, a scale is exactly what a
  single camera has no independent handle on.

### Changed

- The CLI's `scale:` summary line now reports the calibration the metrics were
  actually computed with, including its provenance. It previously printed the
  whole-clip body measurement while the calibration used the flight-window one,
  so the two lines of output could not be reconciled.
- Gravity is now always a *cross-check* and never the primary scale on ordinary
  footage, with the comparison reported either way. Its real job is the clock.
- `smartdive.com` is superseded by `smartdive.pose`; it remains for the tests
  that document the old behaviour.

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
  detection** (the length of the clip). Flight is now detected.
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
