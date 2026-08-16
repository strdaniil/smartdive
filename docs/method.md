# Method

How a video becomes a set of numbers, and why each stage is built the way it is.

## 1. Detection

A pose backend reports every person it can see in every frame, normalised to the
COCO-17 keypoint layout (`smartdive/keypoints.py`). Backends are deliberately
*identity-free*: they do not attempt to link people across frames. That job
belongs to the tracker, and keeping the two separate is what lets the same
tracker and the same physics run on top of any detector.

Two defaults differ from the original project and both matter more than they
look:

- **`yolo11m-pose` rather than `yolov8n-pose`.** The original used the smallest
  checkpoint available. On motion-blurred, mid-flight bodies the medium model
  closes detection gaps entirely — on `adive.mp4` the nano model lost the diver
  for four consecutive frames at full extension over the water, and the medium
  model does not.
- **Inference at 1280 px rather than the default 640.** A diver is about 400 px
  wide in a 1920 px frame; the usual letterbox throws away two thirds of the
  detail the keypoint head could have used.

## 2. Association

`tracking.py` runs a constant-velocity tracker. Each frame, existing tracks
predict where they should be, and a cost matrix over (track, detection) pairs is
solved with Hungarian assignment (`scipy.optimize.linear_sum_assignment`) so the
frame's assignment is globally optimal rather than greedy.

The cost has three terms, all dimensionless:

```
cost = |detection_centre - predicted_centre| / body_scale     position
     + 0.5 · |log(detection_size / track_size)|               scale consistency
     - 0.5 · IoU(predicted_box, detection_box)                overlap
```

Normalising the position term by body scale is what makes a single gate value
(0.9) work at any resolution and camera distance.

Tracks survive eight unmatched frames before closing, which is what carries the
athlete through the detector dropouts that happen at full extension.

## 3. Subject selection

This is the stage the original pipeline did not have, and its absence is the
root cause of the largest errors.

Selection is a **whole-clip decision**. Each track is scored by:

```
score = displacement_in_body_lengths
      + 0.35 · log(1 + n_observations)
      + 0.5  · mean_detector_confidence
```

Displacement dominates because it is the property that actually distinguishes an
athlete performing a start from everyone else in frame: divers cover two to four
body lengths, spectators cover none. Measuring it in body lengths rather than
pixels keeps the ranking independent of how far away the camera is.

Tracks whose mean detector confidence is below 0.4 are excluded from selection
before ranking. Without that gate the ranking can be won by a low-confidence
phantom that wanders across the frame and therefore scores well on displacement.
A track the detector was never sure about is not a person who moved. (This was
not hypothetical — it was picking a confidence-0.25 ghost on `ginz_dive.mp4`.)

The full ranking, including rejected tracks, is written to the JSON report so a
wrong pick is diagnosable without re-running anything.

## 4. Landmark: whole-body centre of mass

`com.py` computes the centre of mass as a mass-weighted sum of body segments,
using Winter/Dempster segment inertial parameters adapted to the joints COCO-17
provides (forearm and hand merged, shank and foot merged).

Two reasons this beats tracking a joint:

1. **Rotation invariance.** A diver who pikes or extends in flight moves their
   hips relative to their true centre of mass. Tracking the hips records that as
   a non-ballistic wobble, which then corrupts both the phase detection and the
   fit. The mass-weighted sum is, by construction, insensitive to rotation about
   the centre of mass.
2. **Noise averaging.** Independent per-joint detector noise averages down by
   roughly `sqrt(n_segments)`.

The cost is coverage: the estimate is returned only when at least 75% of body
mass is resolvable, so it goes NaN as the athlete submerges. That is the correct
behaviour — a CoM computed from whichever limbs happen to be above water is
biased towards them — but it means `--landmark hips` sometimes has more usable
frames on short clips. The reports print both coverage figures so the trade is
visible.

The same mass fraction, computed per frame, is reused as the **water entry
detector**; see §6.

## 5. Conditioning

`signal.py`, in a deliberate order: **reject, then fill, then smooth.**
Smoothing first would smear a spike across its neighbours instead of removing
it.

1. **Hampel filter** — a rolling median-absolute-deviation test, applied per
   axis. Unlike a mean/standard-deviation test it is not dragged around by the
   outlier it is trying to find. Catches left/right limb swaps, which displace
   one axis by roughly a body width in a single frame.
2. **PCHIP interpolation** across gaps of at most eight frames. Monotone cubic
   rather than a natural spline because it does not overshoot: a spline through
   a dropout either side of the trajectory apex can invent a bump that never
   happened. Longer gaps are left as NaN — a pipeline that invents a second of
   missing trajectory is lying. Every filled sample is flagged in the CSV.
3. **Savitzky-Golay, window 7, order 2.** Order 2 reproduces
   constant-acceleration motion *exactly*, so conditioning a projectile
   trajectory cannot bias the gravity estimate the next stage depends on. This
   is tested directly (`test_savgol_order_two_preserves_constant_acceleration`).

Derivatives use the Savitzky-Golay derivative rather than finite differences,
and are computed only within contiguous runs so a gap cannot create a phantom
acceleration spike at its edges.

### Estimating the measurement noise

The flight search needs a noise scale to set its inlier tolerance. Taking the
best-fitting window's residual would be biased low precisely *because* that
window was chosen for having the smallest residual.

Instead, for a smooth signal sampled with independent noise `σ`, the second
difference `y[i-1] − 2y[i] + y[i+1]` has variance `6σ²`, and the smooth motion
contributes almost nothing because a second difference annihilates anything
locally linear. So `σ = RMS(second difference) / sqrt(6)`, needing no knowledge
of the trajectory.

The tolerance then uses the **larger** of that estimate and the seed fit's own
residual, because two different error sources are in play and either can
dominate: detector jitter, and how well the projectile model describes the
chosen landmark at all.

## 6. Phase detection

The airborne window is found by seeding a model and expanding it to a consensus
set — RANSAC, essentially — with a different cue anchoring each end.

**Seed.** Every contiguous window of at least six frames is fit with the
projectile model, and the best seed minimises `residual_rms / distance
travelled`. Normalising by travel rather than by pixels is what stops the
near-motionless crouch on the block from winning purely by having nothing to be
wrong about.

**Takeoff, from the physics.** The seed's model is scored against every frame,
and the window grows left through inliers, bridging up to two consecutive
outliers so one badly localised joint does not truncate a flight that plainly
continues. While a foot is still on the block the athlete is being accelerated
by their legs, which a constant-acceleration model cannot express, so the
residual rises steeply — on `adive.mp4`, from 16 px at takeoff to 260 px eight
frames earlier. Sharp and unambiguous.

**Entry, from visibility.** The same growth cannot find the far end, and it is
worth being clear about why rather than tuning until it appears to: a
streamlined swimmer in the first tenth of a second underwater *is still
travelling on very nearly a parabola*. The residual has nothing to detect. On
`adive.mp4` every window from frame 15 out to frame 37 fits within the noise.

So entry uses an independent observation. A pose detector can only label joints
it can see, so the fraction of body mass it resolves collapses the instant the
athlete breaks the surface — measured across the sample clips, from 1.00 to 0.35
in a single frame, every time:

```
adive       ... 24:1.00 25:1.00 26:0.35 27:0.35 29:0.33 ...
adive2      ... 102:1.00 103:1.00 104:0.33 105:0.33 ...
adive_wide  ...  71:1.00  72:1.00  73:0.35  74:0.35 ...
test1       ...  97:1.00  98:1.00  99:0.38 100:0.35 ...
```

Frames with no detection at all are treated as *unknown*, not submerged — not
seeing the athlete is not evidence that they are underwater, and treating it as
such would end the flight at the first dropped frame.

**Rejection.** If no window survives — too little travel, or an implied athlete
stature outside 1.4–2.6 m — the analysis reports no flight rather than a number
it cannot support. Two of the five sample clips take this path.

## 7. Scale

See the README's calibration section for the practical summary. The mechanism:

The flight fit **holds `a_y` fixed** at `g · px_per_metre` from a body-scale
prior, rather than estimating it. This is not a shortcut, it is a conditioning
fix. A free fit has to isolate a curvature comparable in size to the keypoint
noise, and on the sample clips its scale estimate ranges over a factor of 2.6
depending only on where the window ends. Fixing `a_y` removes the degeneracy,
and as a side effect the residual becomes a much sharper test of whether a frame
was airborne.

The prior comes from the **90th-percentile body scale** — the athlete stretched
out in flight, not folded on the block — divided by `stature × 1.15`, the
approximate ratio of widest keypoint separation to stature for a swimmer at full
extension.

After the window is known, a *free* fit is run over it and its gravity estimate
is compared with the prior. Within 35%, gravity is reported as the scale.
Outside that, the prior is reported and the disagreement is recorded as a
warning. Either way the method is named in the output.

## 8. Metrics and uncertainty

`metrics.py` converts the fit and the scale into SI kinematics. Displacements
are read off the *modelled* path rather than from raw endpoints, which are the
noisiest samples in the window.

Uncertainty is a residual bootstrap: resample the fit residuals, refit, and take
the 2.5th and 97.5th percentiles of each parameter. The resampled fits use the
same constrained model as the reported metrics — an interval from a free fit
does not bound a constrained estimate, and would quietly fail to contain the
number it is printed next to.

## Assumptions

Collected in one place:

1. Motion is roughly parallel to the image plane; the camera does not pan.
2. Constant frame rate, read from the container.
3. One start per clip.
4. Adult athlete, 1.4–2.6 m widest keypoint separation.
5. Free flight is unresisted — air drag over a 0.3 s flight is far below the
   measurement noise, so it is not modelled.

Violations of 1 and 2 will produce a confident wrong answer rather than a
failure, which is the main thing to watch for.
