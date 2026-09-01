# Method

How a video becomes a set of numbers, and why each stage is built the way it is.

The pipeline runs in one direction with one loop in it. Detection, camera-motion
estimation, association, subject choice and skeleton conditioning all happen
once; the physics runs twice, because two of its inputs are things it has to
find the flight in order to measure.

## 1. Detection

A pose backend reports every person it can see in every frame, normalised to the
COCO-17 keypoint layout (`smartdive/keypoints.py`). Backends are deliberately
identity-free: they do not link people across frames. That job belongs to the
tracker, and keeping the two separate is what lets the same tracker and the same
physics run on top of any detector.

Two defaults differ from the original project and both matter more than they
look:

- **`yolo11m-pose` rather than `yolov8n-pose`.** The original used the smallest
  checkpoint available. On motion-blurred, mid-flight bodies the medium model
  closes detection gaps the nano model leaves open.
- **Inference at 1280 px rather than the default 640.** A diver is about 400 px
  wide in a 1920 px frame; the usual letterbox throws away two thirds of the
  detail the keypoint head could have used.

## 2. Camera motion

`camera.py`. The projectile model is only valid in a frame of reference that is
not itself accelerating, and poolside footage is almost never shot from a
locked-off tripod. Whoever is filming pans to follow the athlete, starting the
pan the moment the athlete leaves the block, inside the flight window, where the
fit cannot tell a camera acceleration from the athlete's own.

The size of the problem is easy to underestimate. On the sample clips the
background drifts 40 to 260 px over the clip, one to eight percent of the
athlete's travel. That sounds negligible and is not, for two reasons.

1. The vertical component lands directly on the quantity the physics depends on.
   Over a 0.35 s flight the parabola departs from its own chord by only
   `g*T^2/8`, some 25 px at typical framing, so a 5 px vertical drift is a 20%
   error in the measured gravity and therefore in any scale derived from it.
2. It is not white noise. A pan is smooth and monotone, so it does not average
   out over the window; it biases the fit in one direction.

Sparse Shi-Tomasi corners are tracked between consecutive frames with
Lucas-Kanade optical flow, with a forward-backward consistency check (occlusion
boundaries, which poolside means the edge of every person, produce
confident-looking flow that a one-way check accepts). Everything inside a
detected person's bounding box, dilated by 25%, is discarded before tracking. A
similarity transform is then fitted to what is left by RANSAC, and the per-frame
transforms are composed into a warp from every frame into one reference frame.

A similarity transform rather than a homography, deliberately: hand-held pans
over a few tenths of a second are dominated by translation with a little roll
and zoom, and a homography has eight degrees of freedom to fit that with, enough
to start absorbing part of the athlete's motion through the perspective terms
when the usable background is concentrated on one side of the frame, which
poolside it usually is.

Everything downstream works in the stabilised frame. The overlay renderer maps
back through the inverse warp so the trail sits on the body it describes.

## 3. Association

`tracking.py` runs a constant-velocity tracker. Each frame, existing tracks
predict where they should be, and a cost matrix over (track, detection) pairs is
solved with Hungarian assignment so the frame's assignment is globally optimal
rather than greedy.

The cost blends four dimensionless terms:

```
cost = mean( |detection_anchor - predicted_anchor| / body_scale,
             median |shared keypoints displaced| / body_scale )
     + 0.15 * |log(detection_size / track_size)|
     - 0.35 * IoU(predicted_box, detection_box)
```

Two changes from the previous version, both about a body that is changing shape:

- The position term uses the mid-hip, not the box centre. A diver's arms go from
  tucked to fully extended during takeoff, which shifts the box centre by a
  quarter of a body length without the athlete's mass having gone anywhere.
- A keypoint-agreement term carries most of the weight. The ankles of a diver
  mid-flight are still that diver's ankles, however little their bounding boxes
  overlap; the size and IoU terms are down-weighted accordingly, because a
  diver's box legitimately changes shape by a factor of two through takeoff.

The matching gate widens in proportion to the track's own speed. A person moving
half a body length per frame cannot be held to a fixed positional tolerance,
because a constant-velocity prediction is only as good as the assumption that
they did not accelerate, and at takeoff they did.

## 4. Stitching

Even a good frame-to-frame tracker breaks at takeoff. In two frames the
athlete's box rotates from vertical to horizontal, doubles in width, halves in
height, and moves half its own width; the detector then loses them for several
frames over the water. A tracker tolerant enough to survive that is also
tolerant enough to swap two swimmers standing on adjacent blocks.

So the tracker stays strict and a separate pass reconnects fragments afterwards,
where it has the whole clip to reason over instead of one frame of context. Two
regimes:

- **A clean gap.** Fragment A is extrapolated forwards and fragment B backwards,
  and they have to meet within a tolerance that grows with the gap, since a
  constant-velocity extrapolation over eight frames of an accelerating body
  cannot be held to the same standard as one over two.
- **A temporal overlap.** A strict tracker routinely opens the new fragment
  several frames before it gives up on the old one, so two fragments overlapping
  by up to ten frames are merged when they agree on the frames they share.

Merging is greedy rather than globally optimal on purpose: the cost is a
hard-gated plausibility check, not a soft preference, and a wrong merge is much
more damaging than a missed one.

On `test1.mp4` this turns three fragments of one swimmer into one track of 102
observations. On `ginz_dive.mp4` it reconnects the block phase to the flight
across the three frames where the detector lost the diver entirely.

## 5. Subject selection

This is the stage the original pipeline did not have, and its absence is the
root cause of the largest errors.

Selection is a whole-clip decision, and it ranks primarily on peak speed in body
lengths per second:

```
score = peak_speed_bodies_per_second            (90th percentile, not the max)
      + 0.8  * min(displacement_in_body_lengths, 6)
      + 0.35 * log(1 + n_observations)
      + 0.5  * mean_detector_confidence
```

Total displacement, which the previous version ranked on, is fragile in exactly
the case that matters. A diver whose track is broken into three pieces shows
little displacement per piece, while a swimmer walking the length of the deck
shows a lot. Peak speed does not have that problem: it is a property of an
instant rather than of a whole track, so it survives fragmentation, and being
expressed in body lengths it is invariant to camera distance. Divers reach three
to eight body lengths per second at takeoff; walking is about one; standing is
none.

The 90th percentile rather than the maximum, so that one badly localised frame
cannot manufacture a diver out of a bystander.

Two eligibility filters run before the ranking, both to stop a fragment of a
person from winning:

- mean detector confidence below 0.35: a track the detector was never sure about
  is not a person who moved;
- median body scale below 45% of the clip's median: a mis-detected limb, which
  because speed is expressed in body lengths would otherwise score
  spectacularly well. (Not hypothetical: a 1-pixel "body" on `adive2.mp4` scored
  86 body lengths per second.)

The full ranking, including rejected tracks and why they were rejected, goes
into the JSON report so a wrong pick is diagnosable without re-running anything.

## 6. Refinement: a second look at a follow-crop

`refine.py`. A whole-frame pose pass has to spend its input resolution on the
whole frame. At 1280 px of network input on a 1920 px clip, an athlete 400 px
tall is 270 px to the network and their wrist is a handful of pixels. That is
survivable while they stand on the block and it is not survivable at the moment
they are a motion-blurred horizontal streak over the water, which is precisely
the part of the clip the entire measurement depends on.

The symptom is not a wrong answer, it is a missing one. On `ginz_dive.mp4` the
detector stops reporting the diver at frame 40 of a flight that ends at frame
50, so half the flight is unmeasured. No amount of downstream cleverness
recovers a measurement that was never made.

So once the athlete has been identified, the video is read again, at a crop
around where the tracker says they are, resized so they fill the network's
input. Same weights, same frames, several times the effective resolution. The
window extends eight frames past both ends of the track, because frames the
first pass dropped entirely are exactly how a track ends.

Two checks keep the second opinion from doing harm:

- an accepted crop detection must overlap the predicted box by 35% and sit
  within 0.45 body lengths of its centre, because on a crowded pool deck two
  swimmers on adjacent blocks produce boxes that overlap a prediction
  comfortably while being different people;
- every refined frame is compared against a local prediction built from its own
  neighbours and discarded if it disagrees by more than half a body length. Only
  refined frames can be discarded: the pass is never allowed to leave the track
  worse than it found it.

That second check earns its place: on `ginz_dive.mp4` two frames that landed on
a bystander tripled the estimated measurement noise, which is the quantity every
tolerance downstream is scaled by.

Measured across the sample clips, the pass recovers frames the first pass had
none for and improves the keypoint count on roughly half the rest.

## 7. The skeleton, then the landmark

`pose.py`. The previous version conditioned the landmark, one `(N, 2)`
trajectory, after computing the centre of mass frame by frame from whichever
keypoints happened to be confident. That ordering has a failure mode which is
easy to miss and expensive to live with.

The centre of mass is a weighted sum over body segments. If a segment drops out
of one frame, the sum is taken over a different set of segments in that frame, so
the estimate moves even when the athlete does not. Losing one shank shifts it by
around 3% of body length toward the head, eight pixels at typical framing,
comparable to the entire curvature signal a 0.35 s flight contains. The
trajectory then looks noisy, and worse, the noise is correlated with body
configuration rather than random, so it does not average out over the fit. The
obvious defence, refusing to compute a centre of mass unless most of the body is
visible, trades the bias for dropouts and threw away nearly half of every sample
clip.

So the order is inverted. Each of the 17 keypoints is completed and conditioned
as its own time series, and only then are the segment masses summed:

1. **Contralateral mirroring.** In a lateral view the two legs and the two arms
   project to nearly the same place, so a missing keypoint is filled from its
   mirror when that mirror is confident. This is what rescues the frames where
   the detector sees only the near half of a body. It is not valid for footage
   shot down the lane, and `lateral_view_score`, the ratio of shoulder
   separation to trunk length, reports how side-on the camera actually is.
2. **Reject, then fill, then smooth**, per keypoint (`signal.py`, below).
3. **Sum the segment masses** using Winter/Dempster parameters adapted to the
   joints COCO-17 provides (forearm and hand merged, shank and foot merged).

Long gaps are still left as gaps. When a swimmer is half-submerged their legs
are not missing because the detector had a bad frame; they are missing because
they are underwater, and inventing them would erase the clearest signal of water
entry the video contains.

For the same reason, the visible mass fraction used as the entry cue is computed
from the raw detector confidences, never from the completed skeleton.

## 8. Conditioning

`signal.py`, in a deliberate order: reject, then fill, then smooth. Smoothing
first would smear a spike across its neighbours instead of removing it.

1. **Hampel filter**, a rolling median-absolute-deviation test, applied per
   axis. Unlike a mean/standard-deviation test it is not dragged around by the
   outlier it is trying to find.
2. **PCHIP interpolation** across gaps of at most eight frames. Monotone cubic
   rather than a natural spline because it does not overshoot: a spline through
   a dropout either side of the trajectory apex can invent a bump that never
   happened. Longer gaps are left as NaN, since a pipeline that invents a second
   of missing trajectory is lying. Every filled sample is flagged in the CSV.
3. **Savitzky-Golay, window 7, order 2.** Order 2 reproduces
   constant-acceleration motion exactly, so conditioning a projectile trajectory
   cannot bias the gravity estimate the next stage depends on. This is tested
   directly (`test_savgol_order_two_preserves_constant_acceleration`).

Derivatives use the Savitzky-Golay derivative rather than finite differences,
and are computed only within contiguous runs so a gap cannot create a phantom
acceleration spike at its edges.

### Estimating the measurement noise

The flight search needs a noise scale to set its tolerance. Taking the
best-fitting window's residual would be biased low precisely because that window
was chosen for having the smallest residual.

Instead, for a smooth signal sampled with independent noise `s`, the second
difference `y[i-1] - 2y[i] + y[i+1]` has variance `6*s^2`, and the smooth motion
contributes almost nothing because a second difference annihilates anything
locally linear. So `s = RMS(second difference) / sqrt(6)`, needing no knowledge
of the trajectory, and it is measured on the unconditioned landmark, because
conditioning correlates neighbouring samples and would make the measurement look
quieter than it was.

The tolerance is then clipped into a band, 5% to 8% of a body length. The floor
stops an unusually quiet clip from demanding impossible precision; the ceiling
matters more, because on a clip where the detector is having a hard time the
noise estimate comes out large enough to swallow the block phase whole, at which
point the search happily reports a "flight" that starts while the athlete is
still crouched.

## 9. Phase detection

`physics.py`. The two ends of a flight are found by different evidence, because
each cue is sharp at one end and blind at the other.

### Episodes

The search does not begin from a single global maximum of speed. A clip contains
several intervals in which the athlete is measurable and out of the water: the
dive, then the swimmer surfacing, then on a long clip the next swimmer's
breakout. Any of them can be the fastest thing in the clip: on a 120 fps
recording the flight advances a dozen pixels per frame while a swimmer breaking
the surface thirty frames later advances thirty, and their apparent centre of
mass lurches because half of them is hidden.

So `airborne_episodes` enumerates the intervals and each is evaluated
separately. Short interruptions are bridged: two consecutive frames below the
visibility threshold are needed to end an episode, since one frame in which the
detector loses a leg is not water.

### Entry, from visibility

A streamlined swimmer in the first tenth of a second underwater is still
travelling on very nearly a parabola. A residual test has nothing to detect. On
`adive.mp4` every window from frame 15 out to frame 37 fits within the noise.

So entry uses an independent observation. A pose detector can only label joints
it can see, so the fraction of body mass it resolves collapses the instant the
athlete breaks the surface, measured across the sample clips from 1.00 to 0.35
in a single frame, every time:

```
adive       ... 24:1.00 25:1.00 26:0.35 27:0.35 29:0.33 ...
adive2      ... 102:1.00 103:1.00 104:0.33 105:0.33 ...
adive_wide  ...  71:1.00  72:1.00  73:0.35  74:0.35 ...
test1       ...  97:1.00  98:1.00  99:0.38 100:0.35 ...
```

Frames with no detection at all are treated as unknown, not submerged: not
seeing the athlete is not evidence that they are underwater, and treating it as
such would end the flight at the first dropped frame.

The visibility cue slightly precedes full immersion: a fully extended body a few
centimetres above the water loses its legs to a pose detector before it loses
them to the water. So after the window is fixed, entry is pushed forward while
the extra frames still fit the flight model, up to four frames. This is safe in
the direction that matters because water is not gravity: a body decelerating in
it leaves the fixed-acceleration model conspicuously, so the same residual test
that admits the extra airborne frames refuses the submerged ones.

### Takeoff, from the physics, one frame at a time

With entry fixed, the takeoff frame is walked backwards, and the test is the
residual at the proposed takeoff frame, not a summary of the window.

While a foot is still on the block the athlete is being accelerated by their
legs, which a constant-acceleration model cannot express, so that one frame's
residual climbs steeply and unambiguously. On `adive.mp4`, over five consecutive
candidate frames:

```
start frame   14     13     12     11     10
residual    15.3   20.8   29.8   41.3   55.5  px
window q80  15.1   15.7   16.0   19.5   26.7  px
```

The signal is carried by the candidate frame; a summary of the window dilutes
it.

This is also the difference from the previous version, which searched for the
short window that fit best. Short windows always fit better than long ones (a
fit has the same number of parameters either way, so shrinking the window can
only reduce the residual), and it therefore clipped the flight at both ends,
reporting 0.27 s for a flight that visibly lasts 0.40 s.

The obvious alternative, judging the whole window against a threshold that scales
with it, is worse than merely diluted: a window reaching one frame further back
into the block phase is also one frame longer, so the threshold relaxes exactly
as the evidence against the window mounts. That feedback made the reported
flight time swing by a quarter between two runs that differed only in where the
entry frame landed.

A budget of failures is allowed while walking back, the greater of two frames
and 30% of the frames scanned so far, because a fixed run length means a shorter
interruption in real time the higher the capture rate, and slow-motion clips are
exactly where the residual profile is bumpiest.

### Choosing between candidates

Every (episode, entry-trim) pair that produces an accepted window is ranked by
its residual quantile per unit of travel, and the best wins. Ranking by travel
alone is maximised by running the window on into the water, where the athlete is
still moving and no longer ballistic; a scale-free residual is not, and it is
also directly comparable between a dive and a swimmer's breakout.

### Rejection

If no window survives (too little travel, or an implied athlete stature outside
1.3 to 2.4 m when gravity was fitted freely) the analysis reports no flight
rather than a number it cannot support, and says which.

## 10. Scale

`anthro.py` and `calibration.py`. See the README's calibration section for the
practical summary; the mechanism:

### Measuring the athlete

The previous version took the widest separation between any two confident
keypoints and called it 1.15 * stature. That quantity is pose dependent: roughly
stature standing, a third of that crouched on a block, about 1.2 * streamlined in
flight. The pose changes continuously through the clip, so the estimate does too,
and one correction factor can only be right for one pose.

What is measured instead is the length of individual body segments (trunk,
thigh, shank, upper arm, forearm, and the hip-to-ankle composite), each a fixed
fraction of stature that does not depend on what the athlete is doing. Two
properties make it work:

- **Projection only ever shortens.** A segment of true pixel length `L` seen at
  an angle `t` out of the image plane projects to `L*cos(t) <= L`. So the upper
  end of the distribution of a segment's observed length over a clip estimates
  its true length, and every foreshortened frame is discarded by taking a high
  percentile. The 85th rather than the maximum, because detector noise has no
  such one-sided bound.
- **Segments disagree independently.** Each gives its own stature estimate;
  combining them with a weighted median, weighted towards the long, near-planar
  leg segments, means one badly tracked limb cannot move the answer, and
  segments more than 35% from the consensus are trimmed and the consensus
  recomputed. A mislocated wrist can double the apparent length of a forearm,
  and a forearm is short enough that doubling it is not obviously absurd until it
  is compared with the thigh.

Only keypoints the detector reported are measured, never mirrored or interpolated
ones, so the ruler is never built out of reconstructions.

Checked against a standing bystander in `adive.mp4`, whose eye-to-ankle height
is directly measurable in the image, the estimate agrees to 2%.

One ratio is adjusted from Winter's table. Shoulder height (0.818) minus hip
height (0.530) gives a trunk of 0.288 stature, but those are surface heights, and
COCO's shoulder keypoint is annotated at the glenohumeral joint. Measured on the
sample clips the resulting shoulder-to-hip distance sits between 0.85 and 1.15 of
what 0.288 predicts, depending on how side-on the athlete is, because a trunk
pointing at the camera (a track start on the blocks) foreshortens more than a
limb does. So the trunk keeps a slightly reduced ratio and a low weight.

### Depth: when one scale is not enough

Filmed obliquely down the lane rather than square-on, the athlete recedes as they
fly and one pixels-per-metre is wrong at one end of the flight or the other. The
same machinery gives a per-frame size for free: each visible segment votes with
its own length divided by the length it would have at the reference scale, and
the frame's size is the median of those votes. Normalising each segment by its
own reference is what removes the pose dependence: when the athlete pikes, every
segment foreshortens somewhat, but the median vote moves far less than any single
one.

That per-frame estimate is still noisy, so it is not used directly. A straight
line is fitted to log-size across the flight and accepted only if its slope is at
least three times its own standard error. Depth changes smoothly (an athlete
cannot get closer to the camera and back again within a tenth of a second) so the
trend is the signal and the scatter is not. The obvious alternative, a
max-over-min ratio, reads 31% on `adive_wide.mov` across eight frames of a dive
that barely changes depth at all.

Distances are then integrated step by step, each step converted at the scale in
force where it happened. With a constant calibration this reduces exactly to
dividing by it once.

The correction fixes the scale of a receding athlete; it cannot recover the
component of their velocity pointing into the frame, and the report says so.

### Gravity as a cross-check

After the window is known, a free fit is run over it and its gravity estimate is
compared with the body-scale figure. The comparison is reported either way, as
agreement or as disagreement with an explanation. Gravity is never the primary
scale on ordinary footage, for the reason in the README: over a 0.35 s flight at
30 fps the curvature it depends on is comparable to the keypoint noise, and the
estimate ranges over a factor of 1.8 depending only on where the window ends.

For the flight fit itself, `a_y` is held fixed at `g * px_per_metre` from the
body-scale figure rather than estimated. This is a conditioning fix, not a
shortcut: fixing it removes the degeneracy between curvature and launch speed,
and as a side effect the residual becomes a sharper test of whether a frame was
airborne. Where the constrained search finds nothing (which happens when the
prior and the footage disagree) the unconstrained window is used instead, and
both are always run for exactly that reason.

## 11. The clock

`calibration.resolve_time_base`. A clip's container frame rate is what it will be
played at, which for anything shot in a phone's slow-motion mode is not what it
was shot at. Times, speeds and accelerations are all wrong by that factor, and
nothing in the image says so.

Gravity gives an independent clock. If `k` is the number of real seconds per
second of container time, a body in free fall shows an apparent vertical
acceleration of `g * px_per_metre * k^2`, because the acceleration term goes as
time squared. Having pinned `px_per_metre` from the athlete's own body:

```
k = sqrt(a_y_observed / (g * px_per_metre))
capture rate = container_fps / k
```

Three guards, all necessary:

- **The window is trimmed.** Gravity is re-fitted over the last 70% of the
  scouted window. The scout runs with gravity free (that is what lets it bracket
  a clip whose clock is unknown) but a free fit can also flatten its parabola to
  absorb a few frames of block push at the start, and the flatter the parabola,
  the slower the clip appears. Dropping the first third moves the recovered rate
  from 220 to 138 fps on a clip actually shot at 120, and from 143 to 118 on
  another.
- **The acceleration must be well determined**, checked directly from the
  standard error of the quadratic coefficient rather than from a proxy. On
  ordinary 30 fps footage that comes out at 27% of the coefficient, which is why
  gravity cannot be a scale there, and equally why such a clip must not be
  allowed to declare itself slow motion on the strength of a number it cannot
  measure. Slow-motion footage, by construction, has the long flight in frames
  that makes the same measurement easy: 4% and 12% on the two sample clips.
- **The correction must be large.** Below 45%, it is left alone rather than
  "corrected" into the noise.

A recovered rate within 15% of a standard capture rate (24, 25, 30, 48, 50, 60,
120, 240, 480) is snapped to it, because "shot at 120 fps" is a far more likely
explanation for a factor of four than "shot at 118.4 fps".

On the sample footage this recovers 120 fps on both slow-motion clips and leaves
the three real-time clips alone. `--capture-fps` overrides it either way.

## 12. Metrics and uncertainty

`metrics.py` converts the fit and the scale into SI kinematics. Displacements are
read off the modelled path rather than from raw endpoints, which are the noisiest
samples in the window.

Uncertainty is a residual bootstrap: resample the fit residuals, refit, and take
the 2.5th and 97.5th percentiles of each parameter. The resampled fits use the
same constrained model as the reported metrics: an interval from a free fit does
not bound a constrained estimate, and would quietly fail to contain the number it
is printed next to.

A final plausibility pass flags a takeoff speed outside 1.5 to 8.0 m/s or a
flight time outside 0.15 to 0.60 s. The bands are wide on purpose: they are meant
to catch a broken scale or a mis-detected phase, not to grade an athlete.

## Assumptions

Collected in one place:

1. Motion is roughly parallel to the image plane. Camera pan is measured and
   removed; out-of-plane athlete motion is detected, corrected for scale, and
   warned about, but the velocity component into the frame is unrecoverable from
   one camera.
2. The frame rate is constant. It need not be the container's (see section 11)
   but a variable-rate recording is not handled.
3. One start per clip. Several swimmers in frame is the normal case and fine.
4. Adult athlete, implied stature 1.3 to 2.4 m.
5. The view is side-on enough for contralateral mirroring to hold. Measured and
   warned about, not assumed.
6. Free flight is unresisted: air drag over a 0.3 s flight is far below the
   measurement noise, so it is not modelled.
