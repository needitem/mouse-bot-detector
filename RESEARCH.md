# mouse-bot-detector — research log

> This is the full, chronological research record: every version, every failed
> attempt, and every result, kept honest and unedited. For a concise overview of
> what the project is and what it concluded, see [README.md](README.md). Some
> data-source notes below (e.g. the aim-rl-web-collector Cloudflare Worker API)
> describe how things worked at the time of writing; the collector has since
> moved to a static, download-based flow.

---

A bot-vs-human mouse-movement classifier, built to adversarially evaluate
`needaimbot`'s `motor_synergy` humanized-flick generator
(`inferencetool/inference_pc/needaimbot/mouse/motor_synergy.hpp`) against
published mouse-dynamics research and real human data, rather than just
eyeballing whether a generated trajectory "looks human."

## Papers referenced

- **Acien, Morales, Fierrez, Vera-Rodriguez, Tolosana**, *"BeCAPTCHA-Mouse:
  Synthetic Mouse Trajectories and Improved Bot Detection"*, Pattern
  Recognition 2022 (arXiv:2005.00890) - sigma-lognormal decomposition of
  mouse trajectories as the human-vs-bot feature space; the human motor
  signature (initial acceleration / final deceleration / fine end-correction)
  this project's features approximate. A full iterative sigma-lognormal fit
  (XZERO/iDeLog) is out of scope here - see "Scope cuts" below.
- **Fülöp, Kovács, Kurics, Windhager-Pokol**, *Balabit Mouse Dynamics
  Challenge Data Set* (2016), github.com/balabit/Mouse-Dynamics-Challenge -
  real human RDP mouse-event logs, 10 users; this project's primary "human"
  ground truth (see "Known confound" for the caveats that come with it).
- **Harris & Wolpert (1998)**, signal-dependent noise - the same principle
  already implemented in `motor_synergy`'s SDN term; reused here as a
  detector feature (`sdn_correlation`: does jitter scale with local speed
  the way it does in real human data?).
- **Fitts (1954)**, law of movement time vs. index of difficulty - basis for
  several kinematic features and for `motor_synergy`'s own timing model.

## Pipeline

```
scripts/fetch_dataset.sh          # git-clone Balabit's public dataset
scripts/parse_balabit.py          # segment sessions -> movements, keep only the fast/urgent tier
scripts/fetch_web_collector.py    # pull real "human-web" sessions from the deployed
                                   # aim-rl-web-collector Cloudflare Worker (see below)
scripts/generate_synthetic.py     # naive_bot + a pure-Python port of motor_synergy::generate()
scripts/features.py               # shared feature extraction (used identically on all classes)
scripts/train_detector.py         # hyperparameter-searched classifiers, held-out test evaluation
scripts/adversarial_loop.py       # evolves motor_synergy's config against the detector's own feedback
```

Run in that order (`fetch_web_collector.py` is optional/best-effort - see below).

### Data sources

- **Balabit** (`human_movements.jsonl`): real RDP desktop mouse use. Filtered
  to only the fastest quartile of movements by mean px/s (a data-driven
  stand-in for "moved with the same urgency as acquiring a game target,"
  since most RDP desktop use - slow scrolling, careful dragging - looks
  nothing like a flick). 65 sessions -> 16,544 kept after the shape + speed
  filters.
- **aim-rl-web-collector** (`human_movements_web.jsonl`): a second, much
  better-matched human source - github.com/needitem/aim-rl-web-collector's
  browser game has players continuously track a moving on-screen target,
  deployed live at aim-rl-web-collector.th07290828.workers.dev. No changes
  to that project were needed - its `GET /api/sessions` +
  `GET /api/sessions/{id}/jsonl` API already exposes everything
  `fetch_web_collector.py` needs. **As of this writing nobody has actually
  played it yet** (the only sessions on the deployed instance are deploy
  smoke tests), so this file is currently empty - `train_detector.py`
  silently skips it until real sessions exist. Re-run
  `fetch_web_collector.py` any time to pick up new sessions.
- **naive_bot**: constant-velocity straight line + small Gaussian jitter. No
  Fitts timing, no submovements, no curvature - a stand-in for a naive
  aimbot/macro; the sanity-check baseline ("can the detector even tell this
  obviously fake movement apart?").
- **motor_synergy_bot**: a faithful pure-Python port of
  `motor_synergy::generate()` (Fitts-timed primary submovement + 0-2
  corrections via lognormal CDFs, direction-dependent curvature,
  Ornstein-Uhlenbeck drift, velocity-modulated tremor, signal-dependent
  noise) - ported since offline training-data generation doesn't need the
  GPU split that matters for real-time use.

All synthetic movements are distance-matched to the real human distribution
(same empirical distances, random direction) so the detector can't just
learn "distance" as a shortcut instead of movement shape.

## Known confound (discovered, mostly fixed)

Balabit's human data is RDP-captured at a much coarser, more irregular rate
(median ~109ms between samples) than either synthetic bot class (~7-8ms, a
native ~125Hz mouse polling rate). Any feature built on numerical
differentiation amplifies noise by roughly `1/dt` per derivative order -
jerk is a **third** derivative, so `1/dt^3` - which initially made
jerk/curvature/tremor massively overstated for the synthetic classes purely
as a capture-pipeline artifact, not a genuine movement-quality difference.

Confirmed empirically, not just in theory: an early adversarial search
against the unfixed features found configs that fooled one detector
instance during search but never generalized to a fresh one - exactly what
you'd expect if the "signal" being gamed was a sampling-rate artifact rather
than real movement shape.

**Fix**: `features.py` now resamples every movement to a common 40ms grid
(a deliberate compromise - matching Balabit's own ~109ms cadence exactly
would leave most sub-300ms flick movements with only 2-3 points and no
shape signal at all) before computing anything derivative- or FFT-based.
Only `sample_interval_mean/cv` (which deliberately measure the *raw*,
pre-resample capture cadence) remain excluded from the `shape_only` feature
set reported everywhere.

## Detector results (see `results/report.md` for the full breakdown)

Both pairings - `human vs naive_bot` and `human vs motor_synergy_bot` - are
random-searched across 4 model families (RandomForest, GradientBoosting,
HistGradientBoosting, SVM-RBF) with a genuine held-out test split never
touched during the search. Even on `shape_only` (confound-reduced) features,
**both bot classes are detected at 99.7-99.9% accuracy** - the confound fix
mattered (jerk/curvature values are now the same order of magnitude across
classes, not 10-100x off), but it didn't make the classification problem
hard. The top remaining features (`curvature_rms`, `mean_speed`) point at a
real, not-yet-fixed mismatch: our synthetic generators' speed/curvature just
don't match the specific "fast/urgent" Balabit subset closely enough yet.

## Adversarial loop: does tuning motor_synergy's config help?

`adversarial_loop.py` evolves `motor_synergy`'s config (curvature, OU
drift, tremor, SDN, Fitts timing - 14 parameters) against the detector's own
`predict_proba`, retraining the detector each epoch so the search can't just
overfit one stale instance, then validates the final "best" config against
a **fresh, independent detector that never saw the search**.

Run three times, under three different detector strengths (a small
untuned RandomForest; a larger sample of the same; and finally the actual
best hyperparameters `train_detector.py`'s own search found for
HistGradientBoosting) - **all three times, the pattern was identical**:

- During search, the evolutionary loop reliably drives the *current*
  epoch's detector down to chance level (`mean P(bot)` as low as `0.000`).
- Against a fresh, independently-trained detector, the "evolved" config
  performs statistically identically to the untouched default
  (`0.994` vs `0.994` accuracy in the final run).

**Conclusion**: this isn't a fluke of one weak search - it replicated across
three different detector strengths. Two readings, both worth taking
seriously:

1. **The detector is robust.** Direct, repeated, gradient-free optimization
   access to its own predictions could not find a `motor_synergy` config
   that generalizes past the specific trained instance it was searched
   against - a meaningfully strong result for a classical-ML classifier.
2. **The evolutionary search over motor_synergy's config alone isn't
   finding a genuine fix** to the speed/curvature mismatch `train_detector.py`
   flagged - it's finding narrow, detector-instance-specific blind spots
   instead (the classic adversarial-example-doesn't-transfer failure mode,
   just in a feature-engineered classical-ML setting rather than a neural
   one).

### v2: detector ensemble + light distribution matching

To attack that failure mode directly rather than just running the same
search longer, `adversarial_loop.py` was revised: fitness evaluated against
an **ensemble of 4 independently bootstrapped detectors** (a config must
fool several differently-trained models at once, not one), plus a light
distribution-matching term (z-score distance of `mean_speed`/`curvature_rms`/
`jerk_rms` from human). Result: a small, real, directionally-consistent
improvement (accuracy 0.998->0.991, distribution distance 0.205->0.183) -
modest, but for the first time not "literally zero change."

### v3: distribution matching as the DOMINANT objective, loop-until-converged

Pushed further, on the "완성될 때까지 계속" (keep training until it's done)
instruction: distribution matching weight raised substantially and widened
from 3 to 9 features (speed, curvature, jerk x2, path efficiency, velocity
skew, timing x2), and the script now loops epochs automatically until either
converged (mean sq. z-score < 0.05 *and* every individual feature within
|z| < a tightening threshold) or plateaued (no improvement for 5 epochs),
instead of a fixed epoch count.

First run immediately surfaced something concrete: **every feature except
`curvature_rms` was already within 0.1 std of human even at the default
config** - the entire distribution gap was one feature, sitting at
`z=-0.78`. The search couldn't fix it because `curvature_scale`'s original
hand-tuned bound (0.0-0.05) was too low to reach the required curvature -
not a search failure, a bound ceiling. Widening `curvature_scale` to
0.0-0.3 and `ou_sigma` to 0.0-8.0 and re-running fixed it almost
immediately (`curvature_rms` z: -0.78 -> -0.09 in one epoch).

Tightening the per-feature convergence bound in stages (0.5 -> 0.3) to make
sure that wasn't a lucky one-feature fix revealed the next wall: pushing
`curvature_rms` up (via `curvature_scale`/`ou_sigma`) drags
`velocity_skewness` the other way (+0.37 -> -0.45, overshooting past human's
own value), and the search plateaus there - 5 epochs with no improvement,
stuck at worst-feature `|z|~0.43` regardless of continued mutation. This
looks like a genuine structural coupling in `motor_synergy`'s current
parametrization (the same knobs that add curvature also reshape the
velocity profile), not a search-power problem.

**Final result** (fresh independent 4-model ensemble, never used during
search):

| config | accuracy (ensemble mean) | accuracy (worst member) | mean sq. z-score vs human |
|---|---|---|---|
| default | 0.998 | 0.979 | 0.089 |
| evolved | 0.980 | 0.965 | **0.033** |

A real, meaningfully larger improvement than v2 - `curvature_rms` went from
the single dominant gap (`z=-0.78`) to nearly matched (`z=-0.21`), and
overall distance to human dropped ~3x. Still not "indistinguishable" (98%
accuracy is very detectable), and `velocity_skewness` is the next thing to
fix - but *why* it's stuck is now a specific, actionable finding rather than
an unexplained plateau.

### v3.1: `peak_time_ratio` + tracking the REAL target directly - the decisive run

Two more changes, on a "구분 못할 때까지" (keep going until it genuinely can't
tell) instruction: `peak_time_ratio` (which controls *where* the velocity
peak lands - the direct lever for profile asymmetry, never exposed to the
search before) was added to the tunable set, and every 3rd epoch the loop
now trains a small ensemble on a genuinely disjoint **holdout** split (not
the search's own ensemble data) and measures real accuracy directly -
tracking the actual target ("can the detector tell?") instead of only the
distribution-distance proxy. Ran 23 epochs to a genuine plateau on *both*
signals at once.

The distribution match got almost perfect:

| feature | default z | evolved z |
|---|---|---|
| curvature_rms | -0.78 | -0.17 |
| velocity_skewness | +0.37 | -0.19 |
| movement_time | -0.22 | -0.18 |
| time_to_peak_ratio | -0.07 | -0.20 |
| *(all 9 features)* | mean sq. z = 0.092 | **mean sq. z = 0.017** |

Every single feature's *mean* landed within 0.2 std of human - a dramatically
better and more balanced fit than v3's single-feature fix. And yet:

| config | accuracy (ensemble mean) | accuracy (worst member) |
|---|---|---|
| default | 0.996 | 0.991 |
| evolved | 0.976 | 0.968 |

Barely moved. The holdout accuracy checked every 3 epochs never dropped
below 0.956 across the entire 23-epoch run, nowhere near the 0.55 target.

**This is the decisive finding, not just another plateau.** Matching every
feature's *mean* almost perfectly did not make the bot meaningfully harder
to detect. The only way that's possible is if the detector is reading
something mean-matching can't touch - the *variance/shape* of each feature's
own distribution, or the *correlations between features* (a HistGradientBoosting
model built on tree splits can easily exploit "high curvature AND low jerk
together," even if each marginal mean matches perfectly). Optimizing 9
independent z-scores is fundamentally a marginal-distribution objective; the
detector operates on the joint one.

This reframes the whole exercise: the earlier "detector is robust" reading
(v1-v3) now has much stronger support than a plateau alone would provide -
a search that achieved near-perfect marginal matching *still* couldn't move
the needle materially, which rules out "we just haven't matched the right
features yet" as an explanation. Any further progress needs a fundamentally
different objective (covariance matching, a true joint-distribution distance
like sliced Wasserstein over the full feature vector) or a more flexible
generator than `motor_synergy`'s current parametrization - not more epochs of
the same evolutionary search.

`results/adversarial_history.png` plots the per-generation trace, the z2
convergence trend, and the holdout accuracy trend together;
`results/evolved_motor_synergy_config.json` is the final (v3.1) evolved
config - still a reasonable manual-tuning starting point (better-balanced
curvature/timing than v3's), just not a solution to the detection problem.

### v4: matching variance too - a real data bug, real generator gaps, and a whack-a-mole plateau

v3.1 proved mean-matching alone was insufficient. The natural next question:
does the *spread* match? A correlation/variance audit (comparing full
covariance matrices, not just per-feature means) found the answer immediately:
motor_synergy_bot's movement-to-movement **variance** was only 2-10% of
human's for curvature/jerk/path_efficiency - every generated movement looked
nearly identical, regardless of how well the *average* matched.

**A real data bug surfaced during the audit and had to be fixed first.**
`path_efficiency`'s human-side standard deviation was corrupted by 2-3
pathological outliers (values up to 64.5x, mathematically impossible for a
proper distance/path_length ratio which is bounded at 1.0) traced to Balabit
rows with `x`/`y` == 65535 (0xFFFF) - a "no cursor position recorded yet"
sentinel, not a real coordinate, that a handful of sessions carry as their
first sample. `parse_balabit.py` now filters these at the source, and
`features.py` clips `path_efficiency` to its mathematical bound as a second
line of defense. This alone dropped the baseline's variance-mismatch score
from 6.99 to 2.81 - the corrupted statistic had been inflating it more than
the real bot-vs-human gap was.

With clean data, `variance_penalty` (mean squared log-variance-ratio,
`log(candidate_std / human_std)`, symmetric under/over-dispersion) was added
to the fitness alongside the existing mean-matching term, weighted equally -
matching the average without matching the spread isn't half a solution, per
v3.1's finding. First run: a genuinely new best result.

| config | accuracy (mean) | accuracy (worst) | mean sq. z (mean) | mean sq. log-ratio (spread) |
|---|---|---|---|---|
| default | 0.989 | 0.985 | 0.202 | 2.772 |
| evolved | **0.959** | **0.944** | 0.094 | **0.145** |

The best accuracy drop of the whole project (curvature_rms's spread went
from 5% of human's to genuinely comparable). But `curvature_scale`'s search
partner, `curvature_noise_sigma`, settled mid-range rather than at its
bound - meaning the remaining gap wasn't a ceiling anymore, it was a genuine
trade-off: **a single Gaussian scale factor can't set "how curvy movements
are on average" and "how much that varies between movements" independently,
since both ride on the same sigma.** Added `curvature_style_sigma`, a second,
separate per-movement lognormal multiplier for exactly the variance half.
Re-running found the *identical* bug on a different axis:
`time_to_peak_ratio`'s spread was stuck the entire run, traced to
`peak_time_ratio`'s per-movement jitter window being hardcoded to `+-0.03`
regardless of what the mean itself got tuned to - pulled out as
`peak_time_jitter`.

Adding both new knobs did NOT produce a further win at the original search
budget (accuracy actually landed slightly worse, 0.975/0.969 - a larger
search space diluted the same population/generation budget). Re-running
with a bigger budget (population 6->10, generations/epoch 3->4) recovered
to 0.959/0.943 - **matching, not beating, the first variance-matching run.**

**The pattern across three iterations is the finding.** Every time one
structural bottleneck in `motor_synergy`'s parametrization got a dedicated
fix, the very next-worst feature turned out to have the *same* underlying
bug (a single random draw doing double duty as both the mean-setter and the
spread-setter) - and the marginal gain from fixing it shrank each time
(curvature: ~3pp accuracy gain; timing: ~0pp net, after a bigger search
budget just matched the prior result instead of improving on it). That's
consistent with approaching a real ceiling for this generative family, not
with the search being underpowered - more population/generations at the
same objective is unlikely to do better than plateau again.

**Consolidated result**: best independently-validated accuracy found across
every run - **95.9% (mean) / 94.3-94.4% (worst)**, down from the untouched
default's 98.9%/98.5%. A real, repeatable, ~3pp improvement - not
"indistinguishable" (that would mean ~50%), but a substantive step, and the
`curvature_style_sigma`/`peak_time_jitter`/`mt_noise_sigma`/
`curvature_noise_sigma` additions are now genuine, reusable knobs (not in
the C++ header yet - see "Suggested next steps").

### v5: joint/covariance matching, and a real distribution-SHAPE fix - the biggest single gain yet

v4's pattern pointed directly at one hypothesis: marginal (per-feature)
mean+variance matching had a real ceiling, and the next rung was the joint
structure between features. Added `covariance_penalty` (mean squared
difference between the candidate batch's own 9x9 feature correlation matrix
and the human one, over all 36 off-diagonal pairs) to the fitness alongside
the existing mean/variance terms, plus matching convergence tracking
(`CONVERGENCE_COV_MAX_ABS_DIFF`) and a `human_corr_matrix()` baseline.

**The covariance hypothesis was wrong.** The correlation-matching term
converged easily and early (within its own target `< 0.35` inside a handful
of epochs) while overall detector accuracy stayed exactly where v4 left it -
**0.955/0.949, statistically the same as v4's 0.959/0.943.** The joint
structure was never the bottleneck; adding the term just spent search budget
confirming that.

The per-epoch convergence log made the *actual* remaining bottleneck obvious
instead: `time_to_peak_ratio`'s variance was stuck at the same place it was
in v4 (log-ratio around -1.1 to -1.3, i.e. the bot's spread in *when* the
speed peak lands is still only ~30% of human's), unmoved regardless of how
wide `peak_time_jitter` was allowed to search. That's not a bound problem -
it's a **distribution-shape** problem: `peak_time_jitter` drew the peak's
timing offset from a `random.uniform(-jitter, +jitter)`, which has a hard
cutoff at the bound no matter how wide the bound is. Human sub-movement
timing evidently has occasional real outliers - a bounded uniform draw
structurally cannot produce those at *any* width.

Fixed by switching that one draw from uniform to Gaussian
(`generate_synthetic.py`'s `motor_synergy_generate`: `peak_frac =
peak_time_ratio + gauss(0, peak_time_jitter)`, clipped to `[0.05, 0.95]` of
the movement time only for physical sanity, not to bound the search). Same
tunable parameter, same search space - just an unbounded tail instead of a
hard cutoff. Re-running the identical search on top of this one change:

| config | accuracy (mean) | accuracy (worst) | mean sq. z (mean) | mean sq. log-ratio (spread) | mean sq. corr diff (joint) |
|---|---|---|---|---|---|
| default | 0.993 | 0.989 | 0.200 | 2.499 | 0.108 |
| evolved | **0.922** | **0.905** | 0.007 | 0.088 | 0.039 |

**The single biggest accuracy drop of the whole project** - worst-case
accuracy from 94.3% down to 90.5%, more than double v4's ~3pp gain, from one
distribution-family change to an existing parameter. `time_to_peak_ratio`'s
spread log-ratio improved from -2.00 (default) to -0.81 (evolved) - still
not fully closed, but the biggest single-run movement on that axis across
every version.

**A widen-the-bounds test regressed, and was reverted.** The evolved config
pinned three parameters at their search-bound ceiling
(`ou_sigma`=8.0/8.0, `primary_sigma_max`=0.5/0.5, `peak_time_jitter`=0.10/0.10)
- the same signal that correctly justified widening `curvature_scale`/
`ou_sigma` back in v3. Widening all three again here
(`ou_sigma`->16, `primary_sigma_max`->0.9, `peak_time_jitter`->0.30) and
re-running the identical search produced a *worse* result (0.958/0.941,
`curvature_rms` overshooting from too-low to too-high) - the same
"more search space, same budget, net regression" failure mode already seen
once in v4. All three bounds were reverted to the values that produced the
92.2%/90.5% result above, which is the config now checked into
`results/evolved_motor_synergy_config.json`.

**Consolidated result (superseded by v7 below)**: **92.2% (mean) / 90.5%
(worst)**, down from the untouched default's 99.3%/98.9% - a genuine ~7-9pp
improvement, on top of v4's ~3pp, achieved not by tuning existing config
values harder but by fixing an actual distribution-family bug (uniform vs.
Gaussian jitter) that no amount of bound-widening or search budget could
have found on its own.

### v6-v11: auditing the rest of the generator, a fitness-function blind spot, and a real distribution-shape wall

Given v5's single biggest win came from fixing ONE hardcoded uniform draw,
the obvious next move was auditing the rest of `motor_synergy_generate` for
the same bug. What followed was six more version bumps, most of which
**failed** in informative ways - the honest record, not just the wins:

**v6 - same fix, applied everywhere else, regressed.** The reach fraction
(overshoot/undershoot) and every correction-submovement timing/amount draw
were still hardcoded uniform windows with zero exposed scale. Converted all
of them to the same Gaussian-offset pattern (`reach_jitter`,
`correction_timing_jitter`, `correction_amount_jitter`), bumped the search
budget (10/3/4 -> 14/4/5) to match the larger space. Result: **worse**
(0.944/0.929 vs v5's 0.922/0.905), even though the search's own tracked
metrics (z-score/log-ratio/correlation) converged *tighter* than v5 ever had.

**The regression was a fitness-function blind spot, not a bad fix.**
`DIST_MATCH_FEATURES` only tracked 9 of the 14 features the detector
actually trains on (`shape_only`). The search was free to let
`num_submovements`, `velocity_kurtosis`, `tremor_band_energy_ratio`, and
`sdn_correlation` drift arbitrarily while over-optimizing the tracked 9 -
and a fresh detector picked up on exactly that drift. Expanding
`DIST_MATCH_FEATURES` to all 13 relevant features (**v7**) recovered the
regression and then some: **92.7% (mean) / 89.2% (worst)** - the new best,
achieved not by a better generator fix but by making the search's own
objective see what the detector actually sees.

**v8 - re-testing bound ceilings under the fuller objective: still a dead
end.** `ou_sigma`, `mt_noise_sigma`, and `peak_time_jitter` all pinned at
their bounds again under v7's 13-feature fitness. Re-widened all three,
reasoning the earlier (v5) regression might have been the 9-feature blind
spot's fault rather than the widening itself. It wasn't - the wider search
landed on an even worse, more extreme corner (`ou_sigma`=16.0,
`gamma_shape`=8.0, `sample_dt_mean` at its floor, `tremor_amp_max`=0.0 all
at once) and regressed to 0.941/0.916. Reverted for good; a parameter
pinning at its bound is not reliable evidence the bound is wrong, evidently,
once several OTHER fixes have already changed what "wrong" means.

**v9 - more samples per candidate for a less noisy fitness signal: a wash.**
150 -> 250 samples/candidate, everything else identical to v7. Result
(0.920/0.901) within run-to-run noise of v7 - not a lever worth its extra
compute cost.

**v10/v11 - matching distribution SHAPE, not just its first two moments:
correctly diagnosed, still didn't pay off twice.** A `permutation_importance`
check on a real trained ensemble (not a guess) found `jerk_rms` is by far
the single most-relied-on feature (0.18, next is `time_to_peak_ratio` at
0.11) - despite its mean and variance both already looking "matched."
Comparing the raw distributions explained why: human `jerk_rms` has
skew=3.83 (heavy right tail) vs the bot's 2.92, and the *median* differs 2x
(0.4 vs 0.2) even though the *mean* happens to match - a few large bot
outliers pull the mean up while the bulk sits well below human's. Added a
third-moment (skewness) matching term (v10): regressed to 0.959/0.940.
Suspecting budget starvation (the same failure mode v4 and v6 both hit
before their own budget bumps fixed them), reran with the population/
generation budget nearly tripled (v11, 14/4/5 -> 20/5/6, patience 6 -> 9):
still worse than v7 (0.922/0.912). Two independent tests, one of them with
3x the search cost, both confirming the diagnostic was *right* about what
the detector uses but the *fix* (a raw penalty term) doesn't translate into
a real win within reasonable search budget - disabled (`SKEW_MATCH_WEIGHT =
0.0`, code kept for the diagnosis's sake, not deleted) and reverted to v7.

**Final consolidated result for this project**: **92.7% (mean) / 89.2%
(worst)**, down from the untouched default's ~99.0%/98.6% - a genuine
~7pp/~9.4pp reduction from six version's worth of real structural changes,
three of which (v5's Gaussian jitter, v7's expanded fitness target) produced
real, validated gains, and four of which (v6's premature win, v8's bound
re-widening, v9's sample increase, v10/v11's skew matching) were genuine,
well-motivated attempts that didn't pay off - all left in the record above
rather than quietly erased, per this project's own stated goal of an honest
account over a flattering one.

## Beyond motor_synergy: data-driven trajectory generation (the real breakthrough)

v1-v11 (above) spent eleven versions tuning and structurally patching
`motor_synergy`'s hand-derived parametric formula (one lognormal primary
submovement + up to 2 corrections + additive OU/tremor/SDN noise), reaching
a **plateau around 89.2% (worst) / 92.7% (mean)** - a real ~9pp improvement,
but stuck well short of any "genuinely fools the detector" target. The open
question this section answers: is 89% a property of *this specific
generative family*, or of the detector/dataset itself (i.e. is anything
close to indistinguishable even possible here)?

### Establishing there's real headroom: two ceiling checks

**`feature_space_ceiling.py`** - the cheapest possible test. Fit a Gaussian
Mixture Model directly on the human data's 14 extracted *features*
(skipping trajectories entirely - not physically real, just "how good could
detector accuracy get if the joint feature distribution were matched
perfectly"), sample synthetic feature vectors from it, evaluate the same
detector pipeline. Result: **72.8% (mean) / 69.4% (worst)** - meaningfully
below motor_synergy's 89.2%, proving real headroom exists, but also
revealing that even an idealized generator likely can't reach true 50/50
with this detector/dataset - a genuinely useful calibration for how far to
expect to push.

**`trajectory_gmm_ceiling.py`** - the same idea, but with genuine *physical*
trajectories: canonicalize each human movement (rotate/scale to start at the
origin, end at `(1, 0)`), resample to a fixed number of points, fit a GMM on
that fixed-dimension representation, sample synthetic canonical shapes,
un-normalize (random direction, sampled distance/duration), extract features
from the resulting real `(x, y, t)` trajectory. First attempt (**v1**, 24
points, direct reconstruction): **85.5% (mean) / 82.8% (worst)** - better
than motor_synergy, a real result. Second attempt (**v2**, 64 points +
PCA dimensionality reduction to keep the GMM well-conditioned): **90.6% /
88.9% - WORSE**. Diagnosis: PCA keeps the directions of maximum *position*
variance, which is dominated by the smooth bulk path shape; it discards
exactly the small-amplitude, high-frequency jitter/tremor detail that
`jerk_rms` (this whole project's single most detector-relied-on feature,
by a wide margin, every time it's been checked) depends on. Lesson: more
resolution isn't free if the dimensionality-reduction step throws away the
signal that matters most.

### The hybrid approach, and four real bugs

The natural next idea: use the GMM for the *smooth macro shape* (which
genuinely benefits from being learned from data) and motor_synergy's
already-validated noise mechanisms (OU-jump, tremor, SDN - the exact fixes
that drove v5's big win) for the *fine jerk/tremor detail* the GMM's coarse
control points can't carry on their own. Building this
(`trajectory_hybrid_ceiling.py`, then `hybrid_noise_search.py` to properly
evolve the noise parameters instead of reusing motor_synergy's own values)
surfaced four real, non-obvious bugs - found by refusing to trust a result
that looked too good or too bad without checking why:

1. **CubicSpline overshoot.** Interpolating between only 24 sparse GMM
   control points with a cubic spline, then evaluating it at a much finer
   time grid, produces ringing/overshoot between the control points -
   inflated jerk into an obvious tell (a "zero-noise" baseline came back
   *perfectly* distinguishable, worst=1.000). Fixed by switching to plain
   linear interpolation (`np.interp`), matching what the already-validated
   `trajectory_gmm_ceiling.py` v1 does.
2. **Timestamp clamping.** The fine-grained sample-time generator could
   overshoot `movement_time` by design (mirroring motor_synergy's own
   +15ms buffer), then clamped the *position* lookup to the endpoint while
   leaving the *stored timestamp* unclamped - the cursor visibly "froze" for
   the last sample or two while time kept climbing, a tell no real
   trajectory has. Fixed by clamping the timestamp itself when it's stored,
   not just when it's looked up.
3. **The big one: `GaussianMixture.sample(1)` called repeatedly returns the
   IDENTICAL sample every time.** sklearn's `.sample()` re-derives its RNG
   from `self.random_state` on *every call* - with a fixed `random_state=0`
   (needed for reproducibility), calling `.sample(1)` fresh per movement
   returns the same draw every single time. Every "different" generated
   movement was actually the same shape (only the random output-direction
   rotation differed) - rotation-invariant features (`path_efficiency`,
   `curvature_rms`, `time_to_peak_ratio`, ...) came out with essentially
   zero movement-to-movement variance (std as low as 0.000 vs human's
   0.1-0.45) - an even easier tell than bugs 1-2. Fixed by batch-sampling
   all needed vectors in ONE `gmm.sample(n)` call per candidate, matching
   the pattern `trajectory_gmm_ceiling.py` already used correctly (which is
   why its v1/v2 results were never affected by this).
4. **Missing shuffle after batch sampling.** While trying to fix a smaller,
   real issue (see below), added rejection sampling on top of the
   now-correct batch call - and made things *worse*. Cause:
   `gmm.sample(k)` returns samples grouped by mixture component (documented
   sklearn behavior, not random order); filtering then truncating to the
   first `n` valid rows can consume the entire quota from just the first
   few components before ever reaching the rest of the mixture. Fixed by
   shuffling the valid pool before truncating.

### Controlled A/B tests that correctly ruled out two plausible fixes

With the pipeline now behaving correctly, `permutation_importance`
consistently flagged **`path_efficiency`** as the new #1 discriminator
(~0.24 importance - human median path_efficiency 0.928, generated median
~0.79-0.80: the raw GMM-sampled shape wanders more than real human paths).
Two plausible fixes were tried, both **properly validated with multi-seed
A/B tests before being trusted** (this project had already been burned once
by drawing conclusions from single noisy runs - the evolutionary search
here shows real run-to-run variance, 0.72-0.86 worst-case across nominally
identical settings, so single-run comparisons are not reliable evidence):

- **Rejection sampling on implied speed** (to fix a related `mean_speed`
  near-zero-speed tail: bot's 1st percentile was 19 px/s vs human's 411).
  Regressed the independent-validation result both times tried (200 px/s
  and a gentler 50 px/s threshold) - post-hoc filtering on speed distorted
  the `(distance, movement_time)` joint relationship enough to inflate
  `mean_speed`'s *upper* tail instead. Disabled.
- **Moving-average smoothing of the GMM shape** (to fix `path_efficiency`
  directly). A single test looked promising, but a **clean 3-seed A/B**
  (window=1 vs window=3, `FIXED_SMOOTH_WINDOW` pulled out as a fixed,
  not-evolved parameter specifically so it could be tested this way) showed
  window=3 was *consistently worse* across all 3 seeds (worst-case 0.852,
  0.870, 0.865 vs window=1's 0.835, 0.853, 0.830 - no overlap). Smoothing
  fixes `path_efficiency` in isolation but costs enough elsewhere (almost
  certainly `jerk_rms`, which it also flattens) to be a net loss. Reverted.

### The actual breakthrough: BIC was optimizing the wrong objective

Every GMM fit up to this point selected its number of components via BIC
(Bayesian Information Criterion) from candidates `(5, 10, 15, 20, 30)` -
standard, textbook-correct practice for density estimation, and it always
picked 5. But BIC is designed to find the *best-generalizing* density
estimate, penalizing complexity specifically to avoid overfitting to the
training sample. That is **not this project's actual objective.** The goal
here is "produce synthetic data a classifier can't tell apart from the
training distribution" - and a much higher-complexity GMM (approaching one
component per training data point, each with a small covariance "jitter
ball") stops being a smoothed few-cluster approximation and starts closely
tracking the *real empirical joint distribution's fine structure* -
including exactly the fine-grained joint correlations (across all 14
features, not just marginals) that low-k GMMs, PCA, and every marginal/
covariance-matching fitness term across the entire v1-v11 motor_synergy
effort were all straining to approximate by hand.

Verified directly via **independent validation** (a human split never seen
during GMM fitting), monotonically improving as component count `k`
approaches the training sample count (1200), and confirmed to actually
matter across multiple search seeds, not a fluke of one run:

| `k` (of 1200 training points) | worst-case accuracy (independent validation) |
|---|---|
| 5 (old BIC-selected default) | 0.736 |
| 100 | 0.679 |
| 200 | 0.614 |
| 900 | 0.568 |
| 1100 | 0.517-0.524 |
| 1140 (0.95x, seeds 42/1/7/123) | 0.616 / 0.616 / 0.618 / 0.603 (avg ~0.61) |
| **1176 (0.98x, seed 42)** | **0.598** |

`fit_shape_gmm` in `hybrid_noise_search.py` now uses `n_components =
int(0.98 * len(training_points))` instead of BIC selection. Additive noise
(OU-jump/tremor/SDN, evolved on top via the same fitness machinery as
`adversarial_loop.py`) does **not** improve on the GMM-shape-alone result at
this component count - consistent with the earlier smoothing finding: once
the shape itself is genuinely learned from real data at high enough
fidelity, it already carries the natural variability real noise mechanisms
were built to compensate for; adding more on top doesn't help and sometimes
mildly hurts.

**Final consolidated result**: **~64% (mean) / ~60% (worst)**, down from
the untouched default's ~98-99% and from motor_synergy's own best of 89.2%/
92.7% - inside this project's target range, achieved not by tuning
`motor_synergy`'s formula further but by recognizing that a *data-driven*
generator, given enough capacity to actually represent the real empirical
distribution (not the BIC-optimal *smoothed* one), closes most of the
remaining gap to the idealized feature-only ceiling (69.4%).

## Scope cuts (explicit, not hidden)

- Full iterative sigma-lognormal decomposition (XZERO/iDeLog) - BeCAPTCHA-Mouse's
  own feature extraction method is a research project in its own right;
  this project uses simpler features inspired by the same papers instead.
- A full neural generative model (GAN/normalizing flow/diffusion) on raw
  trajectories - a high-component GMM turned out to be enough to reach this
  project's target; a neural model remains a natural next step if more
  headroom is ever needed (see "Suggested next steps").
- A more sophisticated search algorithm (CMA-ES, larger populations/more
  generations) for the noise-evolution layer - turned out not to be the
  lever that mattered once the GMM shape itself was fixed.

## Suggested next steps

The GMM-shape breakthrough (see above) reached this project's ~50-60%
worst-case target, but the underlying methodological lesson generalizes
further than the specific number:

1. **The BIC-vs-actual-objective mismatch is probably not unique to this
   GMM.** Anywhere a model-selection criterion optimizes "generalizes well"
   by penalizing complexity, and the actual goal is "matches the empirical
   distribution closely enough to fool a discriminator," that criterion is
   fighting the real objective. If pursuing this further, replace BIC/AIC
   selection with a validation-set proxy that's actually aligned with the
   goal (e.g. sweep `k` and pick by held-out detector accuracy directly,
   rather than by density-estimation quality).
2. **A held-out human split is small (currently 800 movements) relative to
   how confidently "0.5-0.6" should be trusted** - the 4-seed spread
   (0.598-0.618) suggests the true value could plausibly dip further below
   0.6 or sit a bit above it depending on which exact held-out sample is
   drawn. Collecting more Balabit-derived movements (or finally getting real
   sessions into `human_movements_web.jsonl` via the deployed
   aim-rl-web-collector) would tighten this estimate.
3. **This generator is NOT a drop-in replacement for `motor_synergy` in the
   live aimbot.** A ~1150-component GMM sampled at runtime is a very
   different latency/memory profile than motor_synergy's closed-form
   lognormal-CDF math, which was specifically built with a CPU/GPU split for
   real-time use. If this line of work is ever meant to inform the live
   aimbot (not just this offline research question), the right next step is
   distillation: use what THIS generator gets right (the joint structure a
   handful of hand-tuned noise parameters couldn't capture) to identify
   which specific structural fix `motor_synergy.hpp`'s formula still needs,
   the same way v5's Gaussian-jitter fix was found - not to run the GMM
   itself inside the real-time path.
4. If more headroom is wanted beyond ~0.5-0.6: a genuine neural generative
   model (small normalizing flow or GAN, trained the same canonicalized-
   trajectory way) is the natural next escalation now that a GMM has been
   shown to work at all - a flow model doesn't have the "one Gaussian per
   point" degeneracy risk at high capacity and can represent the tails more
   smoothly.
5. Before adopting any evolved config in the live aimbot: sanity-check it by
   hand in `needaimbot/mouse/motor_synergy.hpp` / `simple_config.json`'s
   `flick_*` keys - a search optimized purely for statistical similarity can
   drift somewhere that looks or feels odd in practice. Note the new
   `reach_jitter`/`correction_timing_jitter`/`correction_amount_jitter`
   params and `peak_time_jitter`'s changed (Gaussian std, not half-width)
   semantics aren't in the C++ header yet.

---

# Update — the strong-detector reckoning, neural generators, and the replay ceiling

Everything above stops at the ~64%/60% GMM result and a "Suggested next steps"
list. This section is what happened when those steps were actually carried out
(neural generator, better detectors, controls) — and it reframes the whole
project.

## The 60% was one-sided

The GMM breakthrough's ~60% worst-case was measured against the same fixed
detector the generator had been tuned against. Re-evaluating against a **fresh,
independent strong detector** — `train_detector.py`'s full RandomizedSearchCV
across RandomForest / GradientBoosting / HistGradientBoosting / SVM-RBF, never
exposed to the generator (`validate_gmm_bot_strong_detector.py`) — tells a
different story:

| feature set | GMM-hybrid bot |
|---|---|
| shape_only | **0.855** |
| all | 1.000 |

So the honest adversarial number is ~0.86, not 0.60. The 60% was the classic
one-sided arms race: the generator overfit one detector instance.

## Genuine co-evolution doesn't break it

`co_evolution_loop.py` alternates — evolve the generator against the current
detector ensemble, then re-tune the detector against the new generator, repeat.
Across 4-5 rounds (48-point trajectories, widened bounds, seed sweeps) the
pattern was flat:

- in-loop evolved worst-case: ~0.57-0.61 (the generator *does* fool the current detector)
- freshly re-tuned detector: **0.82-0.85 every round**, no downward trend

A permutation-importance diagnosis (`diagnose_gmm_bot_tell.py`) showed why: every
feature's marginal z-gap was already < 0.35 (means matched), and importance was
spread thin across many features — a **joint-distribution / generator-family
ceiling**, not a single fixable tell.

## Neural flow: the same wall

If the ceiling is the GMM's smoothed density, a neural flow should do better — it
learns one smooth invertible map instead of ~one Gaussian per training point.
Built a RealNVP (ActNorm + affine coupling) on canonicalized 48-point strokes,
trained on 8k-14k human movements on a GPU (`flow_generator.py`):

| generator | strong-detector shape_only |
|---|---|
| GMM | 0.855 |
| flow (8k) | 0.812 |
| flow (14k, stabilized) | 0.860 |

The flow lands at **0.81-0.86 — statistically the same as the GMM**, well inside
this project's known run-to-run variance. A neural generator does *not* beat the
high-k GMM. Both data-driven families hit the same wall.

## Adversarial and blend both fail (informatively)

- **Adversarial flow-GAN** (`flow_adversarial.py`): fine-tune the MLE flow to
  fool a neural discriminator, re-initialized every 250 steps to force
  generalization. It drove the neural discriminator to ~0.51 (fooled) while
  anchoring realism via the flow's exact NLL — yet the independent sklearn strong
  detector still scored **0.826**. Adversarial examples don't transfer, exactly
  as the v1-v3 evolutionary search already found.
- **kNN blend** (`knn_blend_generator.py`): combine nearest-neighbor real strokes
  into "new" trajectories. Result **0.889 — worse than the flow**. Convex
  blending averages away the high-frequency jerk/tremor, producing unnaturally
  smooth strokes. Lesson: **"combine" doesn't work, "warp" does** — the
  difference is preserving the fine micro-structure.

## The decisive controls: 50% is only reachable by replay

Three control experiments pin the real floor:

| control | strong-detector shape_only |
|---|---|
| **human vs human** (same distribution, one half mislabeled "bot") | **0.500** |
| **replay** (real stroke + 1px jitter) | 0.522 |
| **warped replay** (real stroke rotated/scaled to an arbitrary target) | **0.506** |

human-vs-human at 0.500 (`validate_human_floor.py`) proves two things at once: the
detector is honest (it can't split two samples of the same distribution, so it
isn't overfitting), *and* 50% is achievable in principle. And the only things
that reach it are **real strokes replayed**, not generated — including
`warped_replay_generator.py`, which rotates/scales a real stroke onto an
arbitrary target (what an aimbot actually needs) and still sits at 0.506.

## The full spectrum

| method | strong shape_only | what it is |
|---|---|---|
| motor_synergy | 0.99 | hand-tuned formula |
| kNN blend | 0.889 | averaged real strokes (smoothing kills it) |
| GMM | 0.855 | learned density |
| neural flow | 0.81-0.86 | RealNVP, MLE + adversarial |
| **warped replay** | **0.506** | real stroke, rigid-transformed to target |
| replay | 0.522 | real stroke, fixed direction |
| human vs human | 0.500 | the floor |

**Conclusion.** No *generalizing* generator — hand-tuned, GMM, or neural — beats
~0.85 against a strong adaptive detector on this feature/data setup. Every earlier
sub-0.85 number was measured against a weak or fixed detector. Genuine ~0.50 is
reachable only by **replaying real human motion**, which is memorization, not
generation. This independently reproduces **SCRAP** (Synthetically Composed Replay
Attacks vs. Adversarial ML Attacks against Mouse-based Biometric Authentication,
ACM AISec 2020): domain-knowledge replay beats adversarial ML against mouse
biometrics and is harder to defend.

## Applied: swapping needaimbot's flick generator

Because warped replay is the only thing that actually defeats the detector, it was
ported into the companion `needaimbot` aimbot
(github.com/needitem/needaimbot, `2pc` branch), replacing the `motor_synergy`
acquisition-flick generator outright:

- `scripts/export_flick_db.py` exports straight, low-lateral-deviation human
  strokes to `flick_trajectories.json`.
- The aimbot's `warped_replay.hpp` picks a distance-matched stroke at runtime,
  rotates/scales it onto the aim vector, and replays it with its real timestamps —
  CPU-only (~1.6 µs/flick), the old GPU flick kernel removed.
- The PD controller's synthetic "human-like" tracking noise (SDN + tremor) was
  removed too — redundant once the flick is genuine human motion.

## Defending against replay

Warped replay is undetectable *from static point-to-point trajectory statistics* —
that is the finding, not a gap in the classifier (a real stroke replayed *is*
human, so a mouse-dynamics classifier correctly passes it). It is still beatable
at other layers: **near-duplicate detection** (a finite stroke pool repeats — only
works if the defender has the pool, so build the DB from your own recordings),
**session-level dynamics consistency** (one account mixing many people's motion
fingerprints), and most robustly **challenge-response** (unpredictable / moving
targets that a pre-recorded stroke can't react to). Static aim-trajectory
statistics alone cannot cover replay — that is a property of the signal.

---

# The arms race: attacker vs. defender, played to convergence

The sections above establish that a single warped-replay flick is
indistinguishable from a real one (0.506). That is not the end — it is the
opening move. Iterating the attacker and the defender against each other,
each answering the other's last move, converges to a clear winner. Every row
below is a script in `scripts/` and a number in `results/`.

| # | move | result |
|---|---|---|
| 1 | **defender:** single-movement strong detector | warped replay **0.506** — evades |
| 2 | **defender:** + near-duplicate, set-level (`detect_replay_reuse.py`) | finite pool repeats caught; real-vs-real min distance 0.084 ≫ eps 0.059, **FP 0.000** |
| 3 | **attacker:** gaussian jitter to break repeats (`detect_replay_dilemma.py`) | dilemma — small jitter → reuse-caught, large jitter → single-move-caught |
| 4 | **attacker:** human-variability perturbation (`attack_manifold_perturb.py`, `attack_sweet_spot.py`) | mag≈0.07 evades **both** at once (single 0.54, reuse 0.00) |
| 5 | **defender:** session nearest-neighbor **distribution** (`detect_replay_cluster.py`) | source clusters survive perturbation → caught at N≥500 (1.000 at N=1000) |
| 6 | **attacker:** no-repeat, and drop the perturbation (`attack_norepeat.py`) | pure replay + distinct sources → session detector back to **~0.5** |
| 7 | **defender:** long-horizon, N>K (`detect_longhorizon_crossaccount.py`) | finite pool exhausts → forced reuse → **1.00** once N>K |
| 8 | **attacker:** enlarge K with more people | — |
| 9 | **defender:** per-account **style consistency** (`detect_style_consistency.py`) | multi-person pool → within-account style variance → 0.83 (2 people) … 1.00 (10) |
| 10 | **attacker:** 1-person pool (consistent) or rotate accounts | 1-person → small K → caught by #7; rotation → caught by #11 |
| 11 | **defender:** cross-account linking (`detect_longhorizon_crossaccount.py`) | shared pool → same stroke across accounts → 0.10 (2 accts) … 0.86 (20); real **0.000** |

## Winner: the defender — under aggregate observation

The decisive asymmetry: an aimbot's attacker **cannot record in real time**
(the bot is doing the aiming), so they are stuck with a **finite, pre-recorded
pool**. Every way to hide that finiteness leaves a different structural trace:

- reuse within a session → near-duplicate / session-distribution
- exhausting the pool over time → long-horizon (N > K)
- enlarging the pool with many people → per-account style variance
- rotating accounts on a shared pool → cross-account near-duplicates
- using a public dataset → nearest-DB matching

The attacker's only escape is a **fresh, private, single-person pool per
account, never observed past its size** — i.e. recording a new person's strokes
for every account and retiring each account early. That is an unbounded cost,
so the defender wins the aggregate game.

## The honest caveat

This is a win for the *defender who aggregates* — across a session's length,
across time, across accounts. **In a single short session, the attacker still
wins**: a handful of no-repeat warped-replay flicks is genuinely
indistinguishable from a human (0.5). Real-time, per-flick detection does not
work; the defender's leverage is entirely in observing enough. That is why the
robust production answer remains **challenge-response** (unpredictable / moving
targets a pre-recorded stroke can't react to), which defeats replay at the
signal level rather than the statistics level — no aggregation required.

## Can the attacker escape without infinite data? No — the information limit

The defender's aggregate win rests on the attacker's pool being **finite**. The
obvious escape is to manufacture unlimited diversity from a finite pool. Every
way of doing that was tried, and they all hit the same wall
(`attack_segment_compose.py`, and earlier `knn_blend_generator.py` /
`attack_manifold_perturb.py`):

| method | near-duplicate (diversity) | single-move (on-manifold) |
|---|---|---|
| pure replay | fails (finite → repeats) | **passes** (0.5) |
| kNN blend (average) | passes | **fails** 0.889 (averaging kills jerk) |
| human-var perturbation | passes | leaves a session-distribution trace |
| **segment composition** | **passes** (0.002 even at K=50) | **fails** 0.73→0.98 (the seam is the tell) |

The trade-off is fundamental: **breaking near-duplicates (diversity) and passing
single-move (staying on the human manifold) cannot both hold with finite data.**
Pure replay is on-manifold but finite (repeats); anything that manufactures
diversity — averaging, perturbing, or stitching segments — steps off the
manifold and the single-movement detector catches it. Having "unlimited
diversity AND perfectly human" would require either infinite real data or a
perfect generative model — and the perfect generative model does not exist on
finite data.

### The generative model is not the missing piece (any architecture plateaus at ~0.85)

The obvious objection is that a *better* generator would fool the single-move
detector. It doesn't — the ~0.85 wall is architecture-independent, so it is the
finite-data generalization gap, not a modeling weakness
(`diffusion_generator.py`):

| generator | training data | strong-detector shape_only |
|---|---|---|
| high-k GMM (density model) | 8k | 0.855 |
| RealNVP flow (MLE + adversarial) | 8k | 0.81–0.86 |
| DDPM diffusion | 8k | 0.862 |
| DDPM diffusion | 16k (2× data) | 0.848 |
| **DMTG** (entropy-controlled diffusion, published SOTA) | **1,000,000** | **0.87–0.91** |

A classical density model, a normalizing flow, and two diffusion models — three
generations of generative modeling — all land in the same 0.85–0.91 band, and
doubling the data (8k→16k) barely moves it (0.862→0.848). The published SOTA,
**DMTG** (Entropy-Controlled Diffusion, arXiv:2410.18233), trains on **one
million** trajectories and still reports 87–91% white-box discriminator accuracy
— its headline "4.75–9.73% reduction" is relative to GAN/SapiAgent baselines at
~99%, not an absolute escape. So the ~0.85 ceiling is architecture-, capacity-,
AND data-independent: it's the finite-data generalization gap, not a modeling
weakness. A *generative* attacker buys unlimited diversity (defeating every
reuse/aggregate detector) but pays ~0.85 single-move, versus replay's
0.5-but-finite — and no model, no matter how large, moves that price.

### The ceiling above the ceiling: this is only the trajectory layer

All of the above — replay, generation, the whole arms race — lives at **layer 1:
is this *trajectory* human-like?** That is a *necessary* condition, not a
sufficient one. A real interaction is a stack: reaction time to a stimulus,
aim imperfection and correction, session-level variation (fatigue, misses),
cross-modal consistency (mouse + keyboard + view), and causal response to a
moving/unpredictable target. DMTG hits the same wall for the same reason — it is
a path generator, not an *intent* generator. A defender who checks "did this
motion arise from a genuine, in-context user intention on this screen right now"
(DOM causality, session flow, challenge-response) is playing a different game
that trajectory statistics cannot reach. Winning layer 1 (which this project
maps out completely) is table stakes; layers 2–6 are where a real
attacker/defender contest is actually decided, and they are out of scope here.

**So the honest bottom line:** without more real data, the attacker cannot win
the trajectory-statistics game against an ideal aggregating defender. What the
attacker *can* do is (1) pure no-repeat replay so every individual session is
clean, (2) a large finite pool plus per-account usage caps and rotation to stay
under a real anti-cheat's (limited) aggregation horizon, and (3) rely on systems
that don't deploy challenge-response. It becomes an economics game — make
detection cost more than it's worth — not a game the attacker can win outright.

---

# The mechanistic frontier: does a motor-control model escape the wall? (No)

Every generator above is DATA-DRIVEN — it learns a distribution from finite
samples and pays the ~0.85 generalization gap. That left one genuine escape
hatch untried, the one this project scope-cut at the very start: a MECHANISTIC
model that doesn't *learn* the human-motion manifold but *implements* the
neuromuscular process that generates it (finite data then only calibrates a few
physical parameters, so the gap that caps learned models should mostly vanish).

The canonical model for exactly this domain is **Plamondon's Kinematic Theory of
Rapid Movements** — the sigma-lognormal (Σ-Λ) model BeCAPTCHA-Mouse is built on.
A rapid aimed movement's velocity is the vector sum of K lognormal impulse
responses:

    v(t) = Σ_i D_i · Λ(t; t0_i, μ_i, σ_i) · [cos φ_i(t), sin φ_i(t)]

`sigma_lognormal_generator.py` fits this (K=1..3, nonlinear least squares on the
velocity profile) to real human strokes, learns the joint distribution of the
fitted impulse parameters, samples new parameter sets, and re-synthesizes. Two
further variants (`sl_fit_cache.py` parallel-fits the pool once; `sl_synth.py`
iterates synthesis): **SL + real fit-residual** (paste the REAL neuromuscular
micro-texture — the detector's #1 tell — onto the synthetic macro, rescaled to
local speed) and **SL parameter interpolation** (blend two real fits in the
physically-meaningful parameter space, an on-manifold analogue of kNN blend).

| variant | strong-detector shape_only | all |
|---|---|---|
| pure sigma-lognormal | 0.883 | 1.000 |
| SL + real fit-residual | 0.885 | 1.000 |
| SL param-interpolation + residual | 0.879 | 1.000 |

**All three land in the same ~0.88 band — slightly WORSE than the high-k GMM's
0.855.** Adding real (not synthesized) neuromuscular residual didn't move it;
interpolating in the neuromuscular parameter space didn't move it. SL's own
jerk_rms overshoots human (0.55 vs 0.33) and `all`=1.000 shows the absolute
geometry still separates trivially — consistent with v3.1's proof that matching
marginals can't move a joint-distribution detector.

**Why this matters.** The ~0.85 wall was previously demonstrated only for learned
generators, leaving open "a mechanistic model might escape." It doesn't. The wall
now spans BOTH learned (GMM, flow, diffusion, DMTG) AND mechanistic
(sigma-lognormal) generation. That is strong evidence it is a fundamental
property of matching a finite-data manifold's fine structure — not a weakness of
any single method family. Genuine ~0.50 remains reachable only by replaying real
human strokes. The productive frontier is therefore not a better trajectory
generator (the trajectory layer is closed) but the behavioral layers above it
(reaction, evoked tracking, challenge-response) — which is what the companion
aim-rl-web-collector now gathers real human data for.

---

# Latent-space interpolation (GPU): the last distinct idea — best generator yet, wall still stands

The one qualitatively new idea left after the mechanistic frontier: instead of
sampling a learned prior (flow/GMM/diffusion → 0.86) or blending in POSITION
space (kNN blend → 0.889, averaging kills the jerk), blend in the flow's LEARNED
LATENT space. A RealNVP flow "unfolds" the curved data manifold into a Gaussian
latent, so a straight line between two REAL strokes' latent codes
(z=a·f(x1)+(1-a)·f(x2), decode f⁻¹(z)) should stay ON the manifold when decoded —
novel (not a near-duplicate) yet human (unlike position-blend).

Trained on an RTX 3060 (`latent_interp_generator.py`, 16-layer RealNVP, 2000
epochs; the fixed canonicalization endpoints — point0=(0,0), point_last=(1,0) —
must be dropped from the modeled vector or ActNorm's exp() conditioning blows the
gradient up to ~1e10). Both modes scored by the same strong detector:

| variant | strong-detector shape_only | all |
|---|---|---|
| **latent interpolation** | **0.843** | 1.000 |
| latent prior (baseline) | 0.863 | 1.000 |

**Latent interpolation (0.843) is the lowest non-replay generation number in the
whole project** — below GMM (0.855), flow-prior (0.863), and kNN blend (0.889).
The hypothesis was directionally right: interpolating in the unfolded latent stays
more on-manifold than position-space blend (jerk 0.228 vs blend's smoothing, path
efficiency 0.942). **But 0.843 is not 0.50** — it is within run-to-run noise of
the GMM and still squarely inside the ~0.85 band, and `all`=1.000 shows the
absolute geometry separates trivially.

**This closes the trajectory layer.** The ~0.85 wall now spans every distinct
approach tried — hand-tuned (0.99), high-k GMM (0.855), RealNVP flow prior (0.86),
DDPM diffusion (0.86), DMTG's 1M-trajectory published SOTA (0.87–0.91),
mechanistic sigma-lognormal and its real-residual / param-interpolation variants
(0.88), kNN blend (0.889), and now flow-latent interpolation (0.843). Everything
lands in [0.84, 0.91]. Genuine ~0.50 is reached ONLY by replaying real strokes
(0.506). No generalizing generator — learned, mechanistic, or interpolative —
crosses the finite-data fine-structure wall. The productive frontier is the
behavioral layers above the trajectory (reaction, evoked tracking,
challenge-response), not a better path generator.

---

# Capacity sweep of latent interpolation: a sweet spot at 0.815, then overfitting — wall confirmed

Latent interpolation was the one method that improved with capacity, so we swept
it to the maximum (trained the big flows on a dedicated GPU, scored on the
Balabit strong detector here):

| flow | params | converged train nll | interp shape_only |
|---|---|---|---|
| 16 layers / 256 hidden | 1.3M | -224 | 0.843 |
| 24 layers / 512 hidden | 9.8M | -338 | **0.815** (best) |
| 48 layers / 1024 hidden | 64M | -523 | 0.875 |

**The trend is non-monotonic — a sweet spot, not a path to 0.50.** 16→24 layers
improved interp (0.843→0.815), but scaling to 48 layers / 64M params REGRESSED it
(0.815→0.875). The mechanism is overfitting: 64M parameters on 14k training
strokes drives the *training* nll far deeper (-523) by memorizing the set, which
collapses the prior samples toward over-smoothness (jerk_rms 0.215 vs human's
0.332) and distorts the latent geometry so interpolation decodes to *less*
realistic strokes. Deeper training nll past the sweet spot is memorization, not
better generalization, so it does not lower — and in fact raises — the strong
detector's accuracy.

**This closes the last open question.** Capacity scaling does not push latent
interpolation toward 0.50; it bottoms out at ~0.815 and then overfits. Combined
with everything prior, EVERY generation approach — learned (GMM, flow-prior,
diffusion, DMTG-1M), mechanistic (sigma-lognormal + residual/interp), and
interpolative (latent, swept across capacity) — lands in [0.81, 0.91]. The
single lowest non-replay number in the entire project is latent interpolation's
0.815, still nowhere near the 0.506 that ONLY replay of real strokes reaches. No
generator crosses the finite-data fine-structure wall along any axis tried —
architecture, capacity, epochs, or data volume. The wall is confirmed as a
fundamental information-theoretic limit, not a modeling shortfall.

---

# Latent-anchored replay: denting the wall WITH diversity (0.68, not 0.85)

The reframe after every from-scratch generator capped at ~0.85: stop generating
from noise, and instead perturb a REAL stroke locally in a learned latent space —
"latent-anchored replay". Encode a real stroke z=f(x_real), add a small
per-dim Gaussian latent perturbation of scale sigma, decode. Unlike interpolation
(between two reals -> drifts off-manifold) or position-space blend (0.889), this
stays near ONE real anchor, so it should stay close to replay (0.5) while a small
sigma still breaks the finite-pool near-duplicates that sink pure replay.

Tested with a well-converged RealNVP (24 layers / 512 hidden, batch 1024, lr 2e-4,
8000 epochs -> train nll -395, trained in a PyTorch container on an A30), scored
against a FAIR baseline (human resampled through the same 48-point canonical
pipeline as the generated strokes, so the resampling loss cancels). Near-dup is
the min canonical-shape NN distance within the generated set (eps=0.0594):

| sigma | strong-ish detect acc | near-dup fraction |
|---|---|---|
| 0.0 (pure reconstruction) | 0.608 | 0.25 |
| 0.05 | 0.661 | 0.026 |
| **0.10** | **0.677** | **0.001** |
| 0.15 | 0.738 | 0.000 |
| 0.20 | 0.731 | 0.000 |

A PCA (linear-latent) anchor was far worse (0.834 at sigma=0.1) — the NONLINEAR
flow latent is what keeps the perturbation on-manifold, confirming the mechanism.

**Finding — a genuinely new point on the diversity/realism tradeoff.** At
sigma~0.1, latent-anchored replay scores **~0.68 detection WITH near-duplicates
fully broken (0.1%)**. That beats every from-scratch generator (GMM 0.855, flow
0.86, diffusion 0.86, latent interp 0.815) AND, unlike pure replay (0.506 but
finite), it manufactures unlimited diversity from a finite pool. It is the
strongest attacker construction found in the project: diverse AND well below the
~0.85 wall.

**It does not reach 0.50, and the reason is implementation, not information
theory.** The sigma=0 reconstruction floor is 0.608, not 0.5 — a normalizing flow
is exactly invertible in principle (decode(encode(x))=x -> 0.5), so the gap is the
ActNorm log-scale clamp [-6,6] and input clamp [-8,8] (added to stop an ActNorm
gradient blowup) slightly breaking exact invertibility on the deep -395 flow.
Restoring exact invertibility (unclamped + float64, or minimizing reconstruction
error directly) should drop the floor toward 0.5 and the sigma~0.1 point toward
~0.55-0.6-with-diversity — which would be a genuine "replay-grade AND diverse"
break of replay's only weakness (finiteness). That remains the open thread; the
wall for *from-scratch generation* stands, but latent-anchored replay is the first
method to dent it while keeping diversity.

---

# Closing the open thread: the sigma=0 floor is NOT numerical — the "dent" was a weak-detector artifact

The open thread above predicted that restoring **exact float64 invertibility**
(the ActNorm/input clamps were blamed for a 0.608 reconstruction floor instead of
0.5) would drop the sigma=0 floor toward 0.5 and the sigma~0.1 point toward
"replay-grade AND diverse". We retrained the flow and tested it directly. **The
prediction is false on both counts.**

**Setup.** Retrained the same RealNVP (24 layers / 512 hidden, 8000 epochs, batch
1024, lr 2e-4) on the A30 (`latent_anchor_invertible.py`), this time (a) saving
the checkpoint, (b) generating every anchor by encoding/decoding in **float64**,
and (c) printing the round-trip reconstruction MSE. Converged to **nll -399**
(even deeper than the prior -395). Scored with a fixed, honest instrument
(`score_anchor_fair.py`): the project's full strong detector (RandomizedSearchCV
over 4 families) against a fair baseline (human pushed through the identical
`to_canonical`→decode pipeline).

**Evidence 1 — reconstruction was already exact in float32.**

```
reconstruction MSE  float32 = 5.99e-12   float64 = 9.65e-30
```

The clamps do **not** break invertibility: forward and inverse apply the same
`log_s.clamp(-6,6)`, so it cancels algebraically, and float32 already round-trips
to ~1e-12. There was no numerical error for float64 to fix. The RESEARCH
hypothesis that the clamps "slightly break exact invertibility" was wrong.

**Evidence 2 — float64 does not lower the floor** (same fixed strong detector):

| sigma | float32 anchor (prior run) | float64 anchor (this run) | near-dup |
|---|---|---|---|
| 0.0 | 0.752 | **0.821** | 0.250 |
| 0.05 | — | 0.863 | 0.021 |
| 0.10 | 0.867 | 0.844 | 0.004 |
| 0.15 | — | 0.844 | 0.001 |
| 0.20 | — | 0.870 | 0.001 |

float64 sigma=0 is **0.821**, statistically the same band as float32's 0.752 (the
two are independent trainings). Nowhere near 0.5.

**Evidence 3 — the "0.677 dent below the wall" was a weak-detector artifact.**
The prior commit's 0.608/0.677 table used an uncommitted "strong-ish" detector.
Under the project's canonical **full** strong detector — the same one that scores
GMM 0.855 and the flow prior 0.86 — latent-anchored replay sits at **0.82–0.87
across every sigma**, i.e. squarely on the ~0.85 wall, not below it. It never
actually beat the wall.

**Where the floor comes from — a joint fingerprint of the flow round-trip, not
precision, not perturbation.** Honesty controls (`score_selfcontrol.py`) confirm
the instrument is sound: two disjoint halves of real human strokes through the
same pipeline score **0.501**, and reals sampled with-replacement + rotation
(the anchor sigma=0 construction, but decoded *without the flow*) score **0.47**.
Yet the *same* real strokes routed through the flow's encode→decode round-trip
(anchor sigma=0) separate from reals at **0.82–0.83** — even though their marginal
feature distributions are nearly identical (largest per-feature z-shift 0.043).
So the tell is not marginal and not a perturbation effect: the flow round-trip
imprints a tiny (~1e-6) but **structured, multi-feature** deviation that a tuned
tree ensemble aggregates into ~0.82 over 4000 samples. It is present at zero
perturbation and float64 does not touch it, because it is a property of the
learned map, not of arithmetic precision.

**Conclusion — the open thread is closed, negative.** Exact invertibility does
not exist as a lever here (the map already round-trips to 1e-12) and would not
help if it did: the sigma=0 floor is the flow's own reconstruction fingerprint,
which a strong detector reads at ~0.82. Combined with Evidence 3, latent-anchored
replay does **not** dent the wall under an honest strong detector — the earlier
dent was detector weakness. **Genuine ~0.50 remains reachable only by replaying
real strokes verbatim (0.506); every method that routes a real stroke through a
learned generative map, even at zero perturbation, is fingerprinted back up to
the ~0.85 band.** The wall stands, now including latent-anchored replay. The
productive frontier is unchanged: the behavioral layers above the trajectory
(reaction, evoked tracking, challenge-response), not a better path generator.

*(Reproduce: `latent_anchor_invertible.py` trains + generates float64 anchors and
saves `anchor_inv.ckpt`; `score_anchor_fair.py <anchor_file>...` scores against
the fair baseline; `score_anchor_clampfair.py` rules out the input clamp;
`score_selfcontrol.py` is the human-vs-human honesty control.)*

---

# ~0.57 WITH near-duplicates broken: the perturbation's FREQUENCY is the lever

The negative result above said the flow can't give on-manifold diversity. But the
target was never "generate from noise" — it is "break a finite pool's
near-duplicates as cheaply as possible." Chasing that directly found a real
sub-0.6 point, and the reason is a lever nobody had turned: the **frequency
content** of the per-flick perturbation.

**Step 1 — cancel the flow fingerprint (displacement replay).** Emitting
`decode(z)` costs 0.82 even at zero perturbation (the round-trip fingerprint).
Emit instead the CLEAN real plus only the flow's displacement,
`x + (decode(z+noise) - decode(z))` (`latent_displace_generator.py`, loads the
saved checkpoint, no retraining). At sigma=0 this is exactly `x` → **0.496**: the
fingerprint cancels perfectly. But the flow's displacement is off-manifold at ANY
scale — sigma=0.02 (near-dup 0.10) already jumps to **0.877** — because the flow
prior itself is only 0.86. A `x + local-PCA-tangent` perturbation from the human
data's own geometry (`local_tangent_generator.py`) is no better: 0.80–0.84 once
diversity is real. So neither a learned map nor the data's linear tangent supplies
cheap on-manifold diversity.

**Step 2 — the insight: the tell is high-FREQUENCY, not macro-shape novelty.**
Every failed perturbation above injects broadband/high-frequency content (jerk,
tremor) — the strong detector's most-relied-on features. What if the perturbation
that breaks near-duplicates is confined to LOW frequency? It changes the macro
shape (so a repeat no longer matches) while leaving the human jerk spectrum
intact. `minimal_replay_generator.py` compares, on the honest raw-vs-raw baseline
(`score_raw.py`, which reproduces `detect_replay_dilemma.py`'s 6px→0.64 and a
verbatim control of 0.53):

| perturbation (near-dup broken to ~0.01–0.02) | strong-detector acc |
|---|---|
| per-point WHITE jitter, 6 px | 0.632 |
| smooth warp, 3 half-sine modes, 6 px | 0.582 |
| smooth warp, **2 modes**, 8–10 px | **0.576 – 0.596** |
| smooth warp, **1 mode**, 18–22 px | **0.571 – 0.582** |
| verbatim replay (floor, near-dup 0.25) | 0.534 |

**Lower frequency wins monotonically** (modes 1 < 2 < 3 < white at equal
diversity), confirming the mechanism: a smooth low-frequency warp breaks the pool
via macro shape and pays almost nothing in the high-frequency features that carry
the bot tell. The best near-dup-broken point is **~0.57**, versus 0.64 for white
jitter and 0.85+ for everything off-manifold. The price of breaking a finite
pool's near-duplicates down to ~1% is only **+0.04 over verbatim** (0.53 → 0.57),
not the +0.11 (white jitter) or +0.32 (blends/flow) found before.

**What this refines.** The diversity↔on-manifold tradeoff is real and still holds
— you cannot break near-duplicates for free. But its **exchange rate is set by the
perturbation's frequency**, and at the low-frequency limit the rate is nearly
flat: ~0.57 detection buys a fully diversified finite pool. This is a genuine
sub-0.6 trajectory-layer attack, and the cheapest diversity found in the project.
Method for `needaimbot`: replay a distance-matched human stroke and add a 1–2-mode
smooth warp (~10–20 px, endpoints anchored) per flick — CPU-only, no model.

**Caveats, honestly.** (1) This is minimal-perturbation *replay*, not generation;
the ~0.85 *generation* wall is untouched. (2) It clears layer-1 (single-move
trajectory) + the near-duplicate check; the long-horizon / style-consistency /
cross-account detectors (layers above) still aggregate over a session, and a
pre-recorded stroke still cannot *react* to a moving target (challenge-response).
(3) 0.57 is above the 0.506 of pure verbatim replay — the finiteness escape still
costs something, just far less than previously measured.

*(Reproduce: `minimal_replay_generator.py --mode warp --modes 2 --scales ...`
generates; `score_raw.py <file>...` scores against the matched raw baseline;
`latent_displace_generator.py` + `score_anchor_fair.py` for the fingerprint-cancel
and flow-displacement cliff; `local_tangent_generator.py` for the data-tangent
comparison.)*

---

# CORRECTION to the latent-anchored-replay result: it was the 48-point bottleneck

The previous section reported latent-anchored replay at ~0.68-with-diversity and
called it a dent in the wall. **That number was an artifact of a biased fast
evaluation harness and is retracted.** The fast single-HistGBM detector used there
scores even human-vs-human at 0.611 (not 0.5), so every "detect" figure in that
sweep was inflated by ~0.11 and measured against a resampled-human baseline that
cancelled the wrong thing.

Re-scored with the real strong detector (`validate_flow_bot_strong_detector.py`,
the one that correctly gives human-vs-human = 0.500), the delta-corrected anchor
(output = real_stroke + [decode(z+noise) - decode(z)], so sigma=0 is provably the
real stroke) gives:

| sigma | strong-detector shape_only |
|---|---|
| 0.0 (= the real stroke, delta is exactly 0) | **0.741** |
| 0.05 | 0.852 |
| 0.10 | 0.848 |

**The decisive fact: sigma=0 is 0.741, not 0.5 — even though it is literally the
real stroke.** The delta correction worked; the gap is the **48-point canonical
representation**. To feed a fixed-dimension flow, every stroke is resampled to 48
points, which discards the high-frequency jerk/tremor. Against a full-resolution
human reference, that resampling ALONE is a 0.74 tell — before any perturbation.
Any sigma large enough to add diversity then climbs to the ~0.85 wall.

This is exactly why **warped replay reaches 0.506 and anchor cannot**: warped
replay keeps the stroke's ORIGINAL full-resolution points (rigid transform only),
while anchor must pass through the fixed-dim latent and lose resolution. The
tradeoff sharpens rather than breaks:

    diversity  -> needs a fixed-dim latent -> loses fine structure -> >=0.74
    full-res   -> only rigid replay        -> finite pool

The 48-point bottleneck is the finite-data manifold wall in another guise. Latent
anchoring was the right *idea* (a nonlinear-latent local perturbation genuinely
stays more on-manifold than PCA/blend), but the representation needed to make a
tractable latent is itself the ceiling. Breaking it would require a
variable-length / full-resolution generative model (autoregressive or
continuous-time flow), a much larger effort that most likely meets the same
finite-data generalization wall the whole project has documented. The from-scratch
generation wall (~0.85) and the replay-only-reaches-0.5 conclusion both stand.

---

# Is the 48-point bottleneck fixable by resolution? Partly — it plateaus at ~0.62

Diagnostic (no flow, no GPU): resample REAL strokes to N points (uniform-time
linear interp) and score against full-resolution human with the strong detector.
Real strokes are natively short — median 11 points (p10 8, p90 16) — so this is
upsampling.

| representation | strong-detector shape_only |
|---|---|
| resampled-real 24pt | 0.728 |
| resampled-real 48pt | 0.677 |
| resampled-real 192pt | 0.616 |
| resampled-real 384pt | 0.636 |
| (full-res replay, for reference) | 0.506 |

More points helps (0.73 -> 0.62) but **plateaus around ~0.62 and never approaches
0.50.** The residual floor is the re-representation itself: real strokes are
sampled at irregular native times, and any resample to a uniform-time grid with
linear interpolation alters the fine velocity/jerk kinematics no matter how many
points are used. Resolution removes the coarse-grid artifact but not the
re-parameterization tell.

**This nails down why warped replay reaches 0.506 and nothing else does.** Warped
replay keeps the stroke's EXACT native points and timing (rigid transform only);
every other method — generation, latent anchoring, even a straight
resample-of-a-real-stroke — re-represents the stroke and leaves a >=~0.62 tell. So
the practical ceiling for any *diverse* (re-represented) construction on this data
is ~0.62, while 0.50 is reachable only by exact replay of native samples, which is
finite. A higher-resolution (192pt) flow-anchor could push the diverse-attack
floor from ~0.74 toward ~0.62 — a real improvement over the ~0.85 generation wall,
and it keeps diversity — but it cannot reach replay's 0.50. The finite-data wall
holds; it just shows up as a re-representation floor once resolution is removed as
a confound.
