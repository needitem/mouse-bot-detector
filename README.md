# mouse-bot-detector

An adversarial study of **bot-vs-human mouse movement**: how well a classifier can
tell a generated/replayed mouse trajectory from a real one, and — by playing
attacker and defender against each other to convergence — where the real limits
of both sides lie.

It began as a way to evaluate `needaimbot`'s humanized-flick generator against
published mouse-dynamics research instead of eyeballing whether a trajectory
"looks human," and turned into a full map of the trajectory-realism problem.

> **Full research log:** [RESEARCH.md](RESEARCH.md) keeps the complete,
> chronological record — every version (v1–v11 of the parametric generator, the
> neural escalation, the arms race), every failed attempt, and every number,
> kept honest and unedited. This README is the summary.

---

## The core finding

No *generalizing* generator — hand-tuned formula, learned density, or neural —
beats **~0.85 detection accuracy** against a strong adaptive detector on this
feature/data setup. Genuine **~0.50 (indistinguishable) is reachable only by
replaying real human motion**, which is memorization, not generation.

| method | strong-detector accuracy (shape-only) | what it is |
|---|---|---|
| motor_synergy | 0.99 | hand-tuned parametric formula |
| kNN blend | 0.889 | averaged real strokes (smoothing kills the jerk) |
| high-k GMM | 0.855 | learned density |
| neural flow (RealNVP) | 0.81–0.86 | MLE + adversarial |
| DDPM diffusion | 0.85–0.86 | 8k–16k trajectories |
| DMTG (published SOTA) | 0.87–0.91 | 1,000,000 trajectories |
| **warped replay** | **0.506** | real stroke, rigid-transformed to target |
| human vs human | 0.500 | the floor (detector is honest) |

The **~0.85 wall is architecture-, capacity-, and data-independent** (a classical
density model, a flow, two diffusion models, and the million-trajectory published
SOTA all land in the same band). It is the **finite-data generalization gap**, not
a modeling weakness. This independently reproduces **SCRAP** (ACM AISec 2020):
domain-knowledge replay beats adversarial ML against mouse biometrics.

## The arms race (attacker vs. defender, played to convergence)

A single warped-replay flick is indistinguishable (0.506) — that is the *opening*
move, not the end. Iterating each side against the other's last move:

| side | move | outcome |
|---|---|---|
| defender | single-move detector | replay evades (0.506) |
| defender | + near-duplicate / session-distribution | finite pool repeats → caught |
| attacker | no-repeat pure replay, distinct sources | back to ~0.5 per session |
| defender | long-horizon (N>K), style-consistency, cross-account | forced reuse / shared pool → caught (→1.0) |

**Winner: the defender — but only under aggregate observation.** An aimbot can't
record in real time (the bot is aiming), so it is stuck with a **finite pool**,
and every way of hiding that finiteness leaves a structural trace across a
session, over time, or across accounts. **In a single short session the attacker
still wins** — a few no-repeat warped-replay flicks are genuinely human (0.5).
The defender's entire leverage is observing *enough*.

## The two real limits

- **Diversity ↔ on-manifold can't both hold with finite data.** Pure replay is
  on-manifold but repeats; anything that manufactures diversity (blend, perturb,
  stitch segments) steps off the human manifold and the single-move detector
  catches it. A perfect generative model would escape this — and it doesn't exist
  on finite data (that's the ~0.85 wall).
- **This is only layer 1: "is this *trajectory* human?"** That's necessary, not
  sufficient. Reaction time, aim correction, session variation, cross-modal
  consistency, and causal response to a moving/unpredictable target
  (challenge-response) are layers 2–6 — where trajectory statistics can't reach,
  and where a pre-recorded stroke fundamentally fails because it can't *react*.
  Collecting that data is what the companion
  [aim-rl-web-collector](https://github.com/needitem/aim-rl-web-collector) is for.

## Applied to needaimbot

Because warped replay is the only thing that defeats the trajectory detector, it
was ported into the companion aimbot
([needitem/needaimbot](https://github.com/needitem/needaimbot), `2pc` branch),
replacing `motor_synergy` outright: `warped_replay.hpp` picks a distance-matched
human stroke at runtime, rotates/scales it onto the aim vector, and replays it
with its real timestamps (CPU-only, ~1.6 µs/flick, no-repeat source selection).

## Pipeline

```
scripts/fetch_dataset.sh          # clone Balabit's public human dataset
scripts/parse_balabit.py          # segment sessions → movements, keep the fast/urgent tier
scripts/generate_synthetic.py     # naive_bot + a pure-Python port of motor_synergy
scripts/features.py               # shared feature extraction (identical on every class)
scripts/train_detector.py         # RandomizedSearchCV over 4 model families, held-out test
scripts/adversarial_loop.py       # evolves the generator config against the detector
```

Generators and controls that map the spectrum above:
`trajectory_gmm_ceiling.py`, `hybrid_noise_search.py` (high-k GMM),
`flow_generator.py` / `flow_adversarial.py` (RealNVP), `diffusion_generator.py`
(DDPM), `warped_replay_generator.py` / `replay_generator.py`,
`knn_blend_generator.py`, `validate_human_floor.py`, and the arms-race scripts
(`detect_replay_*.py`, `attack_*.py`, `detect_longhorizon_crossaccount.py`,
`detect_style_consistency.py`). Full details and results per script are in
[RESEARCH.md](RESEARCH.md).

## Data & method notes

- **Human ground truth:** Balabit Mouse Dynamics Challenge (real RDP mouse logs),
  filtered to the fastest quartile of movements as a stand-in for game-target
  urgency.
- **Confound control:** every movement is resampled to a common time grid before
  any derivative/FFT feature, because Balabit's coarse capture rate (~109 ms)
  vs. synthetic (~7–8 ms) otherwise inflates jerk/tremor by a `1/dt³` artifact.
- **Distance-matched** synthetic movements, so the detector can't shortcut on
  distance instead of shape.
- **shape_only** (canonical, rotation/scale-invariant) is the reported feature
  set everywhere; the `all` set trivially separates on absolute geometry.

## Papers referenced

- **SCRAP** — Synthetically Composed Replay Attacks vs. Adversarial ML Attacks
  against Mouse-based Biometric Authentication (ACM AISec 2020).
- **BeCAPTCHA-Mouse** — Acien et al., Pattern Recognition 2022 (arXiv:2005.00890):
  sigma-lognormal decomposition as the human-vs-bot feature space.
- **DMTG** — Entropy-Controlled Diffusion mouse-trajectory generation
  (arXiv:2410.18233): the million-trajectory published SOTA, still 87–91%.
- **Balabit Mouse Dynamics Challenge Data Set** (2016) — the human ground truth.
- **Harris & Wolpert (1998)** signal-dependent noise; **Fitts (1954)** law of
  movement time — the motor-control basis for the parametric generator and
  several detector features.
