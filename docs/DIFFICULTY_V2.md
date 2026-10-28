# Difficulty recovery and native evaluation

The v2 sections below document the historical repair workflow; the v3/native sections describe the current release contract. Paths under `runs/`, `reports/`, and `artifacts/` refer to retained local outputs, not files included in this repository. All previously observed final campaigns, including 4000000, 6000000 and 8000000, are consumed for future selection. Use an untouched campaign such as 10000000 for a new frozen candidate and retain campaign provenance. See [the current release record](VALIDATION.md).

The original teacher runs completed, but all three optimized the same winning reward. Different seeds, opponent mixtures, and checkpoint bands did not reliably produce three difficulty levels. V2 changes the actor's available information and its learning objective, then evaluates the actual standalone actor. It does not change frontend controls, runtime policy switching, scoring, or the Go game rules.

## What the ace evaluation established

`reports/ace-student-final.json` contains 300 matches against guard/hunter/wander on campaign 2000000: enemy win rate 80%, defender win rate 18%, six simulation timeouts (2%). The old aggregate calibration passed.

Against guard and hunter alone, defender win rate is **27%** and timeout rate is **3%**. That still falls inside the ace 10–30% band. However, primary stagnation is **25.25%**, with severe outliers on levels 8 and 9. Five of the six timeouts were on level 9; the six timeout games accounted for about 88.5% of all blocked-enemy steps. Most completed matches moved normally. These observations support keeping the trained policy as a starting point while repairing recovery behavior.

The original report is preserved. `reports/ace-student-v2-review.json` reaggregates its saved records under v2. Missing stationary telemetry is explicitly unknown, so a retrospective report cannot establish v2 eligibility. Bot win rates are proxies, not measured human win rates.

## Start the next training stage

From the project directory, using the existing environment:

```sh
.venv/bin/python main.py tune-difficulties \
  --checkpoint runs/students/ace/latest.pt \
  --output runs/difficulty-v2 \
  --device mps
```

This starts three **new independent** tiny-policy runs, sequentially, with seeds and objectives from `configs/difficulty/{trainee,expert,ace}.yaml`. It freezes the source checkpoint for the run set and leaves the original teachers/student untouched. Each target has fresh optimizers, critic learning, controller state, and checkpoint selection. The source actor stays tiny, so another distillation pass is not required for this workflow.

To inspect the plan without training, add `--plan-only`. To launch only one target, add `--profiles trainee` (or expert/ace). Default duration is bounded at 400 iterations per profile; there is no promise that every target will pass in that budget. Output reports explicitly identify missing calibration.

To resume a target, keep its exact v2 config:

```sh
.venv/bin/python main.py train \
  --config configs/difficulty/trainee.yaml \
  --output runs/difficulty-v2/trainee \
  --resume runs/difficulty-v2/trainee/latest.pt \
  --device mps
```

An interrupted `tune-difficulties` run is resumed per profile; launch remaining profiles with `--profiles`. A new run refuses to overwrite nonempty target directories.

## What changes in the objective

All profiles retain the game's positive victory reward and negative defeat reward. There is no reward for losing, committing suicide, or waiting out a match.

| Profile | Information available | Initial combat fire budget | Initial turn-and-fire budget |
| --- | --- | ---: | ---: |
| trainee | Local terrain/navigation/targets; no bullet telemetry or predicted danger | 0.20 | 0.03 |
| expert | Raw bullets; no precomputed closest-approach/directional danger or player velocity | 0.45 | 0.15 |
| ace | Full current observation contract | 1.00 | 1.00 |

These budgets are soft batch-average probability targets during PPO updates, not hard fire-rate guarantees in the game. The auxiliary loss penalizes excess attack pressure at real shot opportunities and instant turn-and-fire attacks. Brick-clearing shots and close-range self-defense are exempt. A separate navigation-recovery loss discourages repeatedly pushing the same blocked direction or idling without a useful firing line. The recovery loss treats move-with-fire and move-without-fire as the same movement choice.

After a quality-passing calibration, a bounded controller lowers the attack budgets if the policy is too hard, raises them if it is too easy, and holds them inside the desired band. It freezes difficulty adjustment while quality checks fail, so deadlock repair is not mistaken for a successful way to make a policy easier. Controller state is checkpointed and resumed. This is pragmatic penalty-based PPO, not a formal constrained-optimization guarantee; an unreachable band remains an explicit failure.

The information budget is enforced while learning and baked into zero columns of the first weight matrix. Export remains the same plain `320 → 64 → 32 → 10` MLP. There is no per-room neural wrapper, inference RPC, extra server model, or online difficulty controller. A player's live match is not used to adjust these profiles.

## Calibration v2

Guard and hunter are the primary reference bots. Wander remains diagnostic because a policy that consistently beats wander would otherwise face a two-thirds ceiling on the pooled defender win rate. That made the old trainee target misleading.

The defender win-rate bands stay trainee 70–90%, expert 40–60%, ace 10–30%, now measured on the primary pool. Every primary level/opponent pair needs at least five matches; default calibration runs 50 matches per bot (150 total, including wander).

Quality gates apply before difficulty selection:

| Scope | Max timeout rate | Max blocked stagnation | Max unproductive stationary rate |
| --- | ---: | ---: | ---: |
| Primary pool | 5% | 10% | 10% |
| Each primary opponent | 10% | 15% | 15% |
| Each level, primary pool | 10% | 20% | 20% |

Stationary telemetry tracks individual enemy IDs and delayed positions, resetting on respawn/movement. Prolonged stationary behavior without productive aiming/firing is counted separately from the game's blocked-movement timer. This catches additional idle policies; it is not a proof that every loop or camping strategy has been eliminated.

`selection.json` lists exact failed gates. `best.pt` can still be ineligible; only `best_eligible.pt` passed that run's calibration. The controller and checkpoint selector use the same v2 primary metrics. All three profiles must still be compared empirically after training; constraints alone do not prove their ordering or human enjoyment.

## Independent evaluation and export

Campaign 1000000 was used for old selection and 2000000 informed this repair. V2 calibration used **3000000** and the original final evaluation used **4000000**. Both have now informed selection; the CLI default 4000000 must not be reused as an untouched final test. Previously consumed campaigns and transfer/distillation ancestry are tracked to detect reuse.

When a profile has a `best_eligible.pt`, evaluate it, for example:

```sh
.venv/bin/python main.py evaluate \
  --checkpoint runs/difficulty-v2/ace/best_eligible.pt \
  --episodes 200 --seed 10000000 --device cpu \
  --output reports/ace-v2-final.json
```

Repeat for trainee and expert after their calibration passes. If final results influence another training change, reserve a fresh campaign for the subsequent final assessment. Do not replace a missing eligible checkpoint with `latest.pt` and assume it passed.

Trained v2 export requires a fresh independent evaluation tied to the **same checkpoint SHA256**, with passing difficulty and quality gates:

```sh
.venv/bin/python main.py export \
  --checkpoint runs/difficulty-v2/ace/best_eligible.pt \
  --evaluation reports/ace-v2-final.json \
  --output artifacts/ace-v2 --policy-version ace-v2
.venv/bin/python main.py verify --model artifacts/ace-v2
.venv/bin/python main.py benchmark-policy --model artifacts/ace-v2
```

The original v2 sampler was deterministic argmax. Current native-v3 releases use the categorical contract described below; categorical sampling changes behavior and cannot borrow argmax eligibility. `--allow-untrained` remains only for smoke/parity artifacts and cannot bypass evaluation for a trained v2 checkpoint.

The deployment weight size remains 91,816 bytes per default actor. The original 0.5 CPU / 256 MB / 25-room full-server quota still needs a later runtime load test; this training update does not assert that result.

## Calibration v3 and frozen-actor qualification

V3 retains all v2 difficulty and quality gates, then measures moving limit cycles using a bounded temporal observer. It counts repeated travel within a small region without navigation, health, terrain or meaningful combat progress. The default moving-no-progress limits are 5% for the primary pool, 10% per primary opponent, and 15% per level. The report includes detector settings and per-enemy telemetry; missing v2-era telemetry cannot be treated as zero. Training can enable `progress_recovery: {}` with `calibration_version: 3` to supply recovery labels from that observer. The observer is training/evaluation-only; the deployed actor and its 320 inputs are unchanged.

`evaluate --calibration-version 3` can measure an older frozen checkpoint under v3, but does not rewrite its saved contract. A measured actor can be explicitly qualified for the same or another profile without modifying its model tensors:

```sh
.venv/bin/python main.py qualify \
  --checkpoint runs/source/checkpoint.pt \
  --evaluation reports/source-v3-calibration.json \
  --profile expert --output runs/qualified/expert.pt
```

Qualification requires a fresh report tied to the exact source SHA256 and recomputes the target band's gates from complete, unique guard/hunter episode records. It writes a new checkpoint, preserves the original, records tensor/source/report fingerprints, and consumes that report's campaign as selection data. It is not release approval. Evaluate the new checkpoint on a separate campaign before export. For example, use seed 4000000 only if that campaign remains untouched; if it has informed selection or repairs, reserve a new seed interval.

V3 export requires a fresh v3 report for the exact exported checkpoint, with the saved detector settings and sampler, no consumed campaign overlap, complete seed coverage and passing recomputed gates. A v2 report, retrospective reassessment, changed sampler or `--allow-untrained` cannot bypass this requirement. Distillation inherits the teacher's v3 detector, thresholds, sampler and campaign ancestry, but the new student still requires its own evaluation.

## Native categorical evaluation

PPO learns a categorical policy. Replacing it with argmax can create deterministic movement cycles even when a small probability of another action would escape. The saved checkpoint can explicitly declare a deployment sampler:

```yaml
deployment_sampler:
  mode: categorical
  temperature: 1.0
  rng: xorshift64-v1
```

Missing sampler metadata retains the legacy argmax default. Sampler/temperature changes are behavior changes: create a new frozen candidate with explicit provenance, consume any campaigns used to choose it, and evaluate the new checkpoint. Never borrow an argmax report or a different temperature's eligibility.

For categorical checkpoints, the ordinary `evaluate` command automatically serializes the exact captured actor bytes into an evaluation-only temporary artifact and loads it into the actual Go `policymlp` implementation. Python sends `-1` for every enemy action, allowing the installed Go policy to act with the game's per-enemy `PolicyMemory` RNG. The bridge's full model metadata and weight checksum must match the requested artifact. Resets retain the same immutable model and reproduce seeded policy trajectories. `--native-go` also allows an argmax checkpoint to use this path.

The report records `native_policy`, the exact `policy_sampler`, and `reproducible_seeded_policy: true`. Categorical reports have `deterministic_policy: false`: actions are sampled, while complete seeded simulations remain reproducible. The temporary artifact is removed after evaluation; `trained: true` describes optimizer provenance and is not release certification. All existing v3 bands, quality gates and episode coverage requirements remain unchanged.

Export uses the checkpoint's saved sampler and temperature by default. Explicit `--sampler` or `--temperature` values must match that contract. Categorical release requires a fresh native Go report with exact checkpoint SHA256, recomputed actor weight SHA256, matching model schema, sampler/temperature/RNG, complete independent seeds and passing gates. A Torch categorical diagnostic is useful for investigation but cannot certify a Go release. `qualify` accepts native categorical evidence under the same identity checks, consumes its selection campaign, and still requires a fresh report for the newly qualified checkpoint.

Tiny-actor PPO calibration uses the same native helper when the saved contract is categorical. It writes a temporary snapshot of the paused current actor and opens isolated Go worlds using the training bridge's executable, worker count and timeout. These calibration results have no release checkpoint identity and cannot certify export. Larger teachers retain argmax calibration; categorical native calibration explicitly rejects unsupported actor shapes before optimization. Distillation preserves sampler metadata and requires the student's separate CLI evaluation. Core ML exports logits only, validate the saved sampler metadata, and never claim to apply the Go sampler or certify release.

For continuation under the same reward and critic schema, a new fine-tuning run can opt into `initialize_critics_from_source: true` alongside `--initial-enemy-checkpoint`. Both enemy and defender critic weights must exist, be finite, and exactly match the environment's critic input size and configured hidden sizes. Both are validated before either is loaded. This retains the learned value baseline while optimizers, update counters and league state start fresh. `initial_source.critic_initialization` records `source` or `fresh`; omitting the option retains the earlier fresh-critic behavior. A later resume restores that new run's own state normally.

## Method references

The implementation retains [TorchRL PPO/GAE](https://docs.pytorch.org/rl/stable/tutorials/multiagent_ppo.html). The distinction between a reward and a constraint is motivated by [Constrained Policy Optimization](https://proceedings.mlr.press/v70/achiam17a.html), but this code uses bounded auxiliary penalties and empirical gates rather than claiming that paper's guarantees. Optional student distillation continues to collect student-visited states; see [Distilling Policy Distillation](https://proceedings.mlr.press/v89/czarnecki19a.html) for relevant tradeoffs.
