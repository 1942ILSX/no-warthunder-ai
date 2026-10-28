# Training guide

This guide documents the existing teacher, distillation, difficulty-recovery, and native evaluation workflows. Training outputs and frozen checkpoints are local files, not part of this source repository. Begin with [the setup instructions](../README.md); current trained deployment actors are shipped in the [backend models directory](https://github.com/1942ILSX/no-warthunder-backend/tree/main/models). The historical configs remain intact for reproducibility and are not promises that a single default run reproduces the qualified release.

## Choose a training device

```sh
python main.py doctor --device mps
python main.py benchmark-device
```

`--device auto` selects CUDA, then MPS, then CPU by availability. It does not benchmark which is fastest. MPS uses the Apple GPU; CUDA uses a compatible NVIDIA GPU. The Go simulation still runs on CPU. `--workers` controls Go worker concurrency independently of the learning device.

Small networks can favor CPU. One local teacher-MLP microbenchmark measured:

| Batch size | CPU forward + backward | MPS forward + backward |
| --- | ---: | ---: |
| 1 | 0.082 ms | 0.424 ms |
| 160 | 0.228 ms | 0.403 ms |
| 1024 | 0.834 ms | 0.641 ms |

These are model-only measurements, not end-to-end training throughput. Re-run the benchmark on your machine and compare actual iteration time. CPU is a sensible choice for small experiments; batched learning can benefit from MPS/CUDA. Both CPU and MPS passed short functional training checks in this project.

## Original teacher workflow (historical)

Each profile has its own seed, opponent mixture, output directory, and target difficulty band:

| Profile | Default iterations | Target defender win rate against reference bots | Game score multiplier |
| --- | ---: | ---: | ---: |
| trainee | 1200 | 70–90% | 1.0 |
| expert | 2000 | 40–60% | 1.5 |
| ace | 3000 | 10–30% | 2.0 |

The defender is the player side. A higher defender win rate therefore means an easier enemy policy. These bands are empirical targets against a fixed bot pool, not claims about human win rates. Score multipliers are metadata in the training pipeline and do not alter training rewards. The backend now applies the corresponding multiplier to live game scores; it retains one Top 10 per level.

Start all three independent teacher runs sequentially:

```sh
python main.py train-all --output runs/teachers --device mps
```

This creates `runs/teachers/trainee`, `runs/teachers/expert`, and `runs/teachers/ace`. It can take substantial time; no duration or convergence guarantee is implied. To train one profile:

```sh
python main.py train --profile expert --output runs/teachers/expert --device mps
```

Use `--device cpu` or `--device cuda` as appropriate. A new run refuses to overwrite a nonempty output directory. Configuration files live in `configs/`; `--config` selects a custom YAML file and `--iterations` changes the total iteration target. Keep the selected config for resume.

The default teacher has 32 parallel rooms, a 128-step rollout, and four PPO epochs per iteration. The curriculum begins with levels 1–2 and a short episode limit, expands to levels 1–6, then uses all ten levels with the full limit. Curriculum changes apply when a room starts its next episode.

For a deliberately short functional check only:

```sh
python main.py train --config configs/smoke.yaml --output work/my-smoke --device cpu
```

Smoke provenance is retained through resume, distillation, and fine-tuning. Smoke checkpoints are rejected by normal production export.

## Checkpoints, calibration, and resume

Each training run produces:

- `metrics.jsonl`: iteration, trained role, opponent, losses, sample counts, episode outcomes, and elapsed time.
- `checkpoint-NNNNNN.pt` and `latest.pt`: atomic snapshots for resume.
- `evaluation.jsonl`: periodic reference-bot evaluation results.
- `selection.json`: the latest assessment and the best candidate/eligible assessments.
- `best.pt`: the best evaluated candidate, which may still fail the difficulty or quality requirements.
- `best_eligible.pt`: created only after the target band and quality gates pass.

The original teacher configs evaluate **guard, hunter, and wander**, all ten levels, and fixed negative seeds from campaign `1000000`: 20 episodes per bot, or 60 matches total. Training episode seeds are positive. Current v3 releases instead use guard/hunter as the primary pool and the stricter coverage/quality gates described in the difficulty workflow. Candidates are ranked by quality violations first, then distance to the target win-rate band and its center. Stronger enemy play is not automatically better for the trainee profile.

Eligibility requires full level/bot coverage, enough matches, actual enemy optimizer updates, and the configured gates. Legacy teacher defaults allowed at most 10% combined timeouts/truncations and 25% stagnation; they are not the current v3 release criteria. Stagnation measures stuck-enemy observations relative to living-enemy observations across the evaluation, rather than inspecting only the last frame. A simulation timeout is never counted as an enemy victory.

If `best_eligible.pt` is absent, the run has not met its configured calibration requirements. Inspect `selection.json` and `evaluation.jsonl`; do not interpret `best.pt`, a profile label, or a completed iteration count as proof of success. Fixed-bot performance is a useful automatic selection signal, but it cannot establish human fun, fairness, or whether three policies feel sufficiently distinct.

Resume an interrupted expert run, optionally extending its total target:

```sh
python main.py train --profile expert --output runs/teachers/expert \
  --resume runs/teachers/expert/latest.pt --iterations 2500 --device mps
```

Checkpoints restore models, optimizers, RNG state, historical opponents, counters, and selection state. Live Go rooms restart as new episodes; resume does not reproduce an interrupted episode's exact trajectory. Only total iterations, progress/checkpoint cadence, and device may change freely. Optimizer, curriculum, opponent, profile, and calibration settings must match the saved config. Use a new fine-tuning run for intentional changes.

Ctrl+C leaves completed atomic checkpoints intact. A partly completed optimizer update is not presented as a valid new checkpoint. If `train-all` was interrupted, resume the affected profile with `train`, then launch any remaining profiles separately.

## Distill each teacher into a tiny policy

After selecting a suitable teacher, distill its enemy policy to `320 → 64 → 32 → 10`. The student controls the game and visits its own states; the frozen teacher labels those observations. A bounded CPU dataset aggregates recent observations, and masked KL loss trains the student in accelerator minibatches. The default distillation opponent is the teacher run's frozen learned defender.

The following commands assume all three teacher runs produced eligible checkpoints:

```sh
for profile in trainee expert ace; do
  python main.py distill --profile "$profile" \
    --teacher "runs/teachers/$profile/best_eligible.pt" \
    --output "runs/students/$profile" --device mps
done
```

Default distillation runs for 300 iterations and keeps up to 100,000 labeled states. Checkpoints include the dataset, student optimizer, and frozen teacher for resume; allow disk space for the periodic snapshots. A teacher must have recorded enemy-policy updates. Teacher and student profile/band metadata must agree.

To resume one student:

```sh
python main.py distill --profile expert \
  --teacher runs/teachers/expert/best_eligible.pt \
  --output runs/students/expert --resume runs/students/expert/latest.pt \
  --iterations 400 --device mps
```

Distillation does not transfer the teacher's calibration claim to the student. The student requires its own evaluation.

## Fine-tune the tiny policies with PPO

Fine-tuning is a new run, with a new output directory and optimizer state. It initializes the tiny enemy actor and the available defender from the distilled checkpoint. The supplied config trains the enemy policy on all ten levels while keeping opposing learned policies frozen during collection.

```sh
for profile in trainee expert ace; do
  python main.py train --profile "$profile" --config configs/finetune.yaml \
    --initial-enemy-checkpoint "runs/students/$profile/latest.pt" \
    --output "runs/tiny/$profile" --device mps
done
```

This stage also uses periodic difficulty-band selection. Prefer its `best_eligible.pt` when available. A later `latest.pt` can move outside the intended band. Fine-tuning is optional, but whichever student checkpoint you select must be evaluated independently before deployment.

## Independent final evaluation

The releases above have already completed independent final evaluation. Campaigns `1000000`, `2000000`, `3000000`, `4000000`, `6000000` and `8000000` have now been observed during this project's training, selection or review history. For future training/selection, retain earlier provenance, explicitly carry observed `6000000/200` and `8000000/200` into consumed campaigns, and reserve `10000000` or another untouched interval for the next frozen candidate. The CLI's v2/v3 default `4000000` is consumed here: **always specify `--seed`**.

Example for a future candidate, after its training and selection are finished:

```sh
python main.py evaluate --checkpoint runs/next-candidates/expert.pt \
  --episodes 200 --seed 10000000 --max-steps 9000 --device cpu \
  --output reports/expert-next-final.json
```

Reports contain per-opponent and per-level outcomes, timeout/blocked/stationary/moving-loop telemetry, individual records, sampler/model identity and the proxy calibration assessment. `evaluate` updates neither weights nor the checkpoint's selection files.

Legacy checkpoints without a sampler contract use deterministic masked argmax. Current categorical checkpoints automatically evaluate the exact native Go policy, including its per-enemy RNG and saved temperature. Torch categorical diagnostics cannot certify Go behavior. Changing sampler or temperature requires an explicit new candidate and corresponding native evaluation; numerical parity alone does not establish the same difficulty.

If final-test results guide further training or checkpoint selection, they are no longer an untouched final test. Reserve another fresh campaign seed for the next final assessment.

## Export, verify, and benchmark

The canonical release actors are already exported and are shipped in the backend repository. From the standard sibling-checkout layout, repeat their numerical and local inference checks:

```sh
for profile in trainee expert ace; do
  python main.py verify --model "../no-warthunder-backend/models/$profile"
  python main.py benchmark-policy --model "../no-warthunder-backend/models/$profile"
done
```

Export writes a versioned JSON description and contiguous little-endian FP32 weights with a checksum. Each release has 22,954 parameters and 91,816 weight bytes (275,448 bytes for all three). Only the supported tiny shapes are accepted; a `256,128` teacher must be distilled first. No Python interpreter, TorchRL package, centralized critic, or training-time RPC is needed for Go inference.

For a new v3 candidate, `export` requires `--evaluation` pointing to a passing independent report for that exact checkpoint SHA256. It inherits the saved sampler and temperature; categorical evidence must come from the exact native Go model. Use a new output directory and policy version. Earlier failed reports remain historical evidence and cannot authorize a release.

`verify` compares Python and Go logits within the declared tolerance and checks masked actions, including deterministic categorical-sampler parity. `benchmark-policy` measures the Go policy implementation. Export rejects smoke/untrained enemy policies by default; `--allow-untrained` is reserved for pipeline/parity tests and does not make an artifact trained.

These training/export commands do not install a new policy into the running game. The frontend and backend separately integrate the pinned native-v3 actors, a difficulty picker, and server-authoritative score multipliers. Passing fixed-bot release gates establishes the measured proxy behavior, not human playing experience or the whole-server resource budget.

The production target remains **25 independent rooms within 0.5 CPU core at approximately 3 GHz and 256 MB RAM**. An inference microbenchmark does not validate that whole-server limit. A later runtime load test must include physics, observations, AI decisions, WebSockets, leaderboard handling, and the actual CPU/memory quota before claiming the target is met.

## Learning architecture and reward

- **Actors:** a shared feed-forward enemy policy, normally `320 → 256 → 128 → 10` for the teacher; a separate defender policy with its own observation schema. ReLU hidden layers keep tiny deployment straightforward. Enemy role features remain part of the input. Legal-action masks apply in inference and loss computation.
- **Actions:** ten discrete choices, encoded as `move * 2 + fire`, with five movement choices and a fire bit. The Go bridge preserves the production 10 Hz decision point, two-decision observation delay, spawn warmup, and 30 Hz simulation physics.
- **Critics:** separate training-only centralized room critics. The current state has 2,688 features; the dimension comes from the bridge metadata. TorchRL GAE computes room returns, and TorchRL `ClipPPOLoss` supplies PPO clipping and entropy terms. Critic loss is counted once per room transition rather than once per living enemy.
- **Life boundaries:** inactive/dead/warming-up enemies and inactive defenders are excluded from actor loss. An individual enemy death does not end the team's value trajectory. Artificial episode cutoffs bootstrap the final critic state before reset; actual game termination does not.
- **Self-play:** the default schedule alternates four enemy updates with one defender update. Each rollout faces a frozen latest policy, a bounded historical checkpoint pool, or a scripted bot. Actor decisions are batched across rooms and slots; there is no device synchronization per tank.
- **Reward:** enemy objective victory gives `+1` and defender victory gives `-1`, with opposite rewards for the defender. Enemy shaping is `+0.03` per base HP lost, `+0.02` per player life lost, and `-0.005` per enemy HP lost. There is no direct reward for shooting, moving, survival time, or leaderboard score. The actual 900-second game timeout is terminal and gives both sides `-0.25`; an artificial training cutoff adds no terminal reward and allows value bootstrapping.

The Go bridge runs the existing game without rendering, HTTP, real-time sleeps, or leaderboard writes. Actor observations, action timing, and game rules are not reimplemented in Python. The backend's `docs/TRAINING_PROTOCOL.md` documents the binary array protocol and schemas.

## Optional Core ML / ANE experiment

MPS training uses the Apple GPU. **ANE is not a PyTorch autograd device and does not run PPO training here.** The optional Core ML path converts a frozen policy for an inference experiment:

```sh
python -m pip install -e '.[coreml]'
python main.py coreml --checkpoint runs/tiny/expert/best_eligible.pt \
  --output artifacts/expert-coreml.mlpackage --batch-size 160 --benchmark
```

Core ML receives `CPU_AND_NE`, which permits CPU/Neural Engine placement but does not prove that ANE executes this small model. The command reports numerical agreement and local inference timing when benchmarking. It is separate from the TorchRL training loop and the Go deployment artifact.

One local tiny-policy test at batch size 160 successfully converted and ran with coremltools 9.0: Core ML took 0.0546 ms versus 0.0285 ms for PyTorch CPU, with a maximum logit difference of `9.83e-5`. ANE placement was not confirmed and this test showed no speed benefit. coremltools also warned that PyTorch 2.14.1 is outside its tested range, whose latest tested version is 2.7. Conversion success for this model does not establish general compatibility. Optional Core ML conversion is not required for training or release.

## Development checks

```sh
python -m pytest -q
```

The current suite has **149 passing tests** covering the Go bridge, masked inference, room-level GAE, truncation bootstrapping, inactive actors, resume/config invariants, distillation, critic initialization, recovery, native sampler identity, provenance, calibration gates and Python/Go parity. Release training, fixed-bot final evaluation, and local frontend/backend integration are complete. Human playtesting remains separate from automatic qualification; no production deployment has been performed. Consult the sanitized [release manifest](release-native-v3.json) for the bounded resource probe scope.

See [the validation record](VALIDATION.md) for completed checks and their limits. `requirements-macos-arm64.lock.txt` records the verified Python 3.12 macOS environment, including optional Core ML dependencies; use the normal project install for other platforms. This project uses the [MIT License](../LICENSE).
