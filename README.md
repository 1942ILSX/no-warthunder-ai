# NO WarThunder AI

Offline self-play training and tiny Go policy export for **NO WarThunder**.
PyTorch and TorchRL train the enemy policies against learned defenders and
reference bots in the real Go simulation. No human demonstrations or manual
training matches are required.

The three qualified native-v3 actors are integrated into the local game:

| Difficulty | Score multiplier | Categorical temperature | Reference defender win rate |
| --- | ---: | ---: | ---: |
| Trainee | ×1.0 | 1.0 | 74.5% |
| Expert | ×1.5 | 0.3 | 48.75% |
| Ace | ×2.0 | 0.5 | 26.75% |

Each actor is `320 → 64 → 32 → 10` ReLU: **22,954 parameters and 91,816 bytes of
FP32 weights**. Go inference needs no Python, GPU, ANE, ONNX runtime, or inference
service. Weights are shared across rooms; each tank has its own RNG and policy
memory. Difficulty is fixed when a room is created. The backend applies the
score multiplier and keeps one Top 10 per level across all difficulties.

These win rates come from fixed reference bots, not human players. Each final
campaign had 600 matches across ten levels, including 400 primary guard/hunter
matches; all difficulty and quality gates passed, with zero primary timeouts.
The exact weights, sampler and temperature form one deployment contract.
Changing a sampler or substituting a checkpoint requires new evaluation.

## Repositories and release files

- [Backend](https://github.com/1942ILSX/no-warthunder-backend): Go game, training
  bridge, native inference and the [qualified deployment actors](https://github.com/1942ILSX/no-warthunder-backend/tree/main/models).
- [Frontend](https://github.com/1942ILSX/no-warthunder-frontend): browser game,
  difficulty selection and English / Simplified Chinese / Japanese UI.
- [AI](https://github.com/1942ILSX/no-warthunder-ai): this training framework,
  evaluation, export, configs and tests.

[Release metadata](docs/release-native-v3.json) records model/checkpoint/report
hashes, sampler contracts and the scope of completed validation.
[Validation details](docs/VALIDATION.md) distinguish current results from earlier
failed candidates and framework checks.

This repository includes source and reviewed summaries. Virtual environments,
training checkpoints, optimizer state, raw reports, local runs and generated
artifacts are intentionally excluded from Git. Paths under `runs/`, `reports/`
and `artifacts/` in the detailed guides refer to outputs you generate or retain
locally. The deployment weights are committed in the backend repository; they
are sufficient for gameplay and parity checks, but are not training-resume
checkpoints. The older Ace T0.3 candidate is not the qualified Ace release.

## Setup

Use Python **3.12 or 3.13** and the Go version required by the backend's `go.mod`.
Keep both checkouts next to each other for automatic bridge discovery:

```text
workspace/
  no-warthunder-ai/
  no-warthunder-backend/
```

From this repository:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python main.py doctor
```

The learning stack is pinned to PyTorch 2.14.1, TorchRL 0.14.1 and TensorDict
0.14.3. `requirements-macos-arm64.lock.txt` captures the verified Python 3.12
macOS environment, including optional Core ML dependencies; use the regular
project install on other platforms.

For another backend location, set `NO_WARTHUNDER_BACKEND` or pass `--backend`:

```sh
export NO_WARTHUNDER_BACKEND=/path/to/no-warthunder-backend
python main.py doctor --device cpu
```

`doctor` checks the environment, a small forward/backward pass, and the Go
bridge. It does not start training. Running `python main.py` without arguments
also runs `doctor`. `python -m nowarthunder_ai` and the installed
`nowarthunder-ai` command expose the same CLI.

## Training workflow

Start with a short functional check:

```sh
python main.py train --config configs/smoke.yaml \
  --output work/my-smoke --device cpu
```

Smoke checkpoints retain smoke provenance and cannot pass normal release
export. For a new experiment, the original teacher/student workflow is:

```sh
python main.py train --profile expert --output runs/teachers/expert --device mps
python main.py distill --profile expert \
  --teacher runs/teachers/expert/best_eligible.pt \
  --output runs/students/expert --device mps
python main.py train --profile expert --config configs/finetune.yaml \
  --initial-enemy-checkpoint runs/students/expert/latest.pt \
  --output runs/tiny/expert --device mps
```

These stages are run separately: inspect calibration before continuing, and do
not assume `best_eligible.pt` exists. `best.pt` is only a candidate and may fail
quality gates. The provided configs preserve the project's historical training
stages; the qualified release required additional difficulty tuning, recovery,
sampler selection and independent native evaluation. One default command is
not a guarantee of reproducing the release or reaching the desired win-rate
band.

For the full procedure, resume rules, checkpoint selection and learning design,
see the [training guide](docs/TRAINING_GUIDE.md). The
[difficulty recovery guide](docs/DIFFICULTY_V2.md) explains information budgets,
attack-pressure penalties, recovery telemetry, v3 qualification and native
categorical evaluation.

## Evaluation and export

Freeze a candidate before final evaluation. Campaigns 1000000, 2000000, 3000000,
4000000, 6000000 and 8000000 have already informed this project's training,
selection or review. Retain that consumed-campaign provenance. The CLI default
4000000 is therefore not an untouched final test for continuing this project;
choose an unused campaign explicitly, for example:

```sh
python main.py evaluate --checkpoint runs/next-candidates/expert.pt \
  --episodes 200 --seed 10000000 --max-steps 9000 --device cpu \
  --output reports/expert-next-final.json
python main.py export --checkpoint runs/next-candidates/expert.pt \
  --evaluation reports/expert-next-final.json \
  --output artifacts/expert-next --policy-version nwt-expert-next
python main.py verify --model artifacts/expert-next
python main.py benchmark-policy --model artifacts/expert-next
```

Export requires passing evidence tied to the exact checkpoint and saved
sampler contract. Categorical evaluation automatically runs the actual Go
model and per-tank RNG. A report from another temperature, argmax policy, or
checkpoint cannot certify that export. If final results influence more
training or selection, that campaign becomes consumed and the next candidate
needs another fresh campaign.

To verify the already-qualified actors from the backend checkout:

```sh
for profile in trainee expert ace; do
  python main.py verify --model "../no-warthunder-backend/models/$profile"
done
```

## GPU, ANE and deployment budget

`--device auto` selects CUDA, then MPS, then CPU by availability. It does not
benchmark which is fastest. MPS uses the Apple GPU; the Go simulation remains
on CPU. `--workers` controls simulation concurrency separately. Tiny networks
may be faster on CPU, so compare `python main.py benchmark-device` and actual
iteration times before committing to a long run.

ANE is not a PyTorch autograd device and does not train PPO here. The optional
`coreml` extra exports a frozen policy for an inference experiment. Core ML
`CPU_AND_NE` allows Neural Engine placement but does not establish that ANE
actually executed the model or made it faster. It is not part of Go gameplay.

The deployment target is **25 rooms within 0.5 CPU core and 256 MB RAM**. A
completed 30-second target-host probe enforced 0.5 CPU and 192 MiB with no swap;
its simulation/inference/JSON workload used about 0.069 CPU core and 9.70 MiB
peak RSS, with no dropped ticks. This excludes HTTP/WebSocket/TLS, persistence
and other full-application overhead. A separate local 25-room HTTP/WebSocket
smoke test passed. Neither result is a sustained full-server quota test, and no
production deployment has been performed.

## Development

```sh
python -m pytest -q
```

The suite covers the real Go bridge, observation/action parity, PPO/GAE,
truncation bootstrapping, resume invariants, distillation, recovery, sampler
identity, provenance and release gates. The last complete run passed 149 tests.

Licensed under the [MIT License](LICENSE).
