# Validation record

## Current native v3 release — 2026-10-28

Training, selection and independent native Go final evaluation are complete for all three tiny release policies. The authoritative [sanitized release manifest](release-native-v3.json) records checkpoint/weight/evaluation hashes and the separate target-server verification status. Local frontend/backend integration now includes difficulty selection, pinned policy switching, and server-authoritative score multipliers. No production deployment has occurred. The checkpoint and raw-report paths below describe retained local evidence; they are excluded from Git. Public deployment actors are in [backend/models](https://github.com/1942ILSX/no-warthunder-backend/tree/main/models).

| Profile | Native categorical temperature | Final campaign | Primary defender win rate | Primary timeout | Moving-no-progress rate |
| --- | ---: | --- | ---: | ---: | ---: |
| trainee | 1.0 | 6000000 × 200 per bot | 74.5% | 0% | 0.281% |
| expert | 0.3 | 6000000 × 200 per bot | 48.75% | 0% | 0.431% |
| ace | 0.5 | 8000000 × 200 per bot | 26.75% | 0% | 0% |

Each final contains 600 matches: 400 guard/hunter primary matches and 200 diagnostic wander matches. Every v3 difficulty, coverage, per-level and per-opponent quality gate passed. These fixed-bot results do not establish human win rates, fun or fairness.

- Canonical frozen checkpoints: `runs/release-native-v3/{trainee,expert,ace}.pt`.
- Canonical artifacts: `artifacts/release-native-v3/{trainee,expert,ace}/`, using the saved `categorical` / `xorshift64-v1` sampler contracts.
- Final reports: `reports/native-v3/final/trainee-6m-200.json`, `reports/native-v3/final/expert-6m-200.json`, and `reports/native-v3/ace-repair-t05/ace-t05-final-8m-200.json`.
- Independent read-only audit recomputed canonical actor bytes, checkpoint/report/weight hashes, saved sampler contracts, campaign non-overlap and all release gates. All three passed. Each actor has 22,954 parameters and 91,816 weight bytes; the three weight sets total 275,448 bytes.
- Python/Go parity passed for all three releases: 128 observations each, matching masked argmax/categorical actions and RNG transitions. Current Python suite: **149 passed**, including native policy identity/reset replay, categorical export evidence, recovery, qualification and opt-in critic initialization.

The older `runs/native-v3/final-candidates/ace.pt` and `reports/native-v3/final/ace-6m-200.json` are preserved failed evidence, **not the current ace release**. Its subsequent sampler repair consumed that observed campaign before using fresh 8000000. Original failed reports, checkpoints and configs remain available; earlier argmax and smoke findings below must not be substituted for current native categorical evidence.

Future training/selection must retain all earlier consumed campaigns and add the now-observed `6000000/200` and `8000000/200` to provenance. Reserve `10000000` or another untouched campaign for a subsequent frozen candidate. The CLI's default v2/v3 seed `4000000` is already consumed in this project; always supply `--seed` explicitly.

Separate target-server verification passed on the resource-constrained target host: a 30-second probe enforced 0.5 CPU quota, 192 MiB hard memory limit and no swap, running 25 real rooms at 29.9666 Hz with 15 Hz JSON snapshots. Measured mean CPU was 0.06857 core and peak RSS 9.703 MiB, with zero dropped ticks and zero throttled CPU periods. Actual live inference averaged 859.07/s, supplemented by 139.87/s calls on real observations with independent policy memory, for 998.93/s total. Mean occupancy was 3.49 enemies/room and peak live occupancy was 100; supplemental calls are not additional live enemies. Tick-work p99 was 6.587 ms and wake-lateness p99 was 1.121 ms. Existing service MainPIDs were unchanged; all owned temporary units and files were removed. The manifest binds the measured model hashes and records the limitations: this headless workload includes simulation, observations, inference and JSON, but excludes HTTP/WebSocket/TLS, persistence and other application overhead, so it is not a sustained complete-server load test. Local runtime integration is complete, including a 25-room HTTP/WebSocket smoke test through the frontend proxy. That local smoke test is not a measurement under the target server quota. Human playtesting remains distinct from automatic evaluation. The backend now applies score multipliers `1 / 1.5 / 2`.

## Historical difficulty v2 framework follow-up

The first formal teacher runs and ace student have since completed. Their evaluation findings and the new training procedure are documented in [DIFFICULTY_V2.md](DIFFICULTY_V2.md). The original validation below records the earlier framework checks; it does not certify the later trained actor.

At the earlier v2 framework milestone, validation completed:

- **80 Python tests passed**, including pressure gradients, brick/self-defense exemptions, input-gate persistence, controller bounds/quality freeze/resume, explicit profile transfer, inherited distillation contracts, per-level calibration and stationary telemetry.
- Three short real-Go **MPS** runs initialized from the actual ace student: two PPO iterations plus one resumed iteration per profile. Original source SHA256 stayed unchanged. All outputs retain smoke provenance.
- Each profile's masked first-layer columns remained exactly zero and its smoke artifact passed Python/Go logits plus argmax/categorical action/RNG parity (48 observations per profile). Weight shape and size stayed unchanged.
- Actual CLI v2 evaluation on a separate smoke campaign produced stationary telemetry and a checkpoint-bound report. Actual `tune-difficulties --plan-only` validated all three profiles without creating the formal output directory.
- Snapshot regression tests cover checkpoint replacement during evaluation/export, immutable source freezing, altered-source rejection, strict preflight, and calibration lineage across repeated transfer/distillation.
- V2 export rejects missing, mismatched, reused-campaign, unqualified, or sampler-incompatible evaluation evidence for trained v2 policies. Smoke testing cannot bypass those checks for a trained policy.

No formal v2 training had been launched at that milestone. Its smoke campaign was deliberately too short for eligibility and correctly failed coverage/quality gates. Formal training and later v3 release evaluation subsequently completed as recorded above; the older smoke checks themselves establish no playing strength. No frontend or Go production wiring changes were made in that follow-up.

## Historical original framework validation — 2026-10-27

This section records the original bounded implementation checks on the local Apple Silicon machine. All weights generated by those checks were smoke artifacts, marked `trained: false` on export. At that time there had been no long training run or release policy, and no frontend difficulty control, backend policy switch or score multiplier was activated.

## Historical checks completed

- Python: `python -m pytest -q` — **25 passed**. Includes real Go protocol integration, atomic reset rejection, room-level returns, terminal/truncation handling, actor masks, checkpoint/resume invariants, profile guards, student distillation, evaluation seed overlap detection, and both deployment samplers.
- Go: `go test ./...` and `go vet ./...` passed. Go package race checks also passed during implementation. Scripted headless-versus-production parity is covered for 900 physics ticks, including decision delay and spawn/slot reuse.
- MPS: actual Go rooms plus enemy and defender PPO updates; training resume; student distillation and resume with the original teacher path unavailable. A separate bounded check used the real teacher shape `320 → 256 → 128 → 10` and compressed it into `320 → 64 → 32 → 10` against a frozen learned defender.
- Evaluation: bounded actual Go episodes produced per-opponent reports, correct truncation outcomes, and explicit failed calibration for the short smoke campaign.
- Export: each of the three difficulty labels passed the strict Go loader and Python/Go parity checks. Smoke export required the explicit test-only override and retained `trained: false`.
- Parity: 128 normalized random observations and 256 actual observations sampled from all ten levels. Both argmax and categorical actions, RNG transitions, and decision counters matched. Categorical cases exercised all ten actions. Maximum logit error on the actual-observation corpus was approximately `3.73e-9` (tolerance: `rtol=1e-4`, `atol=1e-5`).
- Existing local game processes remained healthy; the frontend working tree and backend runtime entry point/server wiring were unchanged.

## Historical local measurements

The tiny actor has **22,954 FP32 parameters**, **91,816 weight bytes**, and **22,848 multiply-accumulates per decision**. Three independently loaded weight sets require **275,448 bytes total**, excluding metadata and scratch buffers. Models can be shared across rooms when runtime integration is added.

The standalone Go `Decide` benchmark ran 100,000 iterations:

| Measurement | Result |
| --- | ---: |
| Mean decision time | 13.40 µs |
| Heap allocations per decision | 0 |
| Heap bytes allocated in the measured loop | 0 |
| Arithmetic time at 1,000 decisions/second | 13.40 ms/second |

The current ten levels allow at most four simultaneous enemies per room. At 25 rooms and 10 Hz this gives up to 1,000 decisions/second. This arithmetic projection is **not a measured whole-server CPU percentage or quota result**. It excludes observation construction, physics, networking, leaderboard work, and garbage collection elsewhere. It was measured on an M5 Max, not a target 3 GHz server.

Teacher-network forward/backward microbenchmark:

| Batch | CPU | MPS |
| --- | ---: | ---: |
| 1 | 0.0819 ms | 0.4243 ms |
| 160 | 0.2276 ms | 0.4033 ms |
| 1024 | 0.8344 ms | 0.6413 ms |

Optional Core ML conversion and frozen inference succeeded. At batch 160, `CPU_AND_NE` measured **0.0546 ms** versus **0.0285 ms** for Torch CPU, with maximum logit error **9.83e-5**. This did not establish ANE placement or a speed benefit. Core ML Tools 9.0 warns that Torch 2.14.1 is newer than its tested Torch version; optional conversion is isolated from training and Go export.

## Limits of the historical measurements

Sustained training and qualified release models were still pending when the original smoke measurements were recorded; those tasks are now completed above. The historical inference microbenchmark still does not prove a whole-server quota result. A future sustained quota test must cover the complete integrated server with **25 active rooms**, an actual **0.5 CPU quota** and **256 MB memory limit** on representative deployment hardware.

Raw historical smoke checkpoints and reports under ignored `work/` are not release models. Use the canonical `runs/release-native-v3/` and `artifacts/release-native-v3/` paths and authoritative manifest for the current release, rather than selecting an arbitrary `latest.pt` or old failed candidate.
