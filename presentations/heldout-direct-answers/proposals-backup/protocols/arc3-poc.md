# ARC-AGI-3 proof of concept

Research checked 1 November 2026. Status: design/preflight specification only.
No ARC-AGI-3 environment or model run has been executed in this workspace.
The ARC-AGI SDK and arcengine are not installed in the current local environment.
No cluster package or GPU job has been submitted for this experiment.

## Scope

The existing 880/128/138 datasets are static ARC-1/2 tasks. The workspace name
arc-agi3-data-v1 does not denote interactive ARC-AGI-3 data. Current checkpoint-660
was fine-tuned on static demonstration-to-query JSON answers, not action trajectories.
The new comparison tests transfer to an interactive agent, not the same accuracy metric.

## Executable environment

Use the official arc_agi.Arcade and arcengine.GameAction interfaces, with
OperationMode.OFFLINE and one already downloaded public development environment.
Start with ls20, the official quickstart example. Resolve and pin its full versioned
ID and hash the environment assets before running. Inspect metadata for inventory,
not for a game-specific policy. Do not feed game source, metadata solutions,
human-baseline action counts or scorer information to the model.

Install the SDK in a separate CPU environment; do not alter .venv-arc or the
validated Axolotl/Transformers stack. The official toolkit documentation currently
lists arc-agi 0.9.9, Python >=3.12, and arcengine>=0.9.3. Pin the exact resolved
SDK/engine versions and dependencies after an import/offline probe. Do not claim
these versions have been installed or qualified here. Public game assets must be
staged while online before an offline cluster run; availability must be checked.

The environment worker runs on CPU and emits only allowlisted public observations.
The existing native GPU environment loads one quantized model across three H100s.
Base and fine-tuned use adapter off/on, respectively, with separate fresh game
instances at the same game seed and no shared policy history.

## Model interface

Loop: public frames/actions -> model -> strict action JSON -> environment -> new frames.
Frames can be up to 64x64 with values 0..15. The old 0..9 output-grid parser cannot
be reused. Preserve every returned animation frame in lossless row encoding;
record dimensions, color encoding and top-left (x,y) coordinates explicitly.
A candidate representation is one hexadecimal digit per pixel, A..F mapping to
10..15. Verify exact encode/decode round trips and actual tokenizer lengths.
Do not crop, resize, omit animation frames or silently truncate a prompt. If the
full observation/history exceeds the declared context, record a context-limit stop.

The policy receives current public frames, available actions, game state,
levels_completed and a bounded previous-transition history. The exact same
observation construction is used in both arms. Any bounded-history rule is fixed
before the experiment; each observation's complete frame is preserved. No goal
instructions, game-specific hints, demonstrations, scripts or known solutions.

A simple action response has schema {"action":"ACTION1","data":{}}.
ACTION6 instead requires integer x,y in 0..63. Validate membership in current
available actions and reject extra/missing fields, duplicate keys and invalid
coordinate types. No answer extraction/repair, random fallback or hidden retries.
For the first smoke run stop on invalid output, WIN or GAME_OVER. The agent is
not given game-specific meanings for numbered actions. No generated Python runs.

Proposed model settings: native quantization/FP32 adapters and checkpoint-660,
greedy, thinking disabled, seed 42, context 18432, one model call per action,
max128 generated tokens per call. These are a new interactive condition; they are
not the static benchmark's 8192-token answer budget. Freeze and record all limits.

## Staged checks and budgets

1. CPU: enumerate local SDK games; initialize the pinned public game; take five
   legal scripted actions; record raw public frames, available actions and state;
   verify deterministic initial observations for the same game/seed; check source
   isolation and frame encoding. This is an environment check, not model performance.
2. GPU smoke: at most five policy actions for each model on that same game/seed,
   one fresh model context/history per arm. Request three H100s in dev_gpu_h100,
   30-minute cap. Require actual observation-action-observation transitions for
   both arms, saved artifacts and valid scoring/logging. Winning is not required
   for infrastructure qualification. Incomplete arms are not a pass.
3. Only after measured smoke runtime, run a separately budgeted POC of up to
   50 actions per arm. Two hours is an initial allocation proposal, not a measured
   execution estimate or completion guarantee. Count all LLM calls/tokens, actions,
   invalid outputs, resets, deadlines and total allocated GPU-hours. No automatic
   full benchmark or leaderboard submission.

A 5-action smoke has at most10 model calls and1280 generated tokens total;
loading and repeated frame-prefill costs remain unmeasured. Do not extrapolate
runtime solely from static decode throughput. Pin stop conditions and how any
resets count before longer play. Preserve actions/frames so deterministic replay
can be verified before adding resume logic.

## Reporting

Save raw public observations, encoded prompts, response text/token IDs, parsed
actions, resulting states, seed/version identifiers, timings and stop reasons.
Report legal-action rate, levels completed, game WIN state, environment actions,
model calls/tokens and elapsed time separately for each arm. A one-game, limited-
action POC demonstrates interoperability and observed progress only, not general
ARC-AGI-3 performance or a benefit from fine-tuning.

Official ARC-AGI-3 uses Relative Human Action Efficiency (RHAE), which depends on
completed levels and human-relative action efficiency. Use the official pinned
scorer when available; do not invent a percentage from legal actions or pixel
changes. Keep scorer-only human baselines outside all model inputs. Offline public
play is a development experiment, not a competition-mode leaderboard result.

## Primary sources

- https://docs.arcprize.org/
- https://docs.arcprize.org/game-schema
- https://docs.arcprize.org/actions
- https://docs.arcprize.org/local-vs-online
- https://docs.arcprize.org/toolkit/environment_wrapper
- https://docs.arcprize.org/available-games
- https://docs.arcprize.org/methodology
- https://github.com/arcprize/ARC-AGI
- https://github.com/arcprize/ARC-AGI/blob/main/pyproject.toml
- https://github.com/arcprize/ARC-AGI/blob/main/arc_agi/wrapper.py
