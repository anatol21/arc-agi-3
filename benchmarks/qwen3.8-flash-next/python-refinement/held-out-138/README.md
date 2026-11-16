# Held-out Python refinement on 138 ARC tasks

**First pass at 15:59 CET on 16 November 2026.** Both arms used the same frozen NF4 base with the adapter disabled on four H100s. Arm A generated Python solvers and revised them using demonstration feedback. Arm B could also call Python up to twice before its first solver. A task was exact only when its selected program reproduced every query grid.

| First-pass outcome on the same 138 tasks | A · refinement | B · optional exploration + refinement |
|---|---:|---:|
| Exact | 0 | 1 |
| Returned at least one grid, wrong task answer | 8 | 10 |
| Closed without a grid | 116 | 110 |
| Started and still open | 6 | 7 |
| Not attempted | 8 | 10 |

Arm B solved [`e95e3d8e`](python-case-audit/e95e3d8e-B-candidate0.py) with its first candidate. The program recovered a repeating tile and filled the missing cells. Arm A reached the task deadline before finishing a candidate on that puzzle. Across the frozen first-pass records, B made zero accepted exploratory calls. The solve therefore provides evidence for the generated solver, and provides no comparison of probe effectiveness. [Paired task trace](python-case-audit/e95e3d8e-trace.json)

## Data and execution

- [All 276 arm–task records](python-case-audit/per-task-audit.csv) give outcome, stop reason, generated tokens, charged time, selected demonstration matches and comparisons with earlier direct-answer runs. [Case records](python-case-audit/case-records.json) include saved programs, observations, query comparisons and journal hashes. [Snapshot](python-first-pass/snapshot.json) identifies source allocations and time. This release covers first-pass results only.
- [Protocol](python-first-pass/PROTOCOL.md) and [batch controller](producer/arc-python-heldout-length-v1/panel.py) specify generation budgets, admission by current token length, OOM handling, and execution reserve. [Decoder](producer/arc-python-heldout-length-v1/decoding.py), [worker](producer/arc-python-heldout-length-v1/worker.py), [probe and feedback rules](producer/arc-python-heldout-length-v1/protocol_pilot.py), [launcher](producer/arc-python-heldout-length-v1/launch.py), [retry](producer/arc-python-heldout-length-v1/retry30.py) and [CPU scorer](producer/arc-python-heldout-length-v1/score.py) are included.
- The held-out worker directory is the output of the [historical package generator](../thinking-on/internal-exploration-pilot/producer/heldout/package.py) applied to the archived local source. [Checksums](source-provenance.json) identify the released files. The [public input manifest](producer/arc-python-heldout-length-v1/visible.jsonl) contains demonstrations and unlabeled query inputs. The scorer reads a separate reference file. Running on GPUs also requires the native loader, the [earlier Python controller packages](../producer/), prepared control prompts and model assets.
- Run `python3 benchmarks/verify.py` from the repository root to check the snapshot offline. The presentation includes the failure categories, example programs and maps for both arms.
