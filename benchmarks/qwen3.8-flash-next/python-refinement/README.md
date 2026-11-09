# Python refinement on internal ARC validation

The base model and checkpoint-660 generated a Python `solve(grid)` function. A sandbox ran each candidate against the task demonstrations. The model could use the demonstration results to revise its program, up to four candidates per task and model. The scorer compared the selected program’s query outputs with the reference grids. These runs used the 128-task internal split and all stopped with partial coverage.

| Run | Tasks started per arm | Base solved / scored | Checkpoint-660 solved / scored |
|---|---:|---:|---:|
| [Thinking enabled](thinking-on/report.json) | Not tabulated here | 1 / 2 | 3 / 3 |
| [Thinking disabled, serial](serial-no-thinking/report.json) | 7 | 2 / 6 | 4 / 7 |
| [Thinking disabled, length-grouped batches](length-batched-no-thinking/report.json) | 33 | 10 / 19 | 10 / 26 |

The numerators count tasks where every query grid matched. The denominators count tasks with a scored selected candidate for that model. In the length-grouped run, for example, the base arm had 10 exact, 9 scored incorrect, 14 started without a selected score, and 95 unattempted. Checkpoint-660 had 10 exact, 16 scored incorrect, 7 started without a selected score, and 95 unattempted. These runs cover overlapping tasks.

The batched worker grouped rows by input length and padding limits. Its cap was 32. Actual generation calls used sizes 1 through 13 and 15. The serial worker used one row per call.

## Evidence and code

Each run directory contains the first-candidate and selected-candidate scores in `report.json` and archived program text with SHA-256 checksums in `candidates.jsonl`. The two thinking-disabled directories also contain the attempted IDs from the batch journals. The candidate files record available Python programs; a generation that never produced extractable Python has no program to archive.

The [historical producer packages](producer/) include the sandbox, generation loop, selection logic and scorer. The private workspace path is replaced with a path relative to the deployed package. Running these historical packages requires the original model, adapter, staged inputs, environment and package manifests. The published evidence can be checked offline with `python3 benchmarks/verify.py` from the repository root.
