# OpenRouter prompt study on 138 ARC validation tasks

On 15–16 October 2026, ARC Practice evaluated five prompt strategies with `qwen/qwen3.8-flash` via OpenRouter. Each strategy ran twice on the same 138 public tasks, which contain 185 query grids. All 1,380 planned task–strategy–trial combinations have a terminal record. The API model answered at temperature 0.2 and top-p 0.95, with reasoning requested and JSON output. A trial was correct if every query grid matched exactly.

| Prompt strategy | Correct trials / 276 | Tasks solved at least once / 138 | Malformed responses | Provider errors |
| --- | ---: | ---: | ---: | ---: |
| ZS Python Synthesis | 19 | 13 | 64 | 6 |
| 1S Python Synthesis | 19 | 11 | 64 | 7 |
| Python Self-Refinement | 29 | 18 | 66 | 7 |
| NL Rule Induction CoT | 13 | 9 | 60 | 5 |
| ZS ARC DSL | 24 | 14 | 59 | 5 |

Across all strategies, 104 of 1,380 trials were correct and **21 distinct tasks** were solved at least once in ten trials per task. The methods named “Python” used prompt instructions. The application scored grids returned by the API and recorded no external Python execution in this experiment. This study used different prompts, sampling and model serving from the cluster’s greedy base/checkpoint-660 comparison. The same held-out tasks had therefore already been used for these prompt experiments.

## Generated evidence

- [Trial records](trials.jsonl.gz) contain task IDs, strategy and trial indices, recorded validity and per-query counts, attempt outcomes, and the exact saved answer text. `raw_response` is the last answer text retained by the original trial. For 21 of the 30 provider errors it is null. Nine error records retain text from the first attempt after the second attempt failed at the provider. `raw_response_attempt` identifies the 1-based attempt that produced the saved text. The archive stores an ordered status for each request, but the worker overwrote the answer buffer when a repair returned new text. For 198 repaired trials the preceding malformed text was unavailable. All provider errors count as failures.
- [Summary and solved IDs](summary.json) contains per-strategy status counts and the union of solved tasks. “Solved at least once” counts distinct task IDs. “Correct trials” counts individual responses.
- [Settings](settings.json) contains the exact prompt instructions and modifiers, model, decoding parameters and output-token allowances. One format repair was permitted. The repair could increase the output allowance after truncation. The seeds shown for the source runs controlled task selection.
- [Source hashes](sources.json) identify the three private exports used by the extraction script. The source export had 20 completed trial records for two tasks from an interrupted API batch, plus 1,360 records for the other 136 tasks. A separate 23-record aborted launch was excluded. Of the 1,360 records, 804 had been compacted on resume. The extractor retrieves their response text from the preceding checkpoint after matching task, method, trial, outcomes, parsed prediction, attempt history and usage.

No provider reasoning traces, request identifiers, credentials or reference answers are distributed here.

## Code and verification

[ARC Practice](../../apps/arc_practice/server.py) contains the archived local producer. `task_prompt` constructs the request, `OpenRouterClient.request_payload` adds generation settings and `ExperimentBatch._run_one` sends the request, handles retries, parses outputs and checks exact grid equality. [The application guide](../../apps/arc_practice/README.md) describes launching it with a privately supplied task file and API credentials. [The export and verification script](report.py) builds the public evidence from the historical files and calculates the summary. The private exports do not record a source commit for each API request. With the original task file, we verified the initial user-message hashes and scores for all 1,380 trials against the archived code.

From the repository root, reproduce the published table without making API requests:

```sh
python3 benchmarks/openrouter-138/report.py verify
```

To independently recheck exact matches and original prompt identity, supply the same task-level validation file. Each reference record must have `task_id`, `train`, and `test` containing input/output pairs. Reference outputs are used only for scoring.

```sh
python3 benchmarks/openrouter-138/report.py verify --reference /path/to/arc_validate.jsonl
```

To reconstruct this publication from the archived API exports:

```sh
python3 benchmarks/openrouter-138/report.py export \
  /path/to/combined-validation-20261016.json \
  /path/to/cancelled-checkpoint-20261016T183008.json \
  /path/to/batch-settings.json /path/to/new-output
```
