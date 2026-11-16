# Direct answers on the 138 public held-out ARC tasks
16 November 2026. Unit of analysis is one task per condition. Every query grid must match. First completion in the original source precedence is primary. No output repairs are scored.

## Evidence
At release [e37ec8b](https://github.com/anatol21/arc-agi-3/tree/e37ec8b840155c28c843b175cf1f2116d5c5d8cc/benchmarks/qwen3.8-flash-next), the NF4 archive contains 333 terminal source observations including 57 extra records, some carried forward. The BF16 archive contains 276 responses. The two published verification scripts were run with the local ARC-AGI and ARC-AGI-2 evaluation trees. Both completed and reproduced the recorded score flags. The accompanying [audit.json](audit.json) classifies every archived observation, with source, text hash and task-level outcome. Run [analyze.py](analyze.py) with the pinned repository checkout as its argument. These are direct-answer experiments. Python refinement and API prompts are separate protocols.

## Same nominal token allowance, different lengths

The context cap was 18,432 and the answer allowance was min(8,192, context cap minus prompt length). Numbers below are generated tokens. NF4 used greedy decoding; BF16 base used sampled decoding with one draw per setting, thinking off in all four. NF4 and BF16 used different execution layouts. Token budgets concern the serialized answer, including delimiters and whitespace.

| Task | Prompt | Answer allowance | NF4 base | NF4 checkpoint | BF16 low nucleus | BF16 instruct sampling |
|---|---:|---:|---:|---:|---:|---:|
| e66aafb8 | 7,393 | 8,192 | 36 | 24 | 32 | 32 |
| e95e3d8e | 6,945 | 8,192 | 1,457 | 974 | 1,457 | 1,457 |
| af22c60d | 16,377 | 2,055 | 1,806 | 1,806 | 2,055 | 2,055 |
| 1e97544e | 7,575 | 8,192 | 2,777 | 1,064 | 1,592 | 1,592 |

Only the two BF16 answers on af22c60d hit the cap among these 16 responses. All others ended at EOS.

- **e66aafb8.** The query asks for a missing 3×3 patch of a symmetric grid. Correct rows are 684 / 594 / 894. NF4 base emitted 684 / 594 / 894 / 594 / 684. It recovered the full patch, then extended the reflection by two rows. Its JSON was valid. Checkpoint and both BF16 samples stopped at the correct patch boundary.
- **e95e3d8e.** The 22×22 repeating grid had 46 missing cells. All four responses contained the correct grid. NF4 base placed the row list directly under outputs, omitting the required outer list of query grids. Wrapping it once recovers all 484 reference cells exactly. Only this primary NF4 base task passes that diagnostic. The official base score remains a failure.
- **af22c60d.** Both NF4 arms reconstructed the complete 30×30 grid including 56 missing cells. Both BF16 answers emitted exactly 684 correct leading cell values (22 full rows plus 24 cells), then stopped after a comma. They used comma-space separators where NF4 used compact commas. Instruct sampling also omitted the outer grid list at the start. A longer allowance could test completion; the remaining 216 cells were never observed.
- **1e97544e.** The query requires filling holes while preserving a cyclic sequence with a repeated value along a diagonal. All four answers are valid 23×23 grids. NF4 base missed 35 cells (22 originally blank, 13 visible); each BF16 strategy missed the same 23 (19 blank, four visible). Every wrong value across those answers is one ahead in the 1–9 cycle, including 9→1. Row 5, column 5 should be 6 and all three failures give 7. Checkpoint restored all 73 holes and preserved the 456 visible cells. The outputs support a sequence-alignment error; they do not reveal why the adapter changed the logits.

## Task-level failure breakdown

| Condition | Exact | Valid, correct dimensions, wrong cells | Valid, wrong dimensions on at least one query | Invalid at EOS | Token cap |
|---|---:|---:|---:|---:|---:|
| NF4 base | 4 | 87 | 35 | 8 | 4 |
| NF4 checkpoint-660 | 7 | 90 | 23 | 7 | 11 |
| BF16 low nucleus | 5 | 94 | 27 | 8 | 4 |
| BF16 instruct sampling | 5 | 92 | 28 | 7 | 6 |

Each row totals 138. Wrong dimensions means a parseable rectangular grid of the wrong size. Token caps have their own column even when the parser also reports invalid JSON. Among the eight non-truncated invalid NF4 base responses, four failed grid count, two rectangularity and two JSON parsing. Among the seven checkpoint cases, three failed grid count and four rectangularity. NF4 base copied all query inputs unchanged on 15 failed tasks versus eight for checkpoint. On 67b4a34d the primary checkpoint response was one cell off and two later source records were exact. This was the only NF4 task/model pair whose solve status changed across records.

## Concentration of solves
In 14 of the 138 tasks, every demonstration keeps grid dimensions, changes at least one cell, and preserves every nonzero input cell. Six of seven checkpoint solves and all four NF4 base solves fell in those 14. Each BF16 strategy solved four of the 14. The remaining checkpoint/BF16 solve was e66aafb8, which asks for the missing patch by itself. This retrospective filter tests a property of demonstration grids. It does not establish a general mechanism or training-data effect.

## Other evidence and interpretation
In the separate OpenRouter study, each puzzle had ten trials across five prompt strategies. Correct trials were 9/10 for e66aafb8, 10/10 for e95e3d8e, 9/10 for af22c60d and 5/10 for 1e97544e. For 1e97544e four other API trials were labeled malformed and one incorrect. Different served model, prompts, settings and retry policy prevent attributing these differences to precision. None of these four puzzles appeared in the 12-task BF16 greedy pilot. The full comparison combines precision and decoding changes, and the saved direct answers contain no reasoning traces. The cell and schema diagnoses explain scores; only a matched precision comparison can test whether quantization caused them.
