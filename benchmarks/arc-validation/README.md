# Held-out ARC validation task profiles

The [HTML report](task-costs.html) contains a worked example and a table of all tasks in prompt-length order. To view its grid diagrams, download the HTML file and open it in a browser. The [JSONL profiles](task-costs.jsonl) contain one record per task.

Both model arms answered the same 138 public tasks (185 query grids) with greedy decoding, thinking disabled and direct JSON output. The base arm used the native NF4 loading path. The fine-tuned arm used the same base with the checkpoint-660 adapter. The profile uses the first completed response for each task and arm across serial and batched runs.

## Prompt and example

Both arms received this instruction as the start of a single user message, followed by compact JSON containing the demonstration pairs and the query inputs.

```text
Solve this ARC task. Infer the transformation from the training examples and apply it to every test input.

Return exactly one complete JSON object with this schema and no prose, markdown, explanation, or code fence:
{"outputs": [[[0]]]}

The `outputs` array must contain one rectangular 0–9 integer grid for each item in `test`, in the same order. Write every row and every cell explicitly. Never abbreviate repeated values or rows, and never use ellipses (such as `...` or `......`), placeholders, or omitted content.

Task:
```

Task `e8686506` has two demonstrations (13 × 13 → 5 × 5 and 13 × 13 → 8 × 5) and one 15 × 15 query. Its 628 visible cells produced a 1,417-token prompt after chat formatting. The base model generated 255 tokens and checkpoint-660 generated 156. Both reached EOS and returned a 15 × 5 grid. Neither solved this task.

<details>
<summary>Full user message for e8686506</summary>

```text
Solve this ARC task. Infer the transformation from the training examples and apply it to every test input.

Return exactly one complete JSON object with this schema and no prose, markdown, explanation, or code fence:
{"outputs": [[[0]]]}

The `outputs` array must contain one rectangular 0–9 integer grid for each item in `test`, in the same order. Write every row and every cell explicitly. Never abbreviate repeated values or rows, and never use ellipses (such as `...` or `......`), placeholders, or omitted content.

Task:
{"train":[{"input":[[8,8,8,8,8,8,8,8,8,8,8,8,8],[8,8,8,8,8,4,8,8,8,8,8,8,8],[8,8,8,8,8,8,8,8,8,8,8,8,8],[8,8,3,3,8,3,3,8,8,4,8,8,8],[8,8,3,8,8,8,3,8,8,8,8,8,8],[8,8,3,8,8,8,3,8,8,8,8,8,8],[8,8,8,3,8,3,8,8,8,8,8,4,8],[8,8,3,3,8,3,3,8,8,8,8,8,8],[8,8,8,8,8,8,8,8,8,8,8,8,8],[8,8,8,8,8,8,8,8,8,8,8,8,8],[8,8,1,1,1,8,8,8,6,8,8,8,8],[8,8,1,1,1,8,8,8,6,8,8,8,8],[8,8,8,8,8,8,8,8,8,8,8,8,8]],"output":[[3,3,4,3,3],[3,1,1,1,3],[3,1,1,1,3],[4,3,6,3,4],[3,3,6,3,3]]},{"input":[[3,3,3,3,3,3,3,3,3,3,3,3,3],[3,6,3,3,3,3,3,3,8,8,8,3,3],[3,6,3,3,3,3,3,3,3,8,3,3,3],[3,3,3,1,1,1,3,3,3,8,3,5,3],[3,3,1,3,3,3,1,3,3,3,3,3,3],[3,3,1,1,3,1,1,3,5,3,3,3,3],[3,3,3,1,3,1,3,3,3,3,3,3,3],[3,3,3,2,3,2,3,3,3,3,4,3,3],[3,3,2,2,3,2,2,3,3,3,4,3,3],[3,3,2,3,3,3,2,3,3,4,4,4,3],[3,3,3,2,2,2,3,3,3,3,3,3,3],[3,6,3,3,3,3,3,3,3,3,5,3,3],[3,6,3,3,3,3,3,5,3,3,3,3,3]],"output":[[5,1,1,1,5],[1,8,8,8,1],[1,1,8,1,1],[6,1,8,1,6],[6,2,4,2,6],[2,2,4,2,2],[2,4,4,4,2],[5,2,2,2,5]]}],"test":[{"input":[[4,1,4,4,4,4,4,4,4,4,4,4,4,4,4],[4,4,4,4,4,4,4,4,4,4,4,4,4,4,4],[4,4,4,4,4,8,8,8,4,4,4,4,6,6,6],[4,4,3,4,8,4,4,4,8,4,4,4,4,6,4],[4,4,3,4,4,8,4,8,4,4,1,4,4,4,4],[4,4,3,4,4,8,4,8,4,4,4,4,4,4,4],[4,4,3,4,4,8,4,8,4,4,4,4,4,4,4],[4,4,4,4,4,8,4,8,4,4,4,4,4,4,4],[4,4,4,4,8,4,4,4,8,4,4,4,1,4,4],[4,4,4,4,4,8,8,8,4,4,4,4,4,4,4],[4,3,4,4,4,4,4,4,4,4,4,4,4,4,4],[4,3,4,4,4,4,4,9,4,4,4,4,4,4,4],[4,3,4,4,4,4,4,9,4,4,4,6,4,4,4],[4,3,4,4,1,4,4,4,4,4,6,6,6,4,4],[4,4,4,4,4,4,4,4,4,4,4,4,4,4,4]]}]}
```

</details>

## Reading a profile

`prompt_tokens` includes the chat template. `visible_cells` sums demonstration input cells, demonstration output cells and query input cells. The short, medium and long prompt groups contain 52 tasks (≤4,096 tokens), 62 (4,097–8,192), and 24 (>8,192).

`nf4.base` and `nf4.checkpoint_660` record the observed generated-token counts, token budgets, stop reasons and source identifiers. A response marked `token_limit` consumed its applicable allowance. Per-response generation time is included for serial measurements. A null time means no individual latency was measured.

For fixed batches of eight, sorting by prompt length reduced simulated padded input positions from 1,482,854 to 843,354. The calculation covers prefill padding. It does not measure wall time or BF16 performance.

The example reproduces the visible grids from [ARC-AGI-2 task e8686506](https://github.com/arcprize/ARC-AGI-2/blob/main/data/evaluation/e8686506.json), © 2019 ARC Prize Foundation. The task data is distributed under the [Apache License 2.0](LICENSE-ARC-AGI-2.txt).
