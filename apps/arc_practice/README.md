# ARC Practice

Local ARC manual practice and local Ollama benchmarking for ARC-style JSONL
datasets. The app binds only to `127.0.0.1`; it never uploads, copies, or
modifies dataset files.

## Start the packaged environment

```sh
python3 apps/arc_practice/server.py \
  --package-dir /path/to/arc-synthetic-training-package-v1/data
```

The dataset selector provides:

- **ARC Train** — 1,008 official training tasks.
- **ARC Validate** — 138 protected evaluation tasks and the default benchmark dataset.
- **Synthetic ARC Train** — 199,899 generated tasks.

Use `--port 8766` to choose a different local port, `--no-browser` to stop the
browser opening automatically, or `--dataset PATH` to serve one ARC task JSONL
instead of the three packaged files.

## Grid limit and data handling

ARC Practice only serves tasks for which **every** train/test input and output
grid is at most 30 rows by 30 columns. This is a runtime filter, not a cleanup:
the JSONL sources remain untouched.

ARC Train and ARC Validate are entirely within the limit. In Synthetic ARC
Train, 198,293 tasks are eligible and 1,606 unusually large generated tasks are
excluded from both manual practice and benchmark selection.

The server streams each JSONL once to build byte offsets; task bodies are loaded
only when needed. It validates grid shape while indexing and validates colours
and values before a task is served or scored.

## Manual practice

The manual tab recreates the ARC solving workflow: demonstration pairs, test
input, palette, edit/select/flood-fill tools, rectangular selection, `C` copy,
`V` paste, resize, reset, and server-side checking. Browser session scores are
not saved anywhere.

The browser receives demonstration pairs and test inputs only. Test outputs stay
on the local server; a submission receives only correct/incorrect feedback.

## Ollama benchmark

Open the **Ollama benchmark** tab, select a local model, task selection settings,
and optional prompt modifier, then start a sequential run. Ollama must already
be running locally at `http://127.0.0.1:11434` (override with `--ollama-url`).

Each model request receives training demonstrations and test inputs, never test
answers. The required response is strict JSON:

```json
{"outputs": [[[0]]]}
```

There must be one rectangular 0–9 grid in `outputs` for every test input. The
server records exact task/test-grid correctness, parsing failures, Ollama errors,
and latency. Runs are held in memory only until the server stops. Download JSON
or CSV explicitly from the completed/cancelled run; neither export contains the
protected expected output grids.

Use **Inspect** beside a completed task to view its raw local-model response and
any successfully parsed prediction grids. When an Ollama model returns its
separate `thinking` field, the inspector also shows that reasoning trace. This
is loaded on demand rather than included in the polling feed; it never includes
an expected output grid.

The inspector also includes the exact request sent to Ollama: the selected model,
generation settings, response contract, demonstrations, test inputs, and prompt
modifier. Protected test outputs are deliberately excluded.

The benchmark form has an editable **Core instruction** field. It replaces the
initial task-solving instruction for that run and is retained in the request
inspector and detailed JSON export. The strict JSON response contract stays
fixed so every response can be evaluated consistently. **Prompt modifier** is
for optional additional instructions appended after the task data.

## Supported input

Every JSONL record must include a non-empty `task_id`, `source`, `train`, and
`test`, where each pair has rectangular integer grids with colours 0–9.
Pair-level records such as `google_cleaned.jsonl` are rejected because they do
not represent ARC task episodes.
