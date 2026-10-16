# ARC Practice

Local ARC manual practice and model benchmarking for ARC-style JSONL
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

## Model benchmark

Open the **Model benchmark** tab and select either a local Ollama model or
OpenRouter. For Ollama, the service must already be running locally at
`http://127.0.0.1:11434` (override with `--ollama-url`). For OpenRouter, enter
an API key and either load the model catalog or enter a model ID directly.
For OpenRouter, choose one to sixteen parallel requests to run independent ARC
tasks concurrently. OpenRouter models may request up to 131,072 output tokens;
the selected model and provider ultimately enforce their own limits.

The OpenRouter key is sent only from this local page to the local benchmark
server and then to OpenRouter for the run. It is held in process memory only:
it is not written to disk, shown in the request inspector, or included in JSON
or CSV exports. OpenRouter requests use its chat-completions API with bearer
authentication and the model's ID.

OpenRouter completions stream back to the server. While a request runs, the
page shows whether it is connecting, waiting for the model, reasoning, or
receiving an answer, with elapsed time and received character counts. The
configured timeout covers the entire attempt, including provider keep-alives.
JSON responses from providers that do not stream are also accepted.

On macOS, if Python has no configured root certificates, HTTPS uses the existing
system certificate bundle at `/etc/ssl/cert.pem`. Certificate and hostname
verification remain enabled. Explicit `SSL_CERT_FILE` / `SSL_CERT_DIR` settings
are respected; certificate verification failures are not retried.

When reasoning is enabled and Maximum tokens is at least 2,048, the request
allocates half that limit to reasoning through `reasoning.max_tokens`, leaving
room for the final answer on providers that support this setting. Smaller
limits retain provider-default reasoning settings. The inspector records the
actual request, generation ID, provider, finish reason, and returned usage.
An exhausted token budget or a reasoning-only completion is reported as an
incomplete model answer, retaining its trace, rather than losing it as a generic
provider error. Actual upstream errors, including errors inside HTTP 200
responses, retain their message and retryability.

### Parallel experiment batches

The **Parallel experiment batch** panel compares one to five OpenRouter
configurations against the same immutable ARC cohort. Its screening defaults
are five configurations, ten seeded shared tasks, and three trials per task
(150 logical requests). The global request cap defaults to four and applies to
the entire batch, rather than allowing each configuration to saturate the
provider independently.

Every experiment may use a different model, sampling settings, output-token
limit, timeout, reasoning setting, core instruction, and prompt modifier. The
scheduler rotates configurations across task/trial slots, records queue time
separately from provider time, and retries one transient provider failure once.
Malformed JSON and invalid output grids are not retried or hidden: they count
as failed trials alongside provider errors.

Each scorecard reports exact trial accuracy, stable solves (all trials correct),
solved-at-least-once tasks, test-grid accuracy, valid-response rate,
provider-success rate, p50/p95 queue and provider latency, and returned usage
or cost data when OpenRouter supplies it. The task/trial matrix and response
inspector retain the evidence needed to distinguish a strong configuration
from a lucky one-shot outcome.

Only one experiment batch may run at a time. Batch state and exports remain in
memory until the server stops; exports include configuration, task IDs, timing,
and failure classifications but never API keys or protected target grids.

Enable **Request reasoning trace** only for models that support it. For Ollama,
it sends `"think": true`; for OpenRouter, it sends
`"reasoning": {"enabled": true}`. The inspector displays the returned trace
or reasoning details separately from the strict JSON answer.
The setting is off by default because unsupported models reject it; in
particular, Qwen3-Coder is a non-thinking model. Thinking also uses part of the
maximum-token budget.

Each model request receives training demonstrations and test inputs, never test
answers. The required response is strict JSON:

```json
{"outputs": [[[0]]]}
```

There must be one rectangular 0–9 grid in `outputs` for every test input. The
server records exact task/test-grid correctness, parsing failures, provider errors,
and latency. Runs are held in memory only until the server stops. Download JSON
or CSV explicitly from the completed/cancelled run; neither export contains the
protected expected output grids.

Use **Inspect** beside a completed task to view its raw local-model response and
any successfully parsed prediction grids. When an Ollama model returns its
separate `thinking` field, the inspector also shows that reasoning trace. This
is loaded on demand rather than included in the polling feed; it never includes
an expected output grid.

The inspector also includes the exact request sent to the selected provider:
the selected model, generation settings, response contract, demonstrations,
test inputs, and prompt modifier. Secrets and protected test outputs are
deliberately excluded.

The benchmark form has an editable **Core instruction** field (up to 32,768
characters). It replaces the initial task-solving instruction for that run and
is retained in the request inspector and detailed JSON export. The strict JSON
response contract stays fixed so every response can be evaluated consistently.
**Prompt modifier** accepts the same size and is for optional additional
instructions appended after the task data.

## Supported input

Every JSONL record must include a non-empty `task_id`, `source`, `train`, and
`test`, where each pair has rectangular integer grids with colours 0–9.
Pair-level records such as `google_cleaned.jsonl` are rejected because they do
not represent ARC task episodes.
