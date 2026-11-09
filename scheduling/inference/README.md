# Inference batching and restart state

These standard-library Python helpers implement the admission and checkpoint rules used by the inference worker. They can be used inside any single-writer Slurm worker on a POSIX filesystem. Each model and decoding strategy gets a separate journal. Model loading, tokenization, decoding and scoring remain the responsibility of the workload adapter.

## Admission

A task supplies `task_id`, the exact number of input tokens in `prompt_tokens`, and its original output `budget`. Pass the model context limit and a list of increasing length-band endpoints ending at that limit. The worker uses prompt length plus any saved generation prefix to determine the current input width. Similar-length tasks must lie in the same band, with longest width at most 1.35 times the shortest.

Each band starts at width 8. A completed 8-row batch can lead to a 12-row probe, followed by 16. An out-of-memory attempt caps that band's next width at the next smaller permitted size from 1, 2, 4, 8, 12, and 16. The attempt retains elapsed time, task IDs, run identity and hardware identity. Only attempts on compatible hardware affect future batch limits. A singleton OOM blocks that task on the same hardware. Other tasks remain eligible. A successful run may still leave blocked tasks; report these separately from completed tasks.

## Worker integration

Choose a run identity that covers the model revision, tokenizer, precision, decoding policy and task manifest including exact tokenized prompts and budgets. Use the same persistent journal directory when resuming. Hardware identity should include device model/count, available memory class and placement. Keep the model resident across successive batches. Within `journal.writer()`:

1. Read `journal.states()` and `journal.attempts()`. Call `choose_group(tasks, states, attempts, context=..., bands=...)`. A return value of `None` means there are no runnable tasks.
2. Call `journal.start(group)` before invoking the model. Replay partial tokens with their original prompts. Limit the current **physical call** to `group["call_token_ceiling"]` and each row's remaining budget. A row that hits the physical ceiling can resume in a different batch.
3. Use `journal.checkpoint(task, accumulated_tokens, reason)` at durable prefix intervals and as soon as a row finishes. Use `checkpoint` for a partial row, `eos` for its true stop token, and `token_limit` when it exhausts its original allowance. The adapter owns EOS validation. Saved terminal rows are immutable.
4. Close the attempt with `journal.finish(path, status, elapsed_seconds=...)`. Use `complete` for a fully finished group, `segment_complete` for a successful physical context boundary with surviving rows, `oom` on a device memory exception and `interrupted` for a graceful stop. Mark unexpected exceptions as `failed` and propagate them. Continue selecting groups until the deadline or exhaustion.

The journal uses file locks and atomic, checksummed records. A fresh worker learns its admission limits from persisted attempts and resumes from the last durable tokens. Uncommitted computation may be repeated after a crash, and a sampled continuation can change trajectory after regrouping. Keep prompt tokens and generated data on storage approved for the workload. Do not commit runtime journal directories to version control.

## Node-local model staging

`staging.local_snapshot(source, manifest, scratch, workers=32)` copies weight files in parallel, verifies each destination file against the expected SHA-256 and byte count, and yields the verified snapshot path plus timing information. The number of workers is capped by available CPU affinity and file count. Prepare the manifest from a trusted source; computing it on every allocation moves that cost into startup. Point `scratch` at the job's node-local scratch directory. Keep the context open for the model lifetime, then let it remove the temporary copy. Copy time is part of allocated time.

## Run the CPU examples

From this directory run `python3 -m unittest -v`. The tests use synthetic prompts and files. They demonstrate durable skips, probe and backoff behavior, bounded long-context calls, hash verification, and cleanup.
