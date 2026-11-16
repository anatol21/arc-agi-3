# Python exploration before ARC program refinement

Proposed development experiment, base NF4 on four H100s. Test the effect of offering bounded exploratory Python before the first candidate under the same task and allocation budgets. The unit is one internal validation task. Public competition scores and the historical three-GPU run provide context only.

## Paired panel

Eight tasks from the 128 internal validation IDs. Sort base direct-answer prompt lengths (a pre-existing proxy for Python prompt size), breaking ties by task ID. Split into four equal bins. In each bin take the two smallest SHA-256 hashes of UTF-8 "explore-v1:" plus the task ID. Assign first to X and second to Y. This uses input lengths only; the internal panel has prior experimental exposure. Record actual Python input lengths at staging. Start every condition from the initial task, including any task previously completed in the historical run.

| Half | Length quartile | Task | Proxy prompt tokens |
|---|---:|---|---:|
| X | 1 | `29c11459` | 711 |
| Y | 1 | `e633a9e5` | 391 |
| X | 2 | `1b60fb0c` | 1569 |
| Y | 2 | `5792cb4d` | 1161 |
| X | 3 | `342dd610` | 2289 |
| Y | 3 | `6ca952ad` | 2681 |
| X | 4 | `6455b5f5` | 3765 |
| Y | 4 | `d93c6891` | 3353 |

## Conditions and resource assignment

A uses the existing refinement procedure with the shared thinking-enabled template. B adds optional exploration before the first complete program. Give B read-only visible demo pairs and query inputs as ordinary list data. Each call starts a fresh isolated CPU process with the same standard library restrictions as the existing worker. Return bounded stdout or an error. At most two calls, 512 generated tokens per call including syntax and delimiters, 256 returned tokens per call. Charge errors and malformed calls. Allow an immediate complete solver. After exploration, keep observations visible for candidate 1 and freeze the rule for later turns: use the original revision prompt, previous candidate and demo feedback. The checker and selector use the same demonstration-only rules in both conditions. Query answers remain with the scorer.

Both use the same weights and tokenizer, adapter disabled, sampled decoding, thinking enabled with explicit xhigh effort in both conditions, context 18,432, at most four complete candidates and 8,192 generated tokens per candidate subject to available context. Sample at T=0.6 during reasoning and T=0.3 after the reasoning delimiter for emitted solver code and, in B, exploratory calls. The tool-call temperature is our extension. Use temperature-only sampling (top_p=1, top_k disabled); record the resolved sampler. Pair random seeds across arms by task and candidate and persist per-row RNG state at checkpoints. Verify the phase switch and restart before running. L043 motivates the post-thinking setting. Charge all reasoning, tool-call syntax, and final code tokens. Proposed engineering guard: allow up to 4,096 reasoning tokens per candidate within the effective candidate allowance, reserving at least half that allowance for emitting code. Share this reasoning allowance across exploratory turns and the first candidate in B; the 512-token call limit covers emitted call text, excluding reasoning. At the cap transition to the answer/tool channel using the verified native template. Qualify the transition and trace accounting on a non-panel example before committing the paired allocations. The 4,096 setting is a pilot choice, not a published optimum. Charge exploration against the same total ceiling of 32,768 generated tokens per task. Track prefill, tool output and CPU wall time separately; they consume context or wall time. Log context-limited candidates. No helper library or retrieval in either condition.

Proposed four 30-minute allocations, one per condition/block: A-X, B-X, B-Y, A-Y. Each gets identical initial task order within its block (ascending actual control prompt length), batch cap four, same admission policy, checkpoint interval, early saves, CPU path, cold staging and wall-time stop margin. Log actual group membership, peak VRAM, OOMs, time to model ready and to first scored task. Later batch shapes can diverge after tool calls. Maximum eight additional GPU-hours, four per condition. Truncated tasks remain in the assigned denominator. If either block cannot be run for both conditions, mark the comparison incomplete.

Before submission: adapt the three-device loader, check four-device placement and base-only identity; compare a fixed short greedy control prefix on three versus four GPUs using an archived reference. Report any numerical disagreement. Verify probe isolation, timeout, output truncation, no answer-key access, and resume both before and after a tool return using an idempotent call journal. Qualification cost must be included in or explicitly budgeted separately from pilot timings. Do not resume or mix generation from the ongoing campaign into this panel.

## Analysis and decision

Primary: newly solved exact tasks out of eight assigned per condition by deadline. Count and list B-only, A-only, shared and neither, separating scored-incorrect from unfinished. Report paired net tasks and exact paired uncertainty (exact McNemar test on discordant pairs when present); this small exploratory panel cannot establish general validation gains. Report both first-candidate and selected solutions, executable first candidates, completed/scored tasks, solves and completions per actual allocated GPU-hour including startup, token usage and probe participation. Report projected use only separately from actual Slurm accounting. Predeclared signal for a larger replication: at least one net additional exact solve without lower solves per GPU-hour. If coverage is insufficient for a useful comparison, report feasibility and attrition. A larger independent panel is needed to infer an accuracy gain.

Existing evidence: L042 reports an offline reasoning-effort ablation. E007 documents capped thinking-enabled generations in our runner. Both arms use bounded thinking; the paired contrast estimates exploratory Python conditional on that setting. E031 (historical base first 5/19 versus selected 10/19) motivates retaining revision in both arms. E008 and E030 show that missing executable code and time limits can dominate. E027 found numerical differences with batch shape. L038 and L039 report verification and helper-library evidence in RC8; neither isolates exploratory calls. L040 reports gains with more aggregate tokens, motivating a shared ceiling. This pilot isolates the practical value of offering early Python within our existing solver. Sources: research/evidence/EVIDENCE_TABLE.md, artifacts/cluster/arc-flash-python-nothink-v2/controller.py, artifacts/cluster/arc-python-base-speed-8519440/audit.json, RC8 author write-up §§1,5.

## Qualification retry, 11 November

E032 records the first qualification stopping before an ARC task. E007 motivated the forced reasoning transition. The saved call reached Python code but used a python fence for its print statement. Replay the saved call on two distinct grids, then require the model to return a solver using a fresh private observation. A complete python-fenced program defining solve remains a solver. Prior GPU cost was 370 seconds × four H100s. Four equal 28-minute replacement slots keep the maximum total at 7.878 GPU-hours. The panel, arm order and selection rules are unchanged.

## Panel initialization retry

The v2 live control passed after the E032 fence repair, but job 8534498 failed before the first ARC batch. Use the source benchmark ID validated by ready() and hash the complete staged recipe for task provenance. Before submission, exercise initialization, tokenization, one simulated CPU decoding step, checkpointing and restoration for all four real blocks. Keep E031's shared revision procedure and the paired task assignment. Four 26-minute slots on four GPUs cost at most 6.9334 additional GPU-hours.

## Step-memory retry after 8544056

E033 confirmed observation use in a synthetic control; the BX failure occurred during loader startup, before either condition's task logic. AX and BX each received 773200 MiB in the allocation, but BX step .0 received 2 GiB versus 773200 MiB at AX step .0. Run the worker in the batch step and log inherited memory cgroup limits before staging. Preserve AX from v3; retry BX for 26 minutes and allocate 22 minutes each to BY and AY. Compare paired conditions within each half; the Y half now has less runtime than X. Four prior jobs consumed 2935 seconds on four GPUs. Remaining reservations total 4200 seconds on four GPUs.

## Timeout settlement

Job 8544341 reached its durable BX ended record after passing qualification, with zero finished tasks. Unlike the loader failure E034, this block has a valid end record matching the allocation. Accept terminal TIMEOUT on this evidence and retain the original per-task outcomes. Advance to BY then AY without another X allocation. Actual charges before BY were 4509 seconds on four GPUs.

## Forced-code continuation

E035 documents zero panel solver candidates in the four earlier allocations, despite E033 demonstrating observation use on a synthetic control. For the continuation off E035 journals, check the hash of each parent, restore its token prefix and CUDA sampling state, and force the native end-of-reasoning token. Verify that the boundary consumes no RNG draw. Keep original prompts and matched post-thinking temperature. Force code mode on subsequent turns, mask native reasoning tags and cap each emitted response at 1,536 tokens (probe allowance remains 512). A code channel may still contain malformed or natural-language text; only completed valid Python executes. Each four-task batch gets up to twelve minutes generation and three minutes to finish execution/scoring, capped at fifteen minutes elapsed from verified replay. Charge time spent on intermediate executions to those fifteen minutes. Allow less generation if startup is slow. At deadline select the strongest candidate already evaluated against demonstrations. Distinguish unfinished attempts from scored errors. Retain the historical campaign separately, report new and cumulative GPU cost. Maximum four single-node, four-H100 dev jobs of 25 minutes, issued serially AX/BX/BY/AY with a CPU watcher and a cumulative 14.59 GPU-hour cap.
