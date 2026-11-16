# Proposed comparisons

The four directions on slide 29 are prospective. The 138 public tasks have been scored and individually inspected before. Re-evaluations on them can reveal paired changes. Reserve unseen tasks and rule families before data generation for a subsequent transfer claim.

## Python validation

Finish the two existing arms according to the recorded campaign protocol. For each arm and pass, report distinct attempted tasks, closed attempts without a grid, tasks with at least one returned grid, whole-task exact solves, and GPU-hours. Retain task IDs, emitted prefixes, accepted and executed probe counts, candidate histories, clock time and token consumption. Report error clusters separately. E046 and E047 document different reasons for missing an exact solve. A closed generation with no grid and a wrong grid are different observations. Count the same task only once when reporting cumulative coverage. Compare an arm with access to probes against an arm without access, and show how many actually used that option.

## Two training-corpus interventions

Start two fresh adapters from the same initialization on the original frozen NF4 base with BF16 compute. Retain the original LoRA rank, targets, optimizer and direct-answer format. One corpus consists entirely of generated ARC-style examples, for instance from BARC generators. The other consists entirely of input/output transformations extracted from games and logic puzzles whose rules can produce or verify the answer. Record the parent game, rule, generator version, augmentation and seed for each example. Use comparable input sizes, match supervised answer-token budgets and update counts, and report realized totals and compute. Record the number of distinct rule families in addition to the number of sampled grids.

Partition by parent rule or generator before augmentation. Exclude validation-derived seeds, duplicate source tasks and grid pairs. Audit cross-version ARC overlaps. Include the adapter-off base and the existing real-data checkpoint-660 as references. For a causal comparison to real-data training, match its exposure and optimization schedule in an additional real-data control if those budgets differ. Use the same task serialization, greedy decoding, thinking off, token ceilings and exact scorer in all four conditions. Report gains and losses by task ID, invalid grids and context truncation. Neither training-corpus proposal is represented as having run.

L113 describes the nvbanana rule-generation and verification pipeline. L114 describes other contributors to their final system. L008 records a synthetic-training comparison that failed to outperform a real-task configuration. E019 documents overlap across ARC versions. This comparison asks whether changing the training source improves the same NF4 inference system when the training and decoding budgets are matched. [BARC released generators](https://github.com/xu3kev/BARC)

## Interactive ARC-AGI-3

The one-game engineering test and its observation/action constraints are in `arc3-poc.md`. Validate frame round-trips, legal actions and deterministic replay where supported. Compare policies on the same version, seed and action limit. Track level completion, actions, invalid actions and restarts separately from static exact-grid scores. [ARC-AGI-3](https://arcprize.org/arc-agi/3)
