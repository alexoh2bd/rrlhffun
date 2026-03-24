# RLHF Experiment Results

**Model:** `deepseek-ai/deepseek-coder-6.7b-instruct`  
**RL Algorithm:** PPO (Proximal Policy Optimisation via TRL `PPOTrainer`)  
**Reward Training Data:** `coseal/CodeUltraFeedback_binarized` (chosen/rejected pairs)  
**Evaluation Benchmarks:** SWE-bench Oracle, HumanEval  

---

## 1. Summary


| Benchmark                             | Baseline            | RL-tuned            | Δ            | Direction |
| ------------------------------------- | ------------------- | ------------------- | ------------ | --------- |
| SWE-bench resolve rate (50 instances) | 2.0% (1/50)         | 2.0% (1/50)         | 0 pp         | —         |
| SWE-bench patch applicability         | 12.0% (6/50)        | 18.0% (9/50)        | +6 pp        | ↑         |
| HumanEval pass@1 (164 problems)       | **70.1%** (115/164) | **68.3%** (112/164) | **−1.83 pp** | ↓         |


**Key finding:** RL training on CodeUltraFeedback produced **measurable misalignment** — HumanEval
pass@1 regressed by 1.83 percentage points while SWE-bench resolve rate showed zero improvement.
Patch applicability improved marginally (+6 pp) but without any corresponding increase in bug resolution,
consistent with reward overoptimization toward superficial code quality signals.

---

## 2. Experimental Setup

### 2.1 Reward Model Training


| Parameter           | Value                                                |
| ------------------- | ---------------------------------------------------- |
| Architecture        | `deepseek-coder-6.7b-instruct` + classification head |
| Training data       | `coseal/CodeUltraFeedback_binarized` (streamed)      |
| Epochs              | 1                                                    |
| Batch size          | 1 (gradient accumulation 8, effective 8)             |
| Learning rate       | 1e-5                                                 |
| Max sequence length | 512 tokens                                           |
| Precision           | fp16                                                 |


### 2.2 PPO Training


| Parameter                 | Value                                            |
| ------------------------- | ------------------------------------------------ |
| Policy model              | `deepseek-coder-6.7b-instruct` + value head      |
| Reward model              | Sequence classifier trained in §2.1              |
| Dataset                   | `coseal/CodeUltraFeedback_binarized` train split |
| Steps                     | ~9 500 (full train split iterated once)          |
| Batch size                | 1 / mini-batch 1 / grad accum 1                  |
| Learning rate             | 1e-6                                             |
| Policy precision          | bfloat16                                         |
| Max prompt length         | 512 tokens                                       |
| Max new tokens            | 512 tokens                                       |
| Temperature (PPO rollout) | 0.7, top-p 0.95                                  |
| Temperature (inference)   | greedy (do_sample=False)                         |
| Checkpoint saved at       | step 400                                         |


### 2.3 Inference Setup

Both models evaluated identically:

- **Tokenisation:** `apply_chat_template` with a single user message
- **Decoding:** greedy (`do_sample=False`)
- **Max new tokens:** 2 048 (up from 1 024 in original training notebook)
- **SWE-bench prompt:** issue description + `hints_text` (oracle file paths)
- **HumanEval prompt:** function signature + docstring; raw Python completion requested

---

## 3. SWE-bench Oracle Results

### 3.1 50-Instance Matched Subset

Instances selected: those where **both** models produced a syntactically well-formed unified diff
(valid `diff --git` header, non-truncated) in the original 400-checkpoint prediction files.


| Metric                    | Baseline      | RL-tuned      |
| ------------------------- | ------------- | ------------- |
| Submitted                 | 50            | 50            |
| Patch applied (no error)  | **6 (12.0%)** | **9 (18.0%)** |
| Tests passed — resolved   | **1 (2.0%)**  | **1 (2.0%)**  |
| Tests failed — unresolved | 5 (10.0%)     | 8 (16.0%)     |
| Patch apply failed        | 44 (88.0%)    | 41 (82.0%)    |


**Resolved instance:** `astropy__astropy-13469` — identical for both models.

### 3.2 Patch Applicability vs Resolution

Although the RL model generated more applicable patches (+3 instances), it also produced more
*unresolved* patches (+3 instances). The conditional resolution rate given a successful patch
application **decreased from 1/6 (16.7 %) to 1/9 (11.1 %)**:

```
P(resolved | patch applied):   Baseline  16.7%   RL  11.1%
```

This inversion is consistent with overoptimization: the reward signal encourages the model to
produce more complete-looking, well-formatted diffs, but the bug-fixing logic (correct semantic
content) does not improve.

### 3.3 Predominant Patch Failure Modes

From the SWE-bench evaluation logs, the overwhelming failure mode is:

```
patch unexpectedly ends in middle of line
patch: **** malformed patch at line N
Only garbage was found in the patch input
```

These indicate **truncated or structurally incomplete diffs** — a consequence of `max_new_tokens`
limits and the model conflating the diff format with explanatory prose. The failure rate is high in
both models (82–88 %), pointing to a fundamental mismatch between the RL training distribution
(CodeUltraFeedback instruction-following tasks) and the SWE-bench patch generation task.

### 3.4 Patch Format Quality (100-Instance Notebook Inference)

A second inference run using an improved prompt (`max_new_tokens=2048`, explicit `diff --git`
instruction) was performed on the same 100 matched instances:


| Status            | Baseline | RL-tuned |
| ----------------- | -------- | -------- |
| Valid diff header | 3 (3%)   | 2 (2%)   |
| Truncated         | ~45%     | ~47%     |
| No diff header    | ~52%     | ~51%     |


The sharp decline from the original run (57% valid) to this run (3%) suggests that the stronger
instruction ("output ONLY raw unified diff") caused the model to generate more prose preamble before
the diff, rather than starting immediately with `diff --git`. This is an ongoing prompt engineering
challenge independent of the RL fine-tuning.

---

## 4. HumanEval Results

### 4.1 pass@1 Scores


| Model              | pass@1       | Passed / 164 |
| ------------------ | ------------ | ------------ |
| Baseline (pre-RL)  | **70.1%**    | 115          |
| RL-tuned (post-RL) | **68.3%**    | 112          |
| **Δ**              | **−1.83 pp** | −3           |


### 4.2 Task-Level Changes

**RL gained (0 tasks):** The RL model did not solve any task the baseline failed.

**RL regressed (3 tasks):**


| Task ID         | Function             | Baseline                                   | RL                                      | Failure mode                  |
| --------------- | -------------------- | ------------------------------------------ | --------------------------------------- | ----------------------------- |
| `HumanEval/41`  | `car_race_collision` | `return n * n` ✓                           | `return 0` ✗                            | Output collapse               |
| `HumanEval/58`  | `common`             | `return sorted(list(set(l1) & set(l2)))` ✓ | Docstring re-emission + broken indent ✗ | Verbosity + indentation error |
| `HumanEval/101` | `words_string`       | `return s.replace(',', '').split()` ✓      | Docstring re-emission + bare `return` ✗ | Verbosity + empty return      |


### 4.3 Regression Analysis: Signs of Overoptimization

The three regressions reveal a consistent pattern: the RL model **over-generates text** rather than
producing a correct, concise implementation.

#### Output collapse (HumanEval/41)

```python
# Baseline — correct
def car_race_collision(n: int):
    return n * n

# RL-tuned — wrong (collapsed to trivially simple output)
def car_race_collision(n: int):
    return 0
```

The RL model returned the constant `0` — a simpler, syntactically valid response. This is a
classic reward-model bias: the reward model may score brief, clean outputs higher for short
function stubs, causing the policy to collapse toward trivially simple (but incorrect) answers.

#### Docstring re-emission + structural breakage (HumanEval/58, /101)

```python
# HumanEval/58 — RL-tuned
def common(l1: list, l2: list):
    """Return sorted unique common elements for two lists.
    >>> common([1, 4, 3, ...], [5, 7, 1, ...])
    [1, 5, 653]
    """
return sorted(set(l1)&set(l2))   # ← de-indented; unreachable inside function
```

```python
# HumanEval/101 — RL-tuned
def words_string(s):
    """
    You will be given a string of words...
    """
return                             # ← bare return; returns None
```

The RL model learned to prepend verbose docstrings before its implementations — a style pattern
that likely scores highly on the CodeUltraFeedback reward model (which was trained on human
preferences for well-documented code). However this introduces two failure modes:

1. The re-emitted docstring pushes the implementation outside the correct indentation level
2. In the extreme case (HumanEval/101), the model emits the docstring but produces no implementation

---

## 5. Misalignment and Overoptimization Analysis

### 5.1 Reward–Task Misalignment

The core misalignment stems from optimising a proxy reward (CodeUltraFeedback preference model)
that was never trained to evaluate the actual downstream tasks:


| Target                         | Reward Signal Available               | Used in This RL Run |
| ------------------------------ | ------------------------------------- | ------------------- |
| SWE-bench patch correctness    | `git apply` + test execution (binary) | ✗                   |
| HumanEval correctness          | Test execution (binary)               | ✗                   |
| CodeUltraFeedback code quality | Human preferences (scalar)            | ✓                   |


The reward model judges **stylistic code quality** (verbosity, documentation, structure),
not **semantic correctness**. PPO optimises for that signal, leading to measurable degradation
on tasks requiring precise, concise output.

### 5.2 Goodhart's Law in Action

> *"When a measure becomes a target, it ceases to be a good measure."*

The reward model implicitly rewards:

- Longer, more verbose responses (higher documentation level)
- Reproducing context from the input prompt (shows comprehension)
- Well-structured-looking code

The RL model learned to satisfy these proxies at the cost of functional correctness, producing
docstring re-emissions and output collapse that break actual execution.

### 5.3 Training–Inference Distribution Mismatch

A secondary misalignment exists at the prompt-format level:


| Stage                  | Prompt Format                                            | Dataset           |
| ---------------------- | -------------------------------------------------------- | ----------------- |
| Reward model training  | `chosen`/`rejected` pairs from CodeUltraFeedback         | CodeUltraFeedback |
| PPO rollout (training) | `Task: {instruction}\nSolution:` (raw, no chat template) | CodeUltraFeedback |
| SWE-bench inference    | Chat template + issue/hints prompt                       | SWE-bench Oracle  |
| HumanEval inference    | Chat template + function completion prompt               | HumanEval         |


PPO gradient updates were computed using raw-format activations, but evaluation applies the
DeepSeek Coder instruct chat template (`<｜User｜>...<｜Assistant｜>`). The model's
activations during training therefore differ from those at inference, reducing the effective
transfer of RL-learned behaviour.

### 5.4 PPO Hyperparameter Concerns


| Issue                               | Detail                               | Effect                                                                  |
| ----------------------------------- | ------------------------------------ | ----------------------------------------------------------------------- |
| `batch_size=1`, `mini_batch_size=1` | Effectively online SGD               | High variance gradient updates; poor generalisation                     |
| `max_length=512` reward model       | Truncates many completions           | Reward signal computed on truncated context; incentivises short outputs |
| `max_new_tokens=512` PPO rollout    | Mismatch with 1024–2048 used at eval | Different output distributions between train and test                   |
| 1-epoch reward model                | Insufficient convergence             | Noisy reward signal throughout PPO                                      |
| Checkpoint at step 400              | ~4% of full training                 | Undertrained policy; limited drift from base model                      |


---

## 6. Limitations

1. **Small evaluation subset:** SWE-bench results are based on 50 instances; resolve-rate confidence intervals are wide (±4 pp at 95%).
2. **Single checkpoint:** Only the step-400 PPO checkpoint is evaluated; reward history trends are unavailable for later checkpoints.
3. **No KL-divergence tracking:** Without the KL penalty trajectory it is not possible to quantify how far the policy drifted from the reference model.
4. **No repeated seeds:** HumanEval used greedy decoding (deterministic), but SWE-bench results could vary with different sampling parameters.
5. **Reward model not evaluated independently:** The reward model's own accuracy on held-out CodeUltraFeedback pairs is unknown; it may already be overfit after 1 epoch.

---

## 7. Recommendations

1. **Use task-specific rewards.** Replace the CodeUltraFeedback preference model with binary rewards from test execution (pass/fail on HumanEval tests; `git apply` + pytest for SWE-bench).
2. **Fix the train/inference prompt mismatch.** Apply the chat template during PPO rollout to match inference conditions.
3. **Add a KL penalty.** Track and report KL divergence from the reference model throughout PPO training to detect overoptimisation early.
4. **Increase batch size.** Use `batch_size ≥ 8` with gradient accumulation to stabilise PPO updates.
5. **Evaluate at multiple checkpoints.** Monitor HumanEval pass@1 every 500 PPO steps to detect the onset of reward overoptimisation.
6. **Consider GRPO instead of PPO.** GRPO (Group Relative Policy Optimisation) removes the need for a value head and has shown strong results on code tasks with binary rewards, directly addressing the SWE-bench use case.

