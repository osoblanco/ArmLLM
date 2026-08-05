# Test-Time Compute Assignment

You are given a deliberately simple test-time compute pipeline for a fixed Qwen3-1.7B model and a validation set containing 30 math problems. Each problem has a point value, and a correct solution earns that full number of points while an incorrect solution earns zero. Your objective is to improve the pipeline and maximize the total points it earns, then submit your best version for evaluation on a hidden test set. The hidden problems will have a similar distribution of difficulty and point values, but their point values will not be available to your pipeline. Point values are included in the validation set only to help you analyze and debug your approach; during evaluation, `solve()` receives only the problem text and model endpoint. The problem-solving phase has a 15-minute wall-clock budget that begins after the model endpoint is ready, and the hidden test will use the same budget.

## Objective and scoring

For every problem, your pipeline must return one integer answer.

A correct answer earns the problem's full point value. An incorrect answer, failure, timeout, or unattempted problem earns zero.

The evaluator reports:

- total points earned;
- total points available;
- normalized weighted score;
- raw accuracy;
- number of attempted problems.

The hidden-set score determines the final result. Maximizing raw accuracy is useful, but the actual objective is to maximize earned points.

## What you can improve

The supplied `solve.py` is only a baseline and leaves substantial room for improvement. You may change how it uses the fixed model, including:

- prompting and answer-extraction strategies;
- thinking or non-thinking mode;
- sampling parameters;
- generation length and stopping logic;
- repeated model calls;
- self-consistency or voting;
- candidate verification;
- adaptive test-time compute;
- tool use;
- any other inference-time strategy that respects the rules below.

The evaluator calls `solve(problem, endpoint)` once for each problem. Your implementation may make multiple model requests internally before returning its final answer.

## Files you may change

- `solve.py` — the main submission. The evaluator imports `solve()` from this file.
- `model_config.py` — vLLM startup settings such as context length, data type, and GPU-memory utilization.
- Additional helper files imported by `solve.py`.

If your solver depends on helper files, include them with your submission.

## Files you must not change

- `evaluator.py`
- `requirements.txt`
- Validation or test-set files

The model is fixed to Qwen3-1.7B. Your pipeline must use the model served at the supplied endpoint and must not select, load, or call a different model.

Your pipeline must not read validation or hidden test-set files at evaluation time, look up known answers, or access problem metadata such as `answer` or `points`. It should operate only on the `problem` and `endpoint` arguments passed to `solve()`.

## Required interface

Implement the following function in `solve.py`:

```python
def solve(problem: str, endpoint: str) -> str:
    ...
```

Return only the predicted integer as a string:

```python
"40"
```

Scoring uses exact Python equality:

```python
prediction == ground_truth
```

There is no answer normalization. Extra text, malformed output, or a numerically equivalent answer in another format will be marked incorrect.

## Baseline solver

The supplied baseline uses Qwen3's native thinking mode and streams the response from vLLM. Reasoning tokens are exposed separately as `reasoning_content`, while the final response is exposed as `content`.

It starts with Qwen3's recommended thinking-mode sampling parameters:

- `temperature=0.6`
- `top_p=0.95`
- `top_k=20`
- `min_p=0`

The baseline allows up to 32,768 generated tokens within a 40,960-token context. It asks the model to place its answer in `\boxed{}` and stops when it detects a boxed integer in the final-response content.

This is an example, not a required strategy. You may replace its prompt, sampling configuration, stopping policy, or overall inference procedure.

## Optional Python tool

`solve_python.py` demonstrates native Qwen3/OpenAI function calling with a sandboxed Python tool. The evaluator does not import this file automatically. To use it, adapt or copy its implementation into `solve.py`.

Model-written Python runs in a locked-down Docker container with:

- no network access;
- no host mounts;
- a read-only filesystem;
- no Linux capabilities;
- an unprivileged user;
- CPU, memory, process, output, and execution-time limits.

Tool results are returned to the model as native `tool` messages.

Prepare the sandbox image before evaluation:

```bash
sudo docker pull python:3.12-alpine
```

The evaluation uses `--pull=never`, so the tool cannot download or replace the image while running.

## Install and run

Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

Run the evaluator on the provided validation set:

```bash
python evaluator.py validation.jsonl \
  --output validation-results.jsonl
```

The evaluator reuses cached Qwen3-1.7B weights when available and otherwise downloads them once. It starts vLLM using `model_config.py`, shuffles the problems, evaluates them sequentially, saves each result immediately, and shuts the server down afterward.

The 15-minute wall-clock limit applies to the complete problem-solving phase, not separately to each problem. The clock begins after the evaluator has started vLLM and confirmed that the model endpoint is ready, so model download and server startup are excluded. From that point onward, every model call made by your pipeline and all other processing must fit within the total budget. The hidden test uses the same 15-minute limit. If the deadline is reached, the active call is interrupted and remaining problems receive no points.

## Submission

Submit:

- `solve.py`;
- `model_config.py`;
- every helper file imported by `solve.py`.

Do not submit a modified evaluator, requirements file, validation set, or test set.
