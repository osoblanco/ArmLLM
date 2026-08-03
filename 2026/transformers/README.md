# LLaMA + Mixture-of-Experts Implementation Task & Competition

## Overview

You will implement a simplified LLaMA-style decoder-only language model — and then go one step further: replace the dense feed-forward block with a **sparse Mixture-of-Experts (MoE)** layer, the architecture behind models like Mixtral, DeepSeek-V3, and (reportedly) GPT-4.

This module is also a **competition**: everyone trains on the same fixed
dataset under the same constraints, reports perplexity on a fixed validation
benchmark, and submissions are ranked on a shared leaderboard. See
[Competition](#competition) below.

Files:

- `llama_moe_exercise.py` — the exercise. Every part you need to implement is
  marked with a `# TODO` block containing hints, expected tensor shapes, and
  paper references. The data pipeline, training loop, and generation code are
  already written.
- `llama_moe_solution.py` — the reference solution. Try not to peek until you
  have made a real attempt (or use it to check individual components).

## What you will implement

### Core LLaMA components

1. **RMSNorm** (Zhang & Sennrich, 2019) — a cheaper LayerNorm variant that
   only rescales by the root-mean-square of the features.
2. **Rotary Position Embeddings (RoPE)** (Su et al., 2021) — positions are
   encoded by rotating pairs of query/key dimensions with complex
   multiplication: `precompute_freqs_cis` and `apply_rotary_emb`.
3. **Grouped-Query Attention (GQA)** (Ainslie et al., 2023) — fewer key/value
   heads than query heads (`repeat_kv` plus the full attention forward pass).
4. **SwiGLU feed-forward** (Shazeer, 2020) — the gated activation used by
   LLaMA: `w2(silu(w1(x)) * w3(x))`.
5. **Causal mask** — the additive `-inf` upper-triangular mask that keeps a
   language model from peeking at future tokens.

### Mixture-of-Experts components

6. **Router** — a linear gate producing a probability distribution over
   experts for every token.
7. **Top-k routing + dispatch** (Mixtral-style; Jiang et al., 2024) — each
   token is processed by only its top-k experts, and their outputs are
   combined with the renormalized router weights. This is how MoE models get
   many more parameters at (almost) the same compute per token.
8. **Load-balancing auxiliary loss** (Switch Transformer; Fedus et al., 2021)
   — without it the router collapses onto one or two experts and the rest
   never train. You will implement `aux_loss = n_experts * sum_e(f_e * P_e)`
   and it gets added to the language-modeling loss with a small weight.

Setting `n_experts=1` in `ModelArgs` recovers the plain dense LLaMA — useful
for debugging the core components before touching the MoE parts.

## Requirements

Use the environment from the repository root [SETUP.md](../../SETUP.md). The
exercise only needs: `torch`, `transformers`, `datasets`, `tqdm`.

## Suggested workflow

1. **Sanity check first** (no dataset download needed):

   ```
   python llama_moe_exercise.py test
   ```

   It builds the model, runs a forward and backward pass on random tokens,
   and checks output shapes, the auxiliary loss value (~1.0 per MoE layer at
   initialization — can you see why?), and that gradients reach the router.

2. **Implement bottom-up**: RMSNorm → RoPE → `repeat_kv` → attention →
   SwiGLU → causal mask → MoE. Re-run `test` after each component; the
   `NotImplementedError`s will guide you to whatever is still missing.

3. **Train** on the fixed WikiText-2 train split:

   ```
   python llama_moe_exercise.py train
   ```

   Watch both numbers in the progress bar: `ce_loss` (language modeling)
   should fall steadily; `aux_loss` is the sum over all MoE layers, so with
   the default 6-layer top-2 config it should hover around 6.0 (~1.0 per
   layer) — if it climbs well above that, your router is collapsing. The
   reference configuration trains in a few minutes on the class H100.

4. **Evaluate** on the official benchmark (WikiText-2 validation split):

   ```
   python llama_moe_exercise.py eval
   ```

   This loads your checkpoint, re-checks the constraints, and prints
   `VALIDATION PERPLEXITY: xx.xx` — the number you submit.

5. **Generate** text with the trained model:

   ```
   python llama_moe_exercise.py generate
   ```

## Competition

### The benchmark

Your score is **perplexity on the WikiText-2 (raw) `validation` split**,
tokenized with the GPT-2 tokenizer and chunked into 128-token sequences —
exactly what `python <your_file>.py eval` computes. Lower is better.
Submissions are ranked against each other and against the reference-solution
baseline on the leaderboard.

### Rules (enforced in code where possible)

1. **Fixed tokenizer & vocab** — GPT-2, `vocab_size = 50257`. Asserted at
   train and eval time.
2. **Fixed data** — train ONLY on the WikiText-2 raw `train` split, at most
   **3 passes** (epochs). No external data, no pretrained weights, no
   distillation from other models. The validation split must never be used
   for training.
3. **Parameter budgets** — `active params per token ≤ 35M` and
   `total params ≤ 120M`, checked by `check_constraints()`. A dense model is
   capped at 35M total; MoE lets you pack up to 120M parameters into the same
   35M active budget — that asymmetry is the point of the exercise.
4. **Reproducibility** — the seed is fixed (`SEED = 42`, applied to torch,
   CUDA, python, numpy, and the dataloader shuffle). Your submitted result
   must reproduce by running `train` then `eval` on your submitted file.
5. **Frozen eval pipeline** — the competition constants, `evaluate()`,
   `load_and_preprocess_data()`, the `eval` mode block, and
   `count_params`/`check_constraints` must remain byte-identical to the
   handout. Everything else (architecture, optimizer, schedule, batch size,
   routing strategy, aux losses...) is yours to improve.
6. **Single GPU** — training must run on one GPU.

Violations found during audit (organizers re-run submitted files) mean
disqualification of the entry.

### How to submit

Open the leaderboard (URL announced in class / see `competition/README.md`),
fill in the form, and upload:

- your **team name**,
- the **perplexity** printed by `eval` mode,
- your **code file** (the single `.py` you ran).

You can resubmit as often as you like — your best (lowest) perplexity counts.

### Ideas to beat the baseline

Anything not forbidden by the rules is fair game: better learning-rate
schedules (cosine + warmup), weight tying between `tok_embeddings` and
`output`, more/fewer experts within the budgets, top-1 vs top-2 routing,
shared experts (DeepSeek-MoE style), better initialization, gradient
clipping, dropout, sequence-length curricula...

## Things to try afterwards

- Compare the MoE model against a dense one (`n_experts=1`) with a similar
  *active* parameter count. Which reaches lower perplexity per training step?
  Per wall-clock second?
- Set `aux_loss_weight=0.0` and log how many tokens each expert receives per
  batch. How quickly does the router collapse?
- Vary `n_experts_per_tok` (top-1 = Switch Transformer, top-2 = Mixtral).
- Implement smarter decoding: sampling with temperature, top-k, and top-p
  (nucleus) sampling — see the 2025 module for the task description.
- Add a KV cache so generation does not recompute the whole prefix for every
  new token (that is what `start_pos` is hinting at).

## References

- LLaMA: [Touvron et al., 2023](https://arxiv.org/abs/2302.13971)
- RoPE: [Su et al., 2021](https://arxiv.org/abs/2104.09864)
- GQA: [Ainslie et al., 2023](https://arxiv.org/abs/2305.13245)
- SwiGLU: [Shazeer, 2020](https://arxiv.org/abs/2002.05202)
- Sparse MoE: [Shazeer et al., 2017](https://arxiv.org/abs/1701.06538)
- Switch Transformer: [Fedus et al., 2021](https://arxiv.org/abs/2101.03961)
- Mixtral of Experts: [Jiang et al., 2024](https://arxiv.org/abs/2401.04088)

Good luck with your implementation!
