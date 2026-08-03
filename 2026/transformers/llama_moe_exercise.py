"""LLaMA-style decoder with a Mixture-of-Experts feed-forward — EXERCISE.

Your task is to implement a small LLaMA-style decoder-only transformer
(RMSNorm, RoPE, grouped-query attention, SwiGLU) where the dense feed-forward
block can be swapped for a sparse Mixture-of-Experts (MoE) layer in the style
of Mixtral:

  - a learned router assigns every token to its top-k experts,
  - each expert is an ordinary SwiGLU feed-forward network,
  - expert outputs are combined weighted by the (renormalized) router scores,
  - a load-balancing auxiliary loss keeps the router from collapsing onto
    a few experts.

Every part you need to implement is marked with a `# TODO` block containing
hints, expected tensor shapes, and paper references. The data pipeline,
training loop, and generation code are already written for you.

=== COMPETITION RULES (see README.md for details) ===
Fixed and enforced by this file — do NOT modify the eval pipeline:
  - Tokenizer: GPT-2, vocab_size = 50257 (VOCAB_SIZE)
  - Train data: WikiText-2 raw `train` split ONLY, max MAX_EPOCHS passes
  - Benchmark: perplexity on the WikiText-2 raw `validation` split,
    chunked at EVAL_MAX_LENGTH tokens, computed by `eval` mode
  - Budgets: active params/token <= MAX_ACTIVE_PARAMS,
             total params <= MAX_TOTAL_PARAMS
  - From scratch: no pretrained weights, no external data
  - Reproducibility: seed fixed to SEED; run `train` then `eval`

Run modes:
  python llama_moe_exercise.py test      # quick forward/backward sanity check
  python llama_moe_exercise.py train     # train on WikiText-2 (fixed dataset)
  python llama_moe_exercise.py eval      # official benchmark perplexity
  python llama_moe_exercise.py generate  # greedy decoding from a prompt

Start with `test` — it needs no dataset download and will tell you
immediately whether your shapes and gradients are right.
"""

import math
import os
import random
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
from dataclasses import dataclass, asdict
from typing import Optional

# =====================  COMPETITION CONSTANTS (do not edit)  ================

SEED = 42
TOKENIZER_NAME = "gpt2"
VOCAB_SIZE = 50257  # GPT-2 vocabulary — fixed for every submission
TRAIN_DATASET = ("Salesforce/wikitext", "wikitext-2-raw-v1", "train")
EVAL_DATASET = ("Salesforce/wikitext", "wikitext-2-raw-v1", "validation")
EVAL_MAX_LENGTH = 128  # sequence length used for chunking train and eval data
MAX_EPOCHS = 3  # maximum passes over the training split
MAX_ACTIVE_PARAMS = 35_000_000  # params used per token (dense: all of them)
MAX_TOTAL_PARAMS = 120_000_000  # MoE lets you exceed active, not total
CHECKPOINT_PATH = "llama_moe_wikitext_trained.pth"

# ============================================================================


def set_seed(seed: int = SEED):
    """Make training and evaluation reproducible."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    # Deterministic cuDNN kernels (slightly slower, but reproducible)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Model Architecture Components


@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = VOCAB_SIZE
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    max_batch_size: int = 32
    max_seq_len: int = 2048
    # Mixture-of-Experts settings. n_experts = 1 gives a plain dense model.
    n_experts: int = 1
    n_experts_per_tok: int = 2
    aux_loss_weight: float = 0.01

    def __post_init__(self):
        if self.n_kv_heads is None:
            self.n_kv_heads = self.n_heads
        if self.ffn_dim_multiplier is None:
            self.ffn_dim_multiplier = 4 / 3
        self.intermediate_size = int(2 * self.ffn_dim_multiplier * self.dim)
        self.intermediate_size = find_multiple(self.intermediate_size, self.multiple_of)


def find_multiple(n: int, k: int) -> int:
    if n % k == 0:
        return n
    return n + k - (n % k)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (Zhang & Sennrich, 2019).

    Unlike LayerNorm, RMSNorm does not subtract the mean — it only rescales
    by the root-mean-square of the features, which is cheaper and works just
    as well in practice. LLaMA uses it before every attention and FFN block.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        # TODO: Implement RMS normalization
        # Hint: RMS = sqrt(mean(x^2)) over the last dimension
        # Return: x * rsqrt(mean(x^2, dim=-1, keepdim=True) + eps)
        # (torch.rsqrt is 1/sqrt — keepdim=True keeps shapes broadcastable)
        raise NotImplementedError

    def forward(self, x):
        # TODO: Apply normalization and scale by the weight parameter
        # 1. Normalize x using self._norm (convert x to float32 first for
        #    numerical stability: x.float())
        # 2. Convert the result back to the original dtype (.type_as(x))
        # 3. Multiply by self.weight (learned per-feature scale)
        raise NotImplementedError


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    # TODO: Implement RoPE frequency computation (Su et al., 2021 — RoFormer)
    #
    # RoPE encodes positions by *rotating* pairs of feature dimensions by a
    # position-dependent angle. Here we precompute the rotation for every
    # (position, feature-pair) as a unit complex number e^(i * angle).
    #
    # 1. Frequency vector: 1 / (theta^(2i/dim)) for i in [0, dim//2)
    #    Hint: torch.arange(0, dim, 2)[: (dim // 2)].float() / dim
    # 2. Position vector: t = [0, 1, ..., end-1]
    # 3. Outer product of positions and frequencies -> angles [end, dim//2]
    #    Hint: torch.outer
    # 4. Convert to complex numbers with torch.polar(magnitude, angle)
    #    where magnitude = torch.ones_like(angles)
    # Return shape: [end, dim//2], dtype complex64
    raise NotImplementedError


def apply_rotary_emb(
    xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    # TODO: Apply rotary position embeddings
    # xq shape: [batch, seq_len, n_heads, head_dim]
    # xk shape: [batch, seq_len, n_kv_heads, head_dim]
    # freqs_cis shape: [seq_len, head_dim//2]
    #
    # Multiplying two complex numbers adds their angles — so multiplying by
    # e^(i*angle) rotates. That is the whole trick:
    #
    # 1. View xq and xk as complex numbers by pairing consecutive dims:
    #    torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    #    -> shape [batch, seq_len, n_heads, head_dim//2] complex
    # 2. Reshape freqs_cis to broadcast over the head dimension:
    #    freqs_cis[:, None, :] -> [seq_len, 1, head_dim//2]
    # 3. Multiply (this rotates every pair by its position's angle)
    # 4. Convert back with torch.view_as_real(...).flatten(3) and restore
    #    the original dtype with .type_as(xq) / .type_as(xk)
    # Return: (xq_out, xk_out) with the same shapes as the inputs
    raise NotImplementedError


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    # TODO: Implement key/value head repetition for grouped-query attention
    # (Ainslie et al., 2023 — GQA). With fewer kv heads than query heads,
    # each kv head must be shared by n_rep = n_heads // n_kv_heads query
    # heads. The simplest way is to repeat each kv head n_rep times.
    #
    # x shape: [batch, seq_len, n_kv_heads, head_dim]
    # If n_rep == 1, return x unchanged.
    # Otherwise, repeat each kv head n_rep times and merge the repeats into
    # the head dimension. Hint: insert a new axis after the head dim, expand
    # it to n_rep (expand avoids copying, unlike repeat), then reshape so
    # the head dimension becomes n_kv_heads * n_rep. Careful: the copies of
    # one head must stay adjacent (head order [0,0,1,1], not [0,1,0,1]).
    # Return shape: [batch, seq_len, n_kv_heads * n_rep, head_dim]
    raise NotImplementedError


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_kv_heads = args.n_kv_heads
        self.n_heads_q = args.n_heads
        self.n_rep = self.n_heads_q // self.n_kv_heads
        self.head_dim = args.dim // args.n_heads

        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, args.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, args.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

    def forward(
        self, x: torch.Tensor, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]
    ):
        # TODO: Implement grouped-query multi-head attention with RoPE
        # x shape: [batch_size, seq_len, dim]
        # freqs_cis shape: [seq_len, head_dim//2]
        # mask shape: [1, 1, seq_len, seq_len] or None

        bsz, seqlen, _ = x.shape

        # 1. Project x to Q, K, V using self.wq, self.wk, self.wv
        # 2. Reshape to separate the heads:
        #    Q -> [batch, seq_len, n_heads_q, head_dim]
        #    K, V -> [batch, seq_len, n_kv_heads, head_dim]
        # 3. Apply rotary embeddings to Q and K with apply_rotary_emb
        # 4. Repeat K and V with repeat_kv so they match n_heads_q
        #    (repeat_kv expects heads on dim 2, so do this BEFORE transposing)
        # 5. Transpose all three to [batch, n_heads, seq_len, head_dim]
        #    (.transpose(1, 2))
        # 6. Attention scores: Q @ K^T / sqrt(head_dim)
        #    -> [batch, n_heads, seq_len, seq_len]
        # 7. If mask is not None, add it to the scores (it contains -inf above
        #    the diagonal, which softmax turns into zero attention)
        # 8. Softmax over the last dimension (compute in float32 for
        #    stability: F.softmax(scores.float(), dim=-1).type_as(xq))
        # 9. Weighted sum of values: softmax(scores) @ V
        #    -> [batch, n_heads, seq_len, head_dim]
        # 10. Transpose back to [batch, seq_len, n_heads, head_dim], then
        #     .contiguous().view(bsz, seqlen, -1) to merge the heads
        # 11. Apply the output projection self.wo

        raise NotImplementedError


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = find_multiple(hidden_dim, multiple_of)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        # TODO: Implement the SwiGLU feed-forward (Shazeer, 2020)
        # SwiGLU(x) = (Swish(x @ W1) * (x @ W3)) @ W2
        # where Swish(x) = x * sigmoid(x) = F.silu(x)
        # The silu(w1(x)) branch acts as a learned *gate* on the w3(x) branch.
        # Return: self.w2(F.silu(self.w1(x)) * self.w3(x))
        raise NotImplementedError


class MoEFeedForward(nn.Module):
    """Sparse Mixture-of-Experts feed-forward layer (Mixtral-style).

    Instead of one big FFN that every token passes through, we keep
    `n_experts` separate FFNs ("experts") and a small learned router. Every
    token is processed by only its top-k experts, so the model gets many
    more parameters at (almost) the same compute per token.

    References: Shazeer et al., 2017 (sparse MoE); Fedus et al., 2021
    (Switch Transformer, the load-balancing loss); Jiang et al., 2024
    (Mixtral of Experts, the exact routing used here).
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_experts = args.n_experts
        self.n_experts_per_tok = args.n_experts_per_tok
        # The router ("gate"): one logit per expert for every token
        self.gate = nn.Linear(args.dim, args.n_experts, bias=False)
        # Each expert is an ordinary SwiGLU feed-forward network
        self.experts = nn.ModuleList(
            FeedForward(args.dim, args.intermediate_size, args.multiple_of)
            for _ in range(args.n_experts)
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, seqlen, dim = x.shape
        # Routing is per-token, so flatten batch and sequence dimensions
        x = x.view(-1, dim)  # [n_tokens, dim] where n_tokens = bsz * seqlen

        # TODO — Step 1: Route
        # Compute router logits with self.gate, then softmax over the expert
        # dimension to get a probability distribution per token.
        # router_logits: [n_tokens, n_experts]
        # router_probs:  [n_tokens, n_experts]

        # TODO — Step 2: Select top-k experts per token
        # Use torch.topk on router_probs (k = self.n_experts_per_tok) to get
        #   topk_weight: [n_tokens, k]  (the probabilities of the chosen experts)
        #   topk_idx:    [n_tokens, k]  (which experts were chosen)
        # Then renormalize topk_weight so each token's k weights sum to 1
        # (they were part of a softmax over ALL experts, so right now they
        # sum to less than 1).

        # TODO — Step 3: Dispatch tokens to experts and combine
        # Start from out = torch.zeros_like(x) and loop over the experts.
        # For expert e you need to know which tokens picked it AND in which
        # top-k slot (1st or 2nd choice — you need the slot to look up the
        # right weight). Hint: torch.where(topk_idx == e) gives you both as
        # a (token_idx, slot_idx) pair. Skip the expert if nothing routed
        # to it. Otherwise run the expert ONLY on the selected rows of x,
        # scale each output row by that token's routing weight (index
        # topk_weight with the token/slot pair; you'll need a trailing axis
        # so it broadcasts over the feature dim), and add the result into
        # the selected rows of out.
        # This is the "sparse" part: each expert only sees its own tokens.

        # TODO — Step 4: Load-balancing auxiliary loss
        # With no pressure to spread the load, the router quickly collapses
        # onto one or two experts ("expert collapse") and the rest never
        # train. The fix (Switch Transformer, eq. 4-6):
        #   f_e = fraction of routed (token, slot) pairs assigned to expert e
        #   P_e = mean router probability of expert e over all tokens
        #   aux_loss = n_experts * sum_e(f_e * P_e)
        # It equals 1.0 exactly when routing is perfectly uniform and grows
        # when the router favors a few experts. Gradients flow through P_e
        # (f_e comes from a hard top-k, which has no gradient).
        # Hint: F.one_hot(topk_idx, self.n_experts) marks which expert every
        # (token, slot) pair picked — average it over the right dimensions
        # to get f_e; average router_probs over tokens to get P_e.

        # Finally: return (out.view(bsz, seqlen, dim), aux_loss)
        raise NotImplementedError


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        if args.n_experts > 1:
            self.feed_forward = MoEFeedForward(args)
        else:
            self.feed_forward = FeedForward(
                dim=args.dim,
                hidden_dim=args.intermediate_size,
                multiple_of=args.multiple_of,
            )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(
        self, x: torch.Tensor, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]
    ):
        # Pre-norm architecture with residual connections (given — study it!)
        h = x + self.attention.forward(self.attention_norm(x), freqs_cis, mask)
        ffn_in = self.ffn_norm(h)
        if isinstance(self.feed_forward, MoEFeedForward):
            ffn_out, aux_loss = self.feed_forward(ffn_in)
        else:
            ffn_out = self.feed_forward(ffn_in)
            aux_loss = x.new_zeros(())
        return h + ffn_out, aux_loss


class Transformer(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)
        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))
        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = nn.Linear(params.dim, params.vocab_size, bias=False)

        self.freqs_cis = precompute_freqs_cis(
            self.params.dim // self.params.n_heads, self.params.max_seq_len * 2
        )

    def forward(self, tokens: torch.Tensor, start_pos: int):
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

        # TODO: Implement the causal mask for sequences longer than 1
        # A language model must not peek at future tokens: position i may
        # only attend to positions <= i. We build an additive mask that is
        # -inf above the diagonal (softmax turns -inf into 0 attention).
        # 1. Create a [1, 1, seqlen, seqlen] tensor filled with float("-inf")
        #    on tokens.device (torch.full)
        # 2. Keep only the upper triangle with torch.triu(mask, diagonal=1)
        #    — everything on/below the diagonal becomes 0. (Queries and keys
        #    come from the same chunk here — with a KV cache the offset
        #    would depend on start_pos, but we don't have one.)
        # 3. Match dtypes with .type_as(h)
        mask = None
        if seqlen > 1:
            # Create the causal mask here
            raise NotImplementedError

        # Sum the auxiliary losses over all MoE layers (given)
        aux_loss_total = h.new_zeros(())
        for layer in self.layers:
            h, aux_loss = layer(h, freqs_cis, mask)
            aux_loss_total = aux_loss_total + aux_loss
        h = self.norm(h)
        output = self.output(h)
        return output, aux_loss_total


# Competition helpers: parameter budgets and constraint checks


def count_params(model: Transformer) -> tuple[int, int]:
    """Return (total_params, active_params_per_token).

    Active params = parameters that participate in processing one token.
    For a dense model this is every parameter; for MoE, each token only
    passes through n_experts_per_tok of the n_experts expert FFNs, so the
    remaining experts do not count against the active budget.
    """
    total = sum(p.numel() for p in model.parameters())
    inactive = 0
    for module in model.modules():
        if isinstance(module, MoEFeedForward):
            expert_params = sum(p.numel() for p in module.experts[0].parameters())
            inactive += (module.n_experts - module.n_experts_per_tok) * expert_params
    return total, total - inactive


def check_constraints(model: Transformer, model_args: ModelArgs) -> tuple[int, int]:
    """Enforce the competition constraints. Raises on violation."""
    assert model_args.vocab_size == VOCAB_SIZE, (
        f"vocab_size must be {VOCAB_SIZE} (GPT-2), got {model_args.vocab_size}"
    )
    total, active = count_params(model)
    assert active <= MAX_ACTIVE_PARAMS, (
        f"active params {active:,} exceed the budget {MAX_ACTIVE_PARAMS:,}"
    )
    assert total <= MAX_TOTAL_PARAMS, (
        f"total params {total:,} exceed the budget {MAX_TOTAL_PARAMS:,}"
    )
    return total, active


# Dataset and Data Loading


class WikiTextDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings)

    def __getitem__(self, idx):
        return {key: torch.tensor(val) for key, val in self.encodings[idx].items()}


def load_and_preprocess_data(tokenizer, dataset_spec, max_length=EVAL_MAX_LENGTH):
    """Tokenize a fixed dataset split into contiguous max_length chunks.

    The same function prepares both the training split and the evaluation
    split, so the benchmark chunking is identical for every submission.
    """
    path, name, split = dataset_spec
    dataset = load_dataset(path, name, split=split)

    def tokenize_function(examples):
        # Join all texts with EOS token between documents
        text = tokenizer.eos_token.join(examples["text"])

        # Tokenize the entire text
        tokenized = tokenizer(text, return_tensors="pt", truncation=False)
        input_ids = tokenized["input_ids"][0]

        # Create chunks of max_length
        total_length = len(input_ids)
        total_length = (total_length // max_length) * max_length

        # Reshape into chunks
        input_ids = input_ids[:total_length].view(-1, max_length)

        # Create labels (shifted input_ids)
        labels = input_ids.clone()
        labels[:, :-1] = input_ids[:, 1:]
        labels[:, -1] = -100  # Ignore the last token in each sequence for loss

        return {"input_ids": input_ids.tolist(), "labels": labels.tolist()}

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        batch_size=len(dataset),  # single batch => one contiguous token stream
        remove_columns=dataset.column_names,
    )

    return WikiTextDataset(tokenized_dataset)


# Training Functions


def train_epoch(model, dataloader, optimizer, criterion, aux_loss_weight, device):
    model.train()
    total_loss = 0
    progress_bar = tqdm(dataloader, desc="Training")

    for batch in progress_bar:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        logits, aux_loss = model(input_ids, start_pos=0)

        # Language-modeling loss + weighted load-balancing loss
        ce_loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
        loss = ce_loss + aux_loss_weight * aux_loss

        loss.backward()
        optimizer.step()

        total_loss += ce_loss.item()
        progress_bar.set_postfix(
            {"ce_loss": f"{ce_loss.item():.4f}", "aux_loss": f"{aux_loss.item():.4f}"}
        )

    return total_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device):
    """Official benchmark metric — do not modify."""
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            logits, _ = model(input_ids, start_pos=0)
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    perplexity = math.exp(avg_loss)
    return avg_loss, perplexity


def generate_text_greedy(model, tokenizer, prompt, device, max_length=50, temperature=1.0):
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        for _ in range(max_length):
            logits, _ = model(input_ids, start_pos=0)
            next_token_logits = logits[:, -1, :] / temperature
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
            input_ids = torch.cat([input_ids, next_token], dim=-1)

            if next_token.item() == tokenizer.eos_token_id:
                break

    generated_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    return generated_text


def save_checkpoint(model, model_args, path=CHECKPOINT_PATH):
    """Save weights together with the config and seed for reproducibility."""
    torch.save(
        {
            "model_args": asdict(model_args),
            "state_dict": model.state_dict(),
            "seed": SEED,
            "torch_version": str(torch.__version__),
        },
        path,
    )


def load_checkpoint(path=CHECKPOINT_PATH, device="cpu"):
    """Rebuild the model exactly as it was trained."""
    checkpoint = torch.load(path, map_location=device)
    saved_args = {
        k: v
        for k, v in checkpoint["model_args"].items()
        if k in ModelArgs.__dataclass_fields__
    }
    model_args = ModelArgs(**saved_args)
    model = Transformer(model_args).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    return model, model_args


def sanity_check(model_args, device):
    """Quick forward/backward test on random tokens — no dataset needed."""
    print("Running sanity check...")
    set_seed(SEED)
    model = Transformer(model_args).to(device)

    total_params, active_params = count_params(model)
    print(f"Total parameters: {total_params:,} (active per token: {active_params:,})")
    print(f"Budgets: active <= {MAX_ACTIVE_PARAMS:,}, total <= {MAX_TOTAL_PARAMS:,}")
    check_constraints(model, model_args)
    print("Constraint check OK")

    tokens = torch.randint(0, model_args.vocab_size, (2, 16), device=device)
    logits, aux_loss = model(tokens, start_pos=0)

    assert logits.shape == (2, 16, model_args.vocab_size), f"bad logits shape {logits.shape}"
    print(f"Logits shape OK: {tuple(logits.shape)}")
    print(f"Aux loss: {aux_loss.item():.4f} (should be ~1.0 per MoE layer at init)")

    # Check that gradients reach the router
    loss = logits.float().mean() + model_args.aux_loss_weight * aux_loss
    loss.backward()
    if model_args.n_experts > 1:
        gate_grad = model.layers[0].feed_forward.gate.weight.grad
        assert gate_grad is not None and gate_grad.abs().sum() > 0, "no gradient on router"
        print("Router gradient OK")
    print("Sanity check passed!")


def main(mode="test"):
    # Model configuration: a small MoE model within the competition budgets.
    # Set n_experts=1 to recover the plain dense LLaMA.
    # You may change the architecture and hyperparameters — the constraints
    # (vocab, data, epochs, parameter budgets) are checked automatically.
    model_args = ModelArgs(
        dim=256,
        n_layers=6,
        n_heads=8,
        n_kv_heads=4,  # grouped-query attention: 2 query heads per kv head
        vocab_size=VOCAB_SIZE,
        multiple_of=32,
        max_seq_len=EVAL_MAX_LENGTH,
        max_batch_size=32,
        n_experts=4,
        n_experts_per_tok=2,
        aux_loss_weight=0.01,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} | torch {torch.__version__} | seed {SEED}")

    if mode == "test":
        sanity_check(model_args, device)
        return

    # Initialize tokenizer (fixed for the competition)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    if mode == "train":
        set_seed(SEED)

        # Training hyperparameters (free to tune)
        batch_size = 32
        learning_rate = 3e-4
        num_epochs = 3  # must stay <= MAX_EPOCHS
        assert num_epochs <= MAX_EPOCHS, f"at most {MAX_EPOCHS} passes over the data"

        print("Loading and preprocessing dataset...")
        train_dataset = load_and_preprocess_data(tokenizer, TRAIN_DATASET)
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(SEED),  # reproducible order
        )
        print(f"Training chunks: {len(train_dataset)} x {EVAL_MAX_LENGTH} tokens")

        print("Initializing model...")
        model = Transformer(model_args).to(device)
        total_params, active_params = check_constraints(model, model_args)
        print(f"Total parameters: {total_params:,} (active: {active_params:,})")

        optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss(ignore_index=-100)

        print("Starting training...")
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            train_loss = train_epoch(
                model,
                train_dataloader,
                optimizer,
                criterion,
                model_args.aux_loss_weight,
                device,
            )
            print(f"Training Loss: {train_loss:.4f}")
            print(f"Training Perplexity: {math.exp(train_loss):.4f}")

        print("\nSaving model...")
        save_checkpoint(model, model_args)
        print(f"Model saved as '{CHECKPOINT_PATH}'")
        print("Now run: python", sys.argv[0], "eval")

    elif mode == "eval":
        # ===== Official benchmark — this block must remain unmodified =====
        set_seed(SEED)

        print("Loading trained model...")
        model, saved_args = load_checkpoint(CHECKPOINT_PATH, device)
        total_params, active_params = check_constraints(model, saved_args)
        print(f"Total parameters: {total_params:,} (active: {active_params:,})")

        print("Loading evaluation dataset...")
        eval_dataset = load_and_preprocess_data(tokenizer, EVAL_DATASET)
        eval_dataloader = DataLoader(eval_dataset, batch_size=32, shuffle=False)
        print(f"Evaluation chunks: {len(eval_dataset)} x {EVAL_MAX_LENGTH} tokens")

        criterion = nn.CrossEntropyLoss(ignore_index=-100)
        avg_loss, perplexity = evaluate(model, eval_dataloader, criterion, device)

        print(f"\nValidation loss: {avg_loss:.4f}")
        print(f"VALIDATION PERPLEXITY: {perplexity:.2f}")
        print("(submit this number, your code file, and your checkpoint)")

    elif mode == "generate":
        model, _ = load_checkpoint(CHECKPOINT_PATH, device)

        prompt = "In a world where"
        print(f"Prompt: {prompt}\n")

        print("=== Greedy Decoding ===")
        generated_text = generate_text_greedy(
            model, tokenizer, prompt, device, max_length=50
        )
        print(f"Generated text: {generated_text}\n")


if __name__ == "__main__":
    # Run with: python llama_moe_exercise.py test|train|eval|generate
    mode = sys.argv[1] if len(sys.argv) > 1 else "test"
    main(mode)
