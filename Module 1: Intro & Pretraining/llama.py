import math
import time
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm


# Define a dataclass for model arguments
@dataclass
class ModelArgs:
    dim: int = 4096  # Dimension of the model
    n_layers: int = 32  # Number of transformer layers
    n_heads: int = 32  # Number of attention heads
    n_kv_heads: Optional[int] = (
        None  # Number of key/value heads (if different from n_heads)
    )
    vocab_size: int = 32000  # Size of the vocabulary
    multiple_of: int = 256  # Ensures certain dimensions are multiples of this value
    ffn_dim_multiplier: Optional[float] = (
        None  # Multiplier for FFN intermediate dimension
    )
    norm_eps: float = 1e-5  # Epsilon for normalization
    max_batch_size: int = 32  # Maximum batch size
    max_seq_len: int = 2048  # Maximum sequence length

    def __post_init__(self):
        # Set default values and calculate intermediate size
        if self.n_kv_heads is None:
            self.n_kv_heads = self.n_heads
        if self.ffn_dim_multiplier is None:
            self.ffn_dim_multiplier = 4 / 3
        self.intermediate_size = int(2 * self.ffn_dim_multiplier * self.dim)
        self.intermediate_size = find_multiple(self.intermediate_size, self.multiple_of)


# Function to find the nearest multiple of k
def find_multiple(n: int, k: int) -> int:
    if n % k == 0:
        return n
    return n + k - (n % k)


# RMSNorm (Root Mean Square Layer Normalization)
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        raise NotImplementedError("RMSNorm not implemented yet")


# Precompute frequency tensor for rotary positional embedding
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


# Apply rotary positional embedding
def apply_rotary_emb(
    xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = freqs_cis[:, None, :]
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


# Attention mechanism
class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_kv_heads = args.n_kv_heads
        self.n_heads_q = args.n_heads
        self.n_rep = self.n_heads_q // self.n_kv_heads
        self.head_dim = args.dim // args.n_heads

        # Initialize weight matrices for Q, K, V, and output
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, args.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, args.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

    def forward(
        self, x: torch.Tensor, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]
    ):
        bsz, seqlen, _ = x.shape

        # Compute Q, K, V
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        # Reshape Q, K, V
        xq = xq.view(bsz, seqlen, self.n_heads_q, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_kv_heads, self.head_dim)

        # Apply rotary positional embedding
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        # Transpose for attention computation
        xq = xq.transpose(1, 2)  # (bs, n_heads_q, seqlen, head_dim)
        xk = xk.transpose(1, 2)  # (bs, n_kv_heads, seqlen, head_dim)
        xv = xv.transpose(1, 2)  # (bs, n_kv_heads, seqlen, head_dim)

        # Repeat K and V for multi-query attention
        xk = repeat_kv(xk, self.n_rep)  # (bs, n_heads_q, seqlen, head_dim)
        xv = repeat_kv(xv, self.n_rep)  # (bs, n_heads_q, seqlen, head_dim)

        # Compute attention scores
        scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask  # (bs, n_heads_q, seqlen, seqlen)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)

        # Compute attention output
        output = torch.matmul(scores, xv)  # (bs, n_heads_q, seqlen, head_dim)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)


# Feedforward network with SwiGLU
class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = find_multiple(hidden_dim, multiple_of)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        # SwiGLU activation function
        raise NotImplementedError("SwiGLU activation function not implemented yet")


# Transformer block
class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
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
        # Apply attention
        h = x + self.attention.forward(self.attention_norm(x), freqs_cis, mask)
        # Apply feedforward
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out


# Main Transformer model
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

        # Create causal mask
        mask = None
        if seqlen > 1:
            raise NotImplementedError("Causal mask not implemented yet")

        # Apply transformer layers
        for layer in self.layers:
            h = layer(h, freqs_cis, mask)
        h = self.norm(h)
        output = self.output(h)  # Return logits for all positions
        return output


# Helper function to repeat key/value heads
def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


# Create dataloader for training
def create_dataloader(batch_size, max_seq_len):
    # Load dataset
    dataset = load_dataset("wikitext", "wikitext-103-v1", split="train[:1000]")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # Set the pad token to be the eos token
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_seq_len,
            padding="max_length",
            return_overflowing_tokens=True,
            return_length=True,
        )

    # Tokenize the dataset
    tokenized_dataset = dataset.map(
        tokenize_function, batched=True, remove_columns=dataset.column_names
    )

    def collate_fn(examples):
        return torch.tensor(
            [example["input_ids"] for example in examples], dtype=torch.long
        )

    # Create and return the DataLoader
    return DataLoader(
        tokenized_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True
    )


# Training function
def train(model, dataloader, optimizer, scheduler, device, num_epochs, pad_token_id):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        total_tokens = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}")

        for batch in progress_bar:
            batch = batch.to(device)
            optimizer.zero_grad()

            # Prepare input and target
            input_ids = batch[:, :-1]
            target_ids = batch[:, 1:]

            # Forward pass
            logits = model(input_ids, start_pos=0)

            # Compute loss
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                target_ids.reshape(-1),
                ignore_index=pad_token_id,
                reduction="sum",
            )

            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            scheduler.step()

            # Update statistics
            total_loss += loss.item()
            total_tokens += (target_ids != pad_token_id).sum().item()

            # Calculate perplexity
            avg_loss = total_loss / total_tokens
            perplexity = math.exp(avg_loss)

            # Update progress bar
            progress_bar.set_postfix(
                {"loss": f"{avg_loss:.4f}", "ppl": f"{perplexity:.2f}"}
            )

        # Print epoch statistics
        avg_loss = total_loss / total_tokens
        perplexity = math.exp(avg_loss)
        print(
            f"Epoch {epoch + 1}, Average Loss: {avg_loss:.4f}, Perplexity: {perplexity:.2f}"
        )


def main():
    # Model configuration
    max_seq_len = 128
    model_args = ModelArgs(
        dim=64,
        n_layers=4,
        n_heads=4,
        n_kv_heads=4,
        vocab_size=50257,  # GPT-2 vocab size
        multiple_of=32,
        max_seq_len=max_seq_len,
        max_batch_size=32,
    )

    # Training configuration
    batch_size = 16
    num_epochs = 1
    learning_rate = 5e-5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model
    model = Transformer(model_args).to(device)

    # Create dataloader
    dataloader = create_dataloader(batch_size, max_seq_len)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    pad_token_id = tokenizer.pad_token_id

    # Initialize optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=len(dataloader) * num_epochs
    )

    # Train the model
    train(model, dataloader, optimizer, scheduler, device, num_epochs, pad_token_id)

    # Save the trained model
    torch.save(model.state_dict(), "llama_wikitext_trained.pth")


if __name__ == "__main__":
    main()
