import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
from dataclasses import dataclass
from typing import Optional

# Model Architecture Components


@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = 32000
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    max_batch_size: int = 32
    max_seq_len: int = 2048

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
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        # TODO: Implement RMS normalization
        # Hint: RMS = sqrt(mean(x^2))
        # Return: x * rsqrt(mean(x^2) + eps)
        raise NotImplementedError

    def forward(self, x):
        # TODO: Apply normalization and scale by weight parameter
        # 1. Normalize x using _norm (convert to float32 for stability)
        # 2. Convert back to original dtype
        # 3. Multiply by self.weight
        raise NotImplementedError


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    # TODO: Implement RoPE frequency computation
    # 1. Create frequency vector: 1 / (theta^(2i/dim)) for i in [0, dim//2)
    # 2. Create position vector: [0, 1, ..., end-1]
    # 3. Compute outer product of position and frequency vectors
    # 4. Convert to complex numbers using torch.polar (magnitude=1, angle=freqs)
    # Return shape: [end, dim//2]
    raise NotImplementedError


def apply_rotary_emb(
    xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    # TODO: Apply rotary position embeddings
    # xq shape: [batch, seq_len, n_heads, head_dim]
    # xk shape: [batch, seq_len, n_kv_heads, head_dim]
    # freqs_cis shape: [seq_len, head_dim//2]

    # 1. Reshape xq and xk to complex numbers (pair consecutive dimensions)
    # 2. Add batch and head dimensions to freqs_cis
    # 3. Multiply by complex exponentials
    # 4. Convert back to real numbers and flatten
    # Return: (xq_out, xk_out) with same shapes as inputs
    raise NotImplementedError


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    # TODO: Implement key-value repetition for grouped query attention
    # x shape: [batch, seq_len, n_kv_heads, head_dim]
    # If n_rep == 1, return x unchanged
    # Otherwise, repeat each kv_head n_rep times
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
        # TODO: Implement multi-head attention with RoPE
        # x shape: [batch_size, seq_len, dim]
        # freqs_cis shape: [seq_len, head_dim//2]
        # mask shape: [1, 1, seq_len, seq_len] or None

        bsz, seqlen, _ = x.shape

        # 1. Project x to Q, K, V using self.wq, self.wk, self.wv
        # 2. Reshape to separate heads: [batch, seq_len, n_heads, head_dim]
        # 3. Apply rotary embeddings to Q and K using apply_rotary_emb
        # 4. Transpose to [batch, n_heads, seq_len, head_dim]
        # 5. Repeat K and V for grouped query attention using repeat_kv
        # 6. Compute attention scores: Q @ K^T / sqrt(head_dim)
        # 7. Apply mask if provided
        # 8. Apply softmax
        # 9. Apply attention to values: softmax(scores) @ V
        # 10. Transpose back and reshape
        # 11. Apply output projection self.wo

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
        # TODO: Implement SwiGLU activation
        # SwiGLU(x) = (Swish(xW1) * xW3)W2
        # where Swish(x) = x * sigmoid(x) ≈ silu(x)
        # Return: self.w2(silu(self.w1(x)) * self.w3(x))
        raise NotImplementedError


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
        # Pre-norm architecture with residual connections
        h = x + self.attention.forward(self.attention_norm(x), freqs_cis, mask)
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out


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

        # TODO: Implement causal mask for sequences longer than 1
        # Create an upper triangular mask filled with -inf
        # Mask shape: [1, 1, seqlen, seqlen]
        # Use torch.triu with diagonal offset of start_pos + 1
        mask = None
        if seqlen > 1:
            # Create causal mask here
            raise NotImplementedError

        for layer in self.layers:
            h = layer(h, freqs_cis, mask)
        h = self.norm(h)
        output = self.output(h)
        return output


# Dataset and Data Loading


class WikiTextDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}


def load_and_preprocess_data(tokenizer, max_length=128):
    # Load WikiText dataset
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:10%]")

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
        remove_columns=dataset.column_names,
    )

    return WikiTextDataset(tokenized_dataset)


# Training Functions


def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    progress_bar = tqdm(dataloader, desc="Training")

    for batch in progress_bar:
        # Move data to device
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(input_ids, start_pos=0)

        # Reshape for loss calculation
        # outputs: [batch_size, seq_len, vocab_size]
        # labels: [batch_size, seq_len]
        loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))

        # Backward pass
        loss.backward()

        # Update weights
        optimizer.step()

        # Update progress bar
        total_loss += loss.item()
        progress_bar.set_postfix({"loss": loss.item()})

    return total_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in dataloader:
            # Move data to device
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            # Forward pass
            outputs = model(input_ids, start_pos=0)

            # Calculate loss
            loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))

            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    perplexity = math.exp(avg_loss)
    return avg_loss, perplexity


def generate_text_greedy(model, tokenizer, prompt, max_length=50, temperature=1.0):
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors="pt").cuda()

    with torch.no_grad():
        for _ in range(max_length):
            outputs = model(input_ids, start_pos=0)
            next_token_logits = outputs[:, -1, :] / temperature
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
            input_ids = torch.cat([input_ids, next_token], dim=-1)

            if next_token.item() == tokenizer.eos_token_id:
                break

    generated_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    return generated_text


def main(mode="train"):
    # Model configuration
    model_args = ModelArgs(
        dim=128,
        n_layers=64,
        n_heads=64,
        n_kv_heads=64,
        vocab_size=50257,  # GPT-2 vocab size
        multiple_of=32,
        max_seq_len=128,
        max_batch_size=32,
    )

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    if mode == "train":
        # Training hyperparameters
        batch_size = 8
        learning_rate = 3e-4
        num_epochs = 3
        max_length = 128

        # Load dataset
        print("Loading and preprocessing dataset...")
        train_dataset = load_and_preprocess_data(tokenizer, max_length)
        train_dataloader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )

        # Initialize model
        print("Initializing model...")
        model = Transformer(model_args).to(device)

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params:,}")

        # Initialize optimizer and loss
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss(ignore_index=-100)

        # Training loop
        print("Starting training...")
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")

            # Train
            train_loss = train_epoch(
                model, train_dataloader, optimizer, criterion, device
            )
            print(f"Training Loss: {train_loss:.4f}")
            print(f"Training Perplexity: {math.exp(train_loss):.4f}")

        # Save model
        print("\nSaving model...")
        torch.save(model.state_dict(), "llama_wikitext_trained.pth")
        print("Model saved as 'llama_wikitext_trained.pth'")

    elif mode == "generate":
        # Initialize model
        model = Transformer(model_args).to(device)

        # Load trained weights
        print("Loading trained model...")
        model.load_state_dict(
            torch.load("llama_wikitext_trained.pth", map_location=device)
        )

        # Generate text with different methods
        prompt = "In a world where"
        print(f"Prompt: {prompt}\n")

        print("=== Greedy Decoding ===")
        generated_text = generate_text_greedy(model, tokenizer, prompt, max_length=50)
        print(f"Generated text: {generated_text}\n")


if __name__ == "__main__":
    import sys

    # Run with: python llama_template.py train
    # or: python llama_template.py generate
    mode = sys.argv[1] if len(sys.argv) > 1 else "train"
    main(mode)
