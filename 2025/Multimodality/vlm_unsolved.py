#!/usr/bin/env python3
"""
Simplified Vision-Language Model Training Script
Trains a small VLM (Qwen + SigLIP) on LaTeX OCR dataset
- No gradient accumulation for simplicity
- Supports only fp32 and bf16 (no fp16)
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    SiglipVisionModel,
    SiglipImageProcessor,
    get_linear_schedule_with_warmup,
)
from datasets import load_dataset
from PIL import Image
from typing import Optional
from dataclasses import dataclass
from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.panel import Panel

# Setup Rich console
console = Console()


@dataclass
class TrainingConfig:
    # Model configs
    vision_model_name: str = "google/siglip-base-patch16-224"
    language_model_name: str = "Qwen/Qwen2-0.5B"  # Small Qwen model

    # Training configs
    batch_size: int = 8
    learning_rate: float = 2e-5
    warmup_steps: int = 500
    num_epochs: int = 3
    max_length: int = 512

    # Dataset configs
    dataset_name: str = "linxy/LaTeX_OCR"
    train_samples: Optional[int] = None  # None for full dataset

    # System configs
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_bf16: bool = True  # Use bfloat16 if True, otherwise fp32
    save_steps: int = 1000
    logging_steps: int = 100
    output_dir: str = "./vlm_latex_ocr_checkpoint"

    # Connector configs
    connector_hidden_size: int = 1024
    connector_num_layers: int = 2


class VisionLanguageConnector(nn.Module):
    """MLP connector to project vision features to language model space"""

    def __init__(
        self,
        vision_hidden_size: int,
        language_hidden_size: int,
        hidden_size: int = 1024,
        num_layers: int = 2,
    ):
        super().__init__()

        layers = []
        in_size = vision_hidden_size

        # Task 1: Implement the MLP connector

    def forward(self, vision_features: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Not implemented")


class VisionLanguageModel(nn.Module):
    """Simple VLM combining vision encoder, connector, and language model"""

    def __init__(self, config: TrainingConfig):
        super().__init__()

        # Load vision model
        self.vision_model = SiglipVisionModel.from_pretrained(config.vision_model_name)
        self.image_processor = SiglipImageProcessor.from_pretrained(
            config.vision_model_name
        )

        # Load language model
        torch_dtype = torch.bfloat16 if config.use_bf16 else torch.float32

        self.language_model = torch.compile(
            AutoModelForCausalLM.from_pretrained(
                config.language_model_name, torch_dtype=torch_dtype
            )
        )
        self.tokenizer = AutoTokenizer.from_pretrained(config.language_model_name)

        # Set pad token if not exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Get hidden sizes
        vision_hidden_size = self.vision_model.config.hidden_size
        language_hidden_size = self.language_model.config.hidden_size

        # Create connector
        self.connector = VisionLanguageConnector(
            vision_hidden_size=vision_hidden_size,
            language_hidden_size=language_hidden_size,
            hidden_size=config.connector_hidden_size,
            num_layers=config.connector_num_layers,
        )

        # Task 2: Freeze vision model (optional, can be unfrozen for fine-tuning)

        # Task 3: Freeze language model (optional, can be unfrozen for fine-tuning)


    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ):
        # Extract vision features
        vision_outputs = self.vision_model(pixel_values=pixel_values)
        vision_features = (
            vision_outputs.last_hidden_state
        )  # [batch, num_patches, hidden_size]

        # Task: 4 Project vision features to language model space
        projected_features =   # [batch, num_patches, lm_hidden_size]

        # Get language model embeddings
        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)

        # Concatenate vision and text embeddings
        # Simple strategy: prepend vision tokens to text tokens
        batch_size = pixel_values.shape[0]
        vision_attention = torch.ones(
            batch_size,
            projected_features.shape[1],
            device=projected_features.device,
            dtype=attention_mask.dtype,
        )

        combined_embeds = torch.cat([projected_features, inputs_embeds], dim=1)
        combined_attention = torch.cat([vision_attention, attention_mask], dim=1)

        # Adjust labels if provided (shift for vision tokens)
        if labels is not None:
            # Add -100 (ignore index) for vision token positions
            vision_labels = torch.full(
                (batch_size, projected_features.shape[1]),
                -100,
                device=labels.device,
                dtype=labels.dtype,
            )
            combined_labels = torch.cat([vision_labels, labels], dim=1)
        else:
            combined_labels = None

        # Forward through language model
        outputs = self.language_model(
            inputs_embeds=combined_embeds,
            attention_mask=combined_attention,
            labels=combined_labels,
            return_dict=True,
        )

        return outputs


class LaTeXOCRDataset(Dataset):
    """Dataset for LaTeX OCR training"""

    def __init__(self, dataset, tokenizer, image_processor, max_length: int = 512):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        # Process image
        image = item["image"]
        if not isinstance(image, Image.Image):
            image = Image.open(image).convert("RGB")

        pixel_values = self.image_processor(image, return_tensors="pt")["pixel_values"][
            0
        ]

        # Process text (LaTeX formula)
        text = item["text"]
        # Add instruction prefix for better performance
        instruction = "Convert the image to LaTeX: "
        full_text = instruction + text

        # Tokenize
        encoding = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        # Create labels (shift input_ids by 1 for autoregressive training)
        labels = encoding["input_ids"][0].clone()
        # Set padding tokens to -100 (ignore index)
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "pixel_values": pixel_values,
            "input_ids": encoding["input_ids"][0],
            "attention_mask": encoding["attention_mask"][0],
            "labels": labels,
        }


def train_model(config: TrainingConfig):
    """Main training function"""

    # Display training configuration
    config_table = Table(
        title="Training Configuration", show_header=True, header_style="bold magenta"
    )
    config_table.add_column("Parameter", style="cyan", no_wrap=True)
    config_table.add_column("Value", style="yellow")

    config_table.add_row("Vision Model", config.vision_model_name)
    config_table.add_row("Language Model", config.language_model_name)
    config_table.add_row("Batch Size", str(config.batch_size))
    config_table.add_row("Learning Rate", f"{config.learning_rate:.2e}")
    config_table.add_row("Epochs", str(config.num_epochs))
    config_table.add_row("Device", config.device)
    config_table.add_row("Precision", "bfloat16" if config.use_bf16 else "float32")

    console.print(config_table)
    console.print()

    # Load dataset
    console.print(f"[bold cyan]Loading dataset:[/bold cyan] {config.dataset_name}")
    dataset = load_dataset(config.dataset_name, split="train")

    if config.train_samples:
        dataset = dataset.select(range(config.train_samples))

    # Initialize model
    console.rule("[bold yellow]Model Initialization[/bold yellow]")
    with console.status("[bold green]Loading models...", spinner="dots"):
        model = VisionLanguageModel(config)
        model.to(config.device)
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    console.print("[green]✓[/green] Model initialized successfully!")
    console.print(f"  Total parameters: [bold]{total_params:,}[/bold]")
    console.print(f"  Trainable parameters: [bold]{trainable_params:,}[/bold]")
    console.print()

    # Create dataset
    train_dataset = LaTeXOCRDataset(
        dataset, model.tokenizer, model.image_processor, max_length=config.max_length
    )

    # Create dataloader
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    # Setup optimizer
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=0.01,
    )

    # Setup scheduler
    num_training_steps = len(train_loader) * config.num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=num_training_steps,
    )

    # Training loop
    console.rule("[bold green]Training[/bold green]")
    global_step = 0
    model.train()

    # Track training metrics
    training_history = {"steps": [], "losses": [], "lrs": []}

    for epoch in range(config.num_epochs):
        epoch_loss = 0

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(
                f"Epoch {epoch + 1}/{config.num_epochs}", total=len(train_loader)
            )

            for step, batch in enumerate(train_loader):
                # Move batch to device
                pixel_values = batch["pixel_values"].to(config.device)
                input_ids = batch["input_ids"].to(config.device)
                attention_mask = batch["attention_mask"].to(config.device)
                labels = batch["labels"].to(config.device)

                # Forward pass
                if config.use_bf16:
                    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                        outputs = model(
                            pixel_values=pixel_values,
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels,
                        )
                        loss = outputs.loss
                else:
                    outputs = model(
                        pixel_values=pixel_values,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
                    loss = outputs.loss

                # Backward pass
                loss.backward()

                # Update weights
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # Logging
                if global_step % config.logging_steps == 0:
                    current_lr = scheduler.get_last_lr()[0]
                    console.print(
                        f"[bold magenta]Step {global_step}[/bold magenta] - "
                        f"[yellow]Loss: {loss.item():.4f}[/yellow], "
                        f"[cyan]LR: {current_lr:.6f}[/cyan]"
                    )

                    # Track metrics
                    training_history["steps"].append(global_step)
                    training_history["losses"].append(loss.item())
                    training_history["lrs"].append(current_lr)

                # Save checkpoint
                if global_step % config.save_steps == 0:
                    save_checkpoint(model, config, global_step)

                epoch_loss += loss.item()
                progress.update(
                    task,
                    advance=1,
                    description=f"Epoch {epoch + 1}/{config.num_epochs} - Loss: {loss.item():.4f}",
                )

        # Log epoch metrics
        avg_epoch_loss = epoch_loss / len(train_loader)
        console.print(
            Panel(
                f"[bold green]Epoch {epoch + 1} Complete[/bold green]\n"
                f"Average Loss: [bold yellow]{avg_epoch_loss:.4f}[/bold yellow]",
                expand=False,
            )
        )

    # Save final model
    save_checkpoint(model, config, global_step, final=True)

    # Display training summary
    console.rule("[bold green]Training Summary[/bold green]")

    summary_table = Table(show_header=True, header_style="bold cyan")
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Value", style="yellow")

    summary_table.add_row("Total Steps", f"{global_step:,}")
    summary_table.add_row("Total Epochs", str(config.num_epochs))
    if training_history["losses"]:
        summary_table.add_row("Final Loss", f"{training_history['losses'][-1]:.4f}")
        summary_table.add_row("Min Loss", f"{min(training_history['losses']):.4f}")
    summary_table.add_row("Output Directory", config.output_dir + "_final")

    console.print(summary_table)
    console.print("\n[bold green]✓ Training completed successfully![/bold green]")


def save_checkpoint(model, config, step, final=False):
    """Save model checkpoint"""
    output_dir = config.output_dir if not final else f"{config.output_dir}_final"
    os.makedirs(output_dir, exist_ok=True)

    # Save model components
    # Save connector state dict (it's a simple nn.Module, not a HuggingFace model)
    torch.save(model.connector.state_dict(), f"{output_dir}/connector.pt")

    # Save language model and tokenizer (these are HuggingFace models)
    model.language_model.save_pretrained(f"{output_dir}/language_model")
    model.tokenizer.save_pretrained(f"{output_dir}/tokenizer")

    # Save config and training state
    torch.save(
        {
            "step": step,
            "config": config,
            "vision_model_name": config.vision_model_name,
            "connector_config": {
                "vision_hidden_size": model.vision_model.config.hidden_size,
                "language_hidden_size": model.language_model.config.hidden_size,
                "hidden_size": config.connector_hidden_size,
                "num_layers": config.connector_num_layers,
            },
        },
        f"{output_dir}/training_state.pt",
    )

    console.print(f"[green]✓ Checkpoint saved to[/green] [bold]{output_dir}[/bold]")


def load_trained_model(checkpoint_dir: str, device: str = "cuda"):
    """Load a trained VLM checkpoint for inference"""

    # Load training state
    training_state = torch.load(
        f"{checkpoint_dir}/training_state.pt", map_location=device, weights_only=False
    )
    config = training_state["config"]

    # Create model instance
    model = VisionLanguageModel(config)

    # Load saved components
    # Load connector state dict (now saved as connector.pt)
    model.connector.load_state_dict(
        torch.load(f"{checkpoint_dir}/connector.pt", map_location=device)
    )

    torch_dtype = torch.bfloat16 if config.use_bf16 else torch.float32

    model.language_model = AutoModelForCausalLM.from_pretrained(
        f"{checkpoint_dir}/language_model", torch_dtype=torch_dtype
    )
    model.tokenizer = AutoTokenizer.from_pretrained(f"{checkpoint_dir}/tokenizer")

    # Ensure all model components have the correct dtype
    model.to(device)
    if config.use_bf16:
        model = model.to(torch.bfloat16)
    model.eval()

    return model, config


def generate_latex(
    model,
    image_path: str,
    max_new_tokens: int = 256,
    temperature: float = 0.1,
    device: str = "cuda",
):
    """Generate LaTeX code from an image"""

    # Load and process image
    image = Image.open(image_path).convert("RGB")
    pixel_values = model.image_processor(image, return_tensors="pt")["pixel_values"].to(
        device
    )

    # Prepare instruction prompt
    instruction = "Convert the image to LaTeX: "
    input_ids = model.tokenizer(instruction, return_tensors="pt")["input_ids"].to(
        device
    )

    # Extract vision features
    with torch.no_grad():
        vision_outputs = model.vision_model(pixel_values=pixel_values)
        vision_features = vision_outputs.last_hidden_state
        projected_features = model.connector(vision_features)

    # Create combined embeddings
    text_embeds = model.language_model.get_input_embeddings()(input_ids)
    # Ensure both embeddings have the same dtype
    text_embeds = text_embeds.to(projected_features.dtype)
    combined_embeds = torch.cat([projected_features, text_embeds], dim=1)

    # Generate
    with torch.no_grad():
        outputs = model.language_model.generate(
            inputs_embeds=combined_embeds,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=model.tokenizer.pad_token_id,
            eos_token_id=model.tokenizer.eos_token_id,
        )

    # Decode output (skip the instruction part)
    generated_text = model.tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Remove the instruction prefix from output
    latex_output = generated_text.replace(instruction, "").strip()

    return latex_output


def inference_example():
    """Example of how to use the trained model for inference"""

    # Path to your trained model
    checkpoint_dir = "./vlm_latex_ocr_checkpoint_final"

    # Load model
    console.print("[bold cyan]Loading trained model...[/bold cyan]")
    model, config = load_trained_model(checkpoint_dir)

    # Example 1: Single image inference
    image_path = "path/to/your/latex_formula.png"
    if os.path.exists(image_path):
        latex_code = generate_latex(model, image_path)
        console.print(f"[yellow]Generated LaTeX:[/yellow] {latex_code}")

    # Example 2: Batch inference on test dataset
    console.print("[bold cyan]Running batch inference on test samples...[/bold cyan]")
    test_dataset = load_dataset("linxy/LaTeX_OCR", split="train")
    test_samples = test_dataset.select(range(5))  # Take 5 samples

    for i, sample in enumerate(test_samples):
        # Save image temporarily
        temp_image_path = f"temp_test_{i}.png"
        sample["image"].save(temp_image_path)

        # Generate LaTeX
        generated_latex = generate_latex(model, temp_image_path)
        ground_truth = sample["text"]

        console.print(f"\n[bold]Sample {i + 1}:[/bold]")
        console.print(f"[green]Ground Truth:[/green] {ground_truth}")
        console.print(f"[yellow]Generated:[/yellow]    {generated_latex}")
        console.print("-" * 50)

        # Clean up
        os.remove(temp_image_path)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "test":
        # Test training for 10 steps with save/load
        # Set cache dir to /tmp to avoid disk space issues

        config = TrainingConfig(
            train_samples=1000,  # Only use 10 samples
            save_steps=1000,  # Save after 10 steps
            logging_steps=1,  # Log every step
            num_epochs=1,  # One epoch only
            batch_size=1,  # Batch size 1, so 10 samples = 10 steps
            output_dir="test_checkpoint_10steps",
        )
        train_model(config)

        # Test loading the saved model
        console.print("\n[bold cyan]Testing model loading...[/bold cyan]")
        loaded_model, loaded_config = load_trained_model("test_checkpoint_10steps")
        console.print("[bold green]✓ Model loaded successfully![/bold green]")

        # Quick inference test
        console.print("[bold cyan]Testing inference with loaded model...[/bold cyan]")
        test_dataset = load_dataset("linxy/LaTeX_OCR", split="train")
        test_sample = test_dataset[0]

        # Save test image temporarily
        test_image_path = "test_inference_image.png"
        test_sample["image"].save(test_image_path)

        result = generate_latex(loaded_model, test_image_path)
        console.print(f"[yellow]Inference result:[/yellow] {result}")
        console.print(f"[green]Ground truth:[/green] {test_sample['text']}")

        # Clean up
        os.remove(test_image_path)

    elif len(sys.argv) > 1 and sys.argv[1] == "inference":
        # Run inference example
        inference_example()
    else:
        # Run training
        config = TrainingConfig()

        # Optional: Override configs via command line or environment variables
        # Example: config.batch_size = int(os.environ.get('BATCH_SIZE', 8))

        # Start training
        train_model(config)
