# ViT-VQVAE Homework Assignment: Implementing Vector Quantization and Loss Functions

This project provides a framework for implementing a Vector Quantized Variational Autoencoder (VQ-VAE) using Vision Transformers (ViT) for both the encoder and decoder. This is a assignment where you will be responsible for implementing key components of the model.

## Overview

In this assignment, you will work with a partially implemented ViT-VQVAE model. Your task is to complete the implementation by adding crucial components related to vector quantization and loss calculation.

## Components

The main components of this project are:

- `PositionalEncoding2D`: Implements 2D sinusoidal positional encoding for image data.
- `ViTEncoder`: Vision Transformer-based encoder.
- `ViTDecoder`: Vision Transformer-based decoder.
- `VectorQuantizer`: Partially implemented vector quantization layer (to be completed by you).
- `ViT_VQVAE`: The main model class that combines all components (requires your implementation for loss calculation).

## Your Tasks

1. **Implement Vector Quantization Selection**:
   - Location: In the `VectorQuantizer` class
   - Task: Implement the mechanism for selecting the nearest embedding in the codebook for each encoder output.
   - Hints: Consider using distance calculations between the encoder output and the embedding vectors.

2. **Implement Loss Functions**:
   - Location: In the `ViT_VQVAE` class, specifically the `calculate_loss` method
   - Task: Implement the loss function for training the VQ-VAE
   - Components to consider:
     - Reconstruction loss
     - Vector quantization loss (commitment loss)
     - Codebook loss

3. **Integrate Your Implementations**:
   - Ensure your implemented components work correctly with the rest of the provided code.
   - Test your implementation with different hyperparameters and analyze the results.

4. **Bonus: Implement Codebook Diversity Loss**
    - Notice that codebook utilization is quite small.
    - Not included in slides :) Search for yourself.
## Installation

To use this project, you need Python 3.7+ and PyTorch 1.7+. Install the required packages using:

```bash
pip install torch torchvision tqdm matplotlib datasets
```

## Usage

After implementing the required components, you can train the model using:

```bash
python main.py --resolution 64 --batch_size 32 --num_epochs 100 --learning_rate 1e-4 --latent_dim 256 --num_embeddings 512 --num_heads 8 --num_layers 6 --patch_size 8
```
---

Note: The provided code structure is a starting point. Feel free to modify or extend it as needed to complete the assignment successfully.