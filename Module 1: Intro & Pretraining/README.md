# Transformer Implementation from Scratch

This practical session focuses on implementing a basic transformer model from scratch and training it on the ImageNet dataset. The goal is to understand the core components of a transformer architecture by building it ourselves.

## Overview

In this session, we will:
1. Implement a transformer encoder from scratch
2. Use Hugging Face datasets to load and preprocess Cats vs. Dogs data
3. Train our transformer on for image classification. Should get near ~40% trivially.

## Requirements

- Python 3.7+
- PyTorch
- torchvision
- transformers (Hugging Face)
- datasets (Hugging Face)

Install the required packages using:

pip install torch torchvision transformers datasets

## Implementation Details

The `transformer.py` script contains:
- A custom implementation of the Transformer Encoder (that you will implement)
- Data loading and preprocessing using Hugging Face datasets
- Training loop for the transformer on ImageNet

## Running the Code

To run the training process:

`python transformer_imagenet.py`