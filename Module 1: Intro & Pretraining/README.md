# Day 1
## Transformer Implementation from Scratch

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

`python transformer.py`

# Day 2
# LLaMA Implementation Task

## Overview

This project involves implementing a simplified version of the LLaMA architecture. The goal is to create a functional transformer-based language model and train it on a small dataset.

## Task Requirements

1. Complete the implementation of the following components:
   - RMSNorm (Root Mean Square Layer Normalization)
   - SwiGLU activation function in the FeedForward class
   - Causal mask generation in the Transformer class

2. Debug and test the implementation to ensure all components work correctly.

3. Train the model on the provided WikiText dataset.

4. Evaluate the model's performance and report the results.

## Getting Started

1. Clone this repository to your local machine.
2. Install the required dependencies:
   ```
   pip install torch torchvision torchaudio
   pip install transformers datasets tqdm
   ```
3. Review the existing code in the main Python file.

## Task Breakdown

### 1. Implement RMSNorm

In the `RMSNorm` class, implement the Root Mean Square Layer Normalization. This is a variant of Layer Normalization that uses RMS statistics.

### 2. Implement SwiGLU Activation

In the `FeedForward` class, implement the SwiGLU activation function. This is a variant of the GLU (Gated Linear Unit) activation.

### 3. Implement Causal Mask Generation

In the `Transformer` class, implement the causal mask generation for sequences longer than 1 token. This mask ensures that the model can only attend to previous tokens in the sequence.

### 4. Debug and Test

After implementing the missing components, run the script and debug any issues that arise. Ensure that the model can process input data and produce outputs without errors.

### 5. Train the Model

Use the provided training loop to train the model on the WikiText dataset. You may need to adjust hyperparameters or training settings for optimal performance.

### 6. Evaluate and Report

After training, evaluate the model's performance. Report the following metrics:
- Final training loss
- Perplexity on the validation set (if applicable)
- Sample outputs from the model

# Generation Task
# LLaMA Implementation Task

## Overview

This project involves implementing a simplified version of the LLaMA (Large Language Model Meta AI) architecture. The goal is to create a functional transformer-based language model, train it on a small dataset, and use it for text generation.

## Task Requirements

1. Complete the implementation of the following components:
   - RMSNorm (Root Mean Square Layer Normalization)
   - SwiGLU activation function in the FeedForward class
   - Causal mask generation in the Transformer class

2. Debug and test the implementation to ensure all components work correctly.

3. Train the model on the provided WikiText dataset.

4. Evaluate the model's performance and report the results.

5. Implement text generation using the trained model.

## Getting Started

1. Clone this repository to your local machine.
2. Install the required dependencies:
   ```
   pip install torch torchvision torchaudio
   pip install transformers datasets tqdm
   ```
3. Review the existing code in the main Python file.

## Task Breakdown

### 1. Implement RMSNorm

In the `RMSNorm` class, implement the Root Mean Square Layer Normalization. This is a variant of Layer Normalization that uses RMS statistics.

### 2. Implement SwiGLU Activation

In the `FeedForward` class, implement the SwiGLU activation function. This is a variant of the GLU (Gated Linear Unit) activation.

### 3. Implement Causal Mask Generation

In the `Transformer` class, implement the causal mask generation for sequences longer than 1 token. This mask ensures that the model can only attend to previous tokens in the sequence.

### 4. Debug and Test

After implementing the missing components, run the script and debug any issues that arise. Ensure that the model can process input data and produce outputs without errors.

### 5. Train the Model

Use the provided training loop to train the model on the WikiText dataset. You may need to adjust hyperparameters or training settings for optimal performance.

### 6. Evaluate and Report

After training, evaluate the model's performance. Report the following metrics:
- Final training loss
- Perplexity on the validation set (if applicable)
- Sample outputs from the model

### 7. Implement Text Generation

After training the model, implement text generation functionality. This involves:

1. Loading the trained model weights.
2. Implementing different text generation strategies:
   - Greedy decoding
   - Sampling
   - Top-k sampling
   - Top-p (nucleus) sampling

3. Creating a function to generate text given a prompt.

## Text Generation

The `llama_generation.py` script provides functionality for generating text using the trained LLaMA model. Here's an overview of the text generation process:

1. The script loads the trained model weights and initializes the model with the same configuration used during training.

2. It uses the GPT-2 tokenizer for encoding and decoding text.

3. The `generate_text_greedy` function implements greedy decoding for text generation. Given a prompt, it generates text by selecting the most probable token at each step.

4. Additional generation methods (sampling, top-k, and top-p) are outlined but not implemented. These are left as an exercise for further improvement.

To generate text:

1. Ensure you have trained the model and saved the weights as `llama_wikitext_trained.pth`.
2. Run the `llama_generation.py` script.
3. The script will use a default prompt "In a world where" and generate text based on this prompt.

You can modify the `main` function in `llama_generation.py` to experiment with different prompts, generation methods, and parameters.

Good luck with your implementation and text generation experiments!