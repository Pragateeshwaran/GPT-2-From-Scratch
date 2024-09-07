# GPT-2 From Scratch

This project implements a GPT-2 model from scratch using PyTorch. It includes data preprocessing, model architecture, and a custom training loop.

## Project Structure

- `dataset.py`: Handles data preprocessing and tokenization
- `main.py`: Contains the model architecture, training loop, and inference code

## Features

- Implementation of GPT-2 architecture
- Custom data loading and preprocessing
- Configurable model parameters
- Training with learning rate scheduling and gradient accumulation
- Validation loss evaluation
- Checkpoint saving
- Text generation capabilities

## Requirements
 
- PyTorch
- tiktoken
- numpy
- tqdm
- NVIDIA GPU (RTX 3060 or better recommended for training)

## Usage

1. Prepare your dataset:
   - Update the `local_dir` and `remote_name` variables in `dataset.py` to point to your dataset location.
   - Run `dataset.py` to preprocess and tokenize your data.

2. Train the model:
   - Adjust the model configuration in the `GPTConfig` class in `main.py` if needed.
   - Set the appropriate paths for data loading and log saving in `main.py`.
   - Run `main.py` to start training.

3. Generate text:
   - The training loop includes periodic text generation. You can modify the prompt in the "paragraph" section of `main.py`.

## Model Architecture

The GPT-2 model implemented here includes:
- Token and position embeddings
- Multiple transformer blocks with self-attention and feed-forward layers
- Layer normalization
- Linear output layer

## Training

The training process includes:
- Custom learning rate scheduling with warmup and cosine decay
- Gradient accumulation for larger effective batch sizes
- Gradient clipping
- Mixed precision training using `torch.autocast`
- Periodic validation loss evaluation
- Model checkpointing

### Extended Training

This model underwent serious training for 8 days on an NVIDIA RTX 3060 GPU. This extended training period allowed for better convergence and performance of the model.

## Logging

Training progress, including step number, training loss, and validation loss, is logged to a file in the specified log directory.
