# EBM Training with Diffusion Objective on CIFAR-10

This project demonstrates how to train an Energy-Based Model (EBM) using a modern, stable objective derived from Denoising Score Matching, which is the core principle behind Denoising Diffusion Probabilistic Models (DDPMs).

## Core Idea

Instead of using traditional contrastive divergence with MCMC for training, which can be unstable, we train a neural network to approximate the score function ($\nabla_x \log p(x)$) of the data distribution at various noise levels. For an EBM where $p(x) \propto \exp(-E(x))$, the score is simply the negative gradient of the energy: $-\nabla_x E(x)$.

The training process is as follows:
1.  **Forward Process:** Take a real image from CIFAR-10 and add a known amount of Gaussian noise based on a random timestep `t`.
2.  **Training Objective:** Train a U-Net model to predict the noise that was added. The loss is a simple Mean Squared Error (MSE) between the true noise and the predicted noise. This objective effectively trains the model to learn the score function.
3.  **Sampling:** To generate new images, we start with pure Gaussian noise and iteratively apply the learned model to denoise the image over `T` steps, reversing the forward process.

This method is powerful because it replaces the unstable MCMC sampling loop during training with a simple and efficient denoising objective.

## Project Structure

- `main.py`: The main script for training the model, handling data loading, and saving samples.
- `diffusion.py`: Contains the logic for the diffusion process (noise scheduling, forward noising, and reverse sampling).
- `model.py`: Defines the U-Net architecture with time embeddings.
- `requirements.txt`: Python dependencies.

## How to Run

1.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Start training:**
    ```bash
    python main.py --epochs 100 --batch-size 128 --learning-rate 1e-3
    ```

Training will create an `outputs/` directory to save generated image samples periodically and a `checkpoints/` directory for model weights.