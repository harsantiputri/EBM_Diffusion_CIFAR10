# EBM Training with Diffusion Objective on CIFAR-10

This repository contains the code for training an Energy-Based Model (EBM) using a modern, stable objective derived from Denoising Score Matching (the same principle behind DDPMs) on the CIFAR-10 dataset.

## Key Features
- **Model:** A U-Net architecture implicitly defines the time-dependent energy function.
- **Training:** Uses a stable denoising score-matching objective, avoiding unstable MCMC sampling during training.
- **Sampling:** Includes both a standard DDPM sampler and a faster, higher-quality DDIM sampler.
- **Monitoring:** Integrated with TensorBoard for live monitoring of loss and generated image quality.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/YOUR_USERNAME/EBM_Diffusion_CIFAR10.git
    cd EBM_Diffusion_CIFAR10
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Install Git LFS to download the model weights:**
    ```bash
    git lfs install
    git lfs pull
    ```

## How to Run

### Training
To train a new model from scratch:
```bash
python main.py --epochs 300 --batch-size 128 --learning-rate 1e-4
```
To resume training from a checkpoint:
```bash
python main.py --epochs 500 --resume_checkpoint weights/model_final.pth
```
Monitor training progress with TensorBoard:
```bash
# In a separate terminal
tensorboard --logdir=runs
```
