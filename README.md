# EBM Training with Diffusion Objective on CIFAR-10

This repository contains the code for training an Energy-Based Model (EBM) using a modern, stable objective derived from Denoising Score Matching (the same principle behind DDPMs) on the CIFAR-10 dataset.

![Sample Output](outputs/ddim_sharp_sample.png)
*(Note: To show an image like this, create a good sample with `sample.py`, move it to the `outputs` folder, and `git add outputs/ddim_sharp_sample.png` and commit it. Your `.gitignore` allows this because we didn't use a trailing slash `outputs/`)*

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
