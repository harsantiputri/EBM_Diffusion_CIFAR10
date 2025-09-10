# EBM_001_DIFFUSION_CIFAR10/main.py

import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm
import os
import argparse
import numpy as np
import re # Import regex for parsing epoch number

# Import TensorBoard
from torch.utils.tensorboard import SummaryWriter

from model import SimpleUnet
from diffusion import DiffusionProcess

# Function to transform images to [-1, 1]
def get_transform():
    return torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

# Function to reverse the transform
def denormalize(tensor):
    return (tensor / 2.0) + 0.5

def train(args):
    # Setup directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # --- TENSORBOARD SETUP ---
    writer = SummaryWriter(f'runs/EBM_DIFFUSION_CIFAR10_lr{args.learning_rate}_bs{args.batch_size}')
    
    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load dataset
    dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=get_transform())
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    
    model = SimpleUnet().to(device)
    optimizer = Adam(model.parameters(), lr=args.learning_rate)
    
    # --- RESUME LOGIC ---
    start_epoch = 0
    if args.resume_checkpoint:
        try:
            print(f"Resuming training from checkpoint: {args.resume_checkpoint}")
            checkpoint = torch.load(args.resume_checkpoint, map_location=device)
            
            # This handles both old and new checkpoint formats
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                if 'optimizer_state_dict' in checkpoint:
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            else: # Handle old format
                model.load_state_dict(checkpoint)
                print("Warning: Loaded an old checkpoint format without optimizer state.")

            # Try to parse epoch number from filename, e.g., "model_epoch_100.pth"
            match = re.search(r'epoch_(\d+)', args.resume_checkpoint)
            if match:
                start_epoch = int(match.group(1))
                print(f"Successfully parsed epoch number. Resuming from epoch {start_epoch + 1}")
            else:
                 print("Could not parse epoch from checkpoint filename. Starting epoch count from 0 for logging.")

        except FileNotFoundError:
            print(f"Checkpoint not found at {args.resume_checkpoint}. Starting from scratch.")
            start_epoch = 0
        except Exception as e:
            print(f"Could not load checkpoint correctly: {e}. Starting from scratch.")
            start_epoch = 0
    # --------------------

    diffusion = DiffusionProcess(timesteps=args.timesteps, device=device)
    
    print(f"Will train from epoch {start_epoch + 1} to {args.epochs}.")
    global_step = start_epoch * len(dataloader)

    for epoch in range(start_epoch, args.epochs):
        current_epoch_num = epoch + 1
        progress_bar = tqdm(dataloader, desc=f"Epoch {current_epoch_num}/{args.epochs}")
        epoch_losses = []
        for i, (images, _) in enumerate(progress_bar):
            images = images.to(device)
            t = torch.randint(0, args.timesteps, (images.shape[0],), device=device).long()
            noise = torch.randn_like(images)
            x_t = diffusion.q_sample(x_start=images, t=t, noise=noise)
            predicted_noise = model(x_t, t)
            loss = F.mse_loss(noise, predicted_noise)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            progress_bar.set_postfix(loss=loss.item())
            epoch_losses.append(loss.item())
            writer.add_scalar('Loss/train_batch', loss.item(), global_step)
            global_step += 1

        avg_epoch_loss = np.mean(epoch_losses)
        writer.add_scalar('Loss/train_epoch_avg', avg_epoch_loss, current_epoch_num)

        if current_epoch_num % args.sample_interval == 0:
            model.eval()
            with torch.no_grad():
                generated_images = diffusion.sample(model, image_size=32, batch_size=64, channels=3)
                generated_images = denormalize(generated_images.clamp(-1, 1))
                grid = make_grid(generated_images, nrow=8)
                save_image(grid, os.path.join(args.output_dir, f"sample_epoch_{current_epoch_num}.png"))
                writer.add_image('Generated Images', grid, current_epoch_num)
            model.train()

        if current_epoch_num % args.checkpoint_interval == 0:
            # Save model and optimizer state for proper resuming
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join(args.checkpoint_dir, f"model_epoch_{current_epoch_num}.pth"))

    print("Training finished.")
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, os.path.join(args.checkpoint_dir, "model_final.pth"))
    
    writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an EBM with a diffusion objective on CIFAR-10.")
    parser.add_argument('--epochs', type=int, default=100, help='Total number of epochs to train for.')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size for training.')
    parser.add_argument('--learning-rate', type=float, default=1e-3, help='Learning rate for the optimizer.')
    parser.add_argument('--timesteps', type=int, default=1000, help='Number of diffusion timesteps.')
    parser.add_argument('--output_dir', type=str, default='outputs', help='Directory to save generated samples.')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Directory to save model checkpoints.')
    parser.add_argument('--sample_interval', type=int, default=5, help='Epoch interval for saving samples.')
    parser.add_argument('--checkpoint_interval', type=int, default=10, help='Epoch interval for saving checkpoints.')
    parser.add_argument('--resume_checkpoint', type=str, default=None, help='Path to checkpoint to resume training from.')
    
    args = parser.parse_args()
    train(args)