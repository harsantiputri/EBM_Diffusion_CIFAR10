# EBM_001_DIFFUSION_CIFAR10/diffusion.py

import torch
import torch.nn.functional as F
from tqdm import tqdm # Added tqdm import for p_sample_loop

def linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=0.02):
    """
    Creates a linear schedule for beta values from beta_start to beta_end.
    """
    return torch.linspace(beta_start, beta_end, timesteps)

def extract(a, t, x_shape):
    """
    Extracts the value a[t] and reshapes it to match the batch dimension of x.
    This is used to multiply each image in a batch by the correct alpha/beta value.
    """
    batch_size = t.shape[0]
    # --- FIX IS HERE ---
    # The .cpu() call has been removed. `t` is now on the same device as `a`.
    out = a.gather(-1, t) 
    # -------------------
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

class DiffusionProcess:
    def __init__(self, timesteps=1000, device='cuda'):
        self.timesteps = timesteps
        self.device = device

        # Define the noise schedule (betas)
        self.betas = linear_beta_schedule(timesteps).to(self.device)

        # Pre-calculate the alphas and other terms from the DDPM paper
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # These are the terms used in the forward process q(x_t | x_0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

        # These are the terms used in the reverse process q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

    def q_sample(self, x_start, t, noise=None):
        """
        Forward process: q(x_t | x_0)
        Adds noise to an image x_start to create a noisy version x_t at timestep t.
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    @torch.no_grad()
    def p_sample(self, model, x, t, t_index):
        """
        Reverse process step: p(x_{t-1} | x_t)
        Denoises the image x at timestep t by one step.
        """
        betas_t = extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        sqrt_recip_alphas_t = extract(torch.sqrt(1.0 / self.alphas), t, x.shape)
        
        # Use the model to predict the noise
        # This is where our EBM's gradient is evaluated
        predicted_noise = model(x, t)
        
        # Calculate the mean of the distribution for x_{t-1}
        model_mean = sqrt_recip_alphas_t * (x - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t)

        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = extract(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)
            # Add noise to the mean to get the sample for x_{t-1}
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def p_sample_loop(self, model, shape):
        """
        Full reverse process loop to generate an image from pure noise.
        """
        img = torch.randn(shape, device=self.device)
        imgs = []

        for i in tqdm(reversed(range(0, self.timesteps)), desc='Sampling', total=self.timesteps):
            t = torch.full((shape[0],), i, device=self.device, dtype=torch.long)
            img = self.p_sample(model, img, t, i)
            if i % 50 == 0:
                imgs.append(img.cpu())
        
        imgs.append(img.cpu())
        return imgs

    @torch.no_grad()
    def sample(self, model, image_size, batch_size=16, channels=3):
        """
        Public method to generate samples.
        """
        shape = (batch_size, channels, image_size, image_size)
        # The final image is the last one in the sequence
        return self.p_sample_loop(model, shape)[-1]