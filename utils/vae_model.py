"""
VAE (Variational Autoencoder) model for generating synthetic pose data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, latent_dim * 2)  # Output: mean and log-variance
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()  # For data in range [0, 1]
        )
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick for sampling from latent distribution."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        # Pass through encoder
        h = self.encoder(x)
        mu, logvar = torch.chunk(h, 2, dim=-1)  # Split into mean and log-variance
        
        # Sample from latent space
        z = self.reparameterize(mu, logvar)
        
        # Reconstruct data
        x_recon = self.decoder(z)
        
        return x_recon, mu, logvar

def vae_loss_function(recon_x, x, mu, logvar):
    """VAE loss function combining reconstruction loss and KL divergence."""
    # Reconstruction loss (using binary cross entropy)
    recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')
    
    # KL divergence
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return recon_loss + kl_div

def generate_synthetic_data(model, num_samples, latent_dim, device='cpu'):
    """Generate synthetic data using trained VAE model."""
    model.eval()
    with torch.no_grad():
        # Sample from latent space (normal distribution)
        z = torch.randn(num_samples, latent_dim).to(device)
        
        # Pass through decoder
        synthetic_data = model.decoder(z)
    
    return synthetic_data
