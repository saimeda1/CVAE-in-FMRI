"""
Models Module - VAE architectures (Unconditional and Conditional)
"""

import torch
import torch.nn as nn
from torch.nn import functional as F


class Unconditional_VAE(nn.Module):
    """Baseline: Unconditional VAE"""
    
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
        )
        
        self.mu_layer = nn.Linear(256, latent_dim)
        self.logvar_layer = nn.Linear(256, latent_dim)
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.ReLU(),
            nn.Linear(256, 512), nn.ReLU(),
            nn.Linear(512, input_dim),
        )
    
    def encode(self, x):
        h = self.encoder(x)
        return self.mu_layer(h), self.logvar_layer(h)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar, z


class Conditional_VAE(nn.Module):
    """Conditional VAE with semantic features"""
    
    def __init__(self, input_dim, semantic_dim, latent_dim):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim + semantic_dim, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
        )
        
        self.mu_layer = nn.Linear(256, latent_dim)
        self.logvar_layer = nn.Linear(256, latent_dim)
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + semantic_dim, 256), nn.ReLU(),
            nn.Linear(256, 512), nn.ReLU(),
            nn.Linear(512, input_dim),
        )
    
    def encode(self, x, c):
        xc = torch.cat([x, c], dim=1)
        h = self.encoder(xc)
        return self.mu_layer(h), self.logvar_layer(h)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)
    
    def decode(self, z, c):
        zc = torch.cat([z, c], dim=1)
        return self.decoder(zc)
    
    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, c), mu, logvar, z


def vae_loss(recon, x, mu, logvar, beta=1.0):
    """VAE loss: Reconstruction + Regularization"""
    recon_loss = F.mse_loss(recon, x, reduction='mean')
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    
    return recon_loss + beta * kl_loss, recon_loss, kl_loss
