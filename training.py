"""
Training Module - VAE training loop with detailed history tracking
"""

import numpy as np
import torch
import torch.optim as optim

from models import vae_loss


def train_vae(model, train_loader, val_loader, config, is_conditional=True):
    """Train VAE with detailed history tracking for visualization"""
    
    model = model.to(config.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Detailed history tracking
    history = {
        'train_loss': [], 'train_recon': [], 'train_kl': [],
        'val_loss': [], 'val_recon': [], 'val_kl': []
    }
    
    best_val_loss = np.inf
    patience = 0
    
    for epoch in range(config.EPOCHS):
        # Training phase
        model.train()
        train_loss, train_recon, train_kl = 0, 0, 0
        
        for batch in train_loader:
            if is_conditional:
                batch_fmri, batch_semantic = batch
                batch_fmri = batch_fmri.to(config.DEVICE)
                batch_semantic = batch_semantic.to(config.DEVICE)
            else:
                batch_fmri, _ = batch
                batch_fmri = batch_fmri.to(config.DEVICE)
                batch_semantic = None
            
            optimizer.zero_grad()
            
            if is_conditional:
                recon, mu, logvar, _ = model(batch_fmri, batch_semantic)
            else:
                recon, mu, logvar, _ = model(batch_fmri)
            
            loss, recon_loss, kl_loss = vae_loss(recon, batch_fmri, mu, logvar, config.BETA)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_recon += recon_loss.item()
            train_kl += kl_loss.item()
        
        # Validation phase
        model.eval()
        val_loss, val_recon, val_kl = 0, 0, 0
        
        with torch.no_grad():
            for batch in val_loader:
                if is_conditional:
                    batch_fmri, batch_semantic = batch
                    batch_fmri = batch_fmri.to(config.DEVICE)
                    batch_semantic = batch_semantic.to(config.DEVICE)
                else:
                    batch_fmri, _ = batch
                    batch_fmri = batch_fmri.to(config.DEVICE)
                
                if is_conditional:
                    recon, mu, logvar, _ = model(batch_fmri, batch_semantic)
                else:
                    recon, mu, logvar, _ = model(batch_fmri)
                
                loss, recon_loss, kl_loss = vae_loss(recon, batch_fmri, mu, logvar, config.BETA)
                
                val_loss += loss.item()
                val_recon += recon_loss.item()
                val_kl += kl_loss.item()
        
        # Normalize by number of batches
        train_loss /= len(train_loader)
        train_recon /= len(train_loader)
        train_kl /= len(train_loader)
        val_loss /= len(val_loader)
        val_recon /= len(val_loader)
        val_kl /= len(val_loader)
        
        # Store history
        history['train_loss'].append(train_loss)
        history['train_recon'].append(train_recon)
        history['train_kl'].append(train_kl)
        history['val_loss'].append(val_loss)
        history['val_recon'].append(val_recon)
        history['val_kl'].append(val_kl)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience = 0
        else:
            patience += 1
        
        if patience >= 20:
            print(f"Early stopping at epoch {epoch+1}")
            break
        
        # Progress output
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{config.EPOCHS} | "
                  f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
    
    return model, history
