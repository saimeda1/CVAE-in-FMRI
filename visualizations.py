"""
Visualizations Module - All plotting functions for the pipeline
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.manifold import TSNE

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False


def plot_vae_tradeoff(history, config):
    """Visualize BCE vs KL trade-off during training"""
    print("Generating: BCE vs KL Trade-Off plot...")
    
    train_recon = history['train_recon']
    train_kl = history['train_kl']
    epochs = range(len(train_recon))
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    scatter = ax.scatter(train_kl, train_recon,
                        c=epochs, cmap='coolwarm',
                        s=80, alpha=0.7, edgecolors='black', linewidth=0.8)
    
    # Mark start and best points
    ax.scatter(train_kl[0], train_recon[0],
              marker='D', s=300, c='green',
              edgecolors='black', linewidth=2, label='Start (Epoch 0)', zorder=10)
    
    best_epoch = np.argmin(np.array(train_recon) + np.array(train_kl))
    ax.scatter(train_kl[best_epoch], train_recon[best_epoch],
              marker='*', s=500, c='orange',
              edgecolors='black', linewidth=2, label=f'Best (Epoch {best_epoch})', zorder=10)
    
    ax.set_xlabel('KL Divergence (Semantic Complexity)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Reconstruction Loss (fMRI MSE)', fontsize=13, fontweight='bold')
    ax.set_title('VAE Optimization Trajectory: Reconstruction vs. Regularization',
                fontsize=15, fontweight='bold', pad=20)
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(alpha=0.3, linestyle='--')
    
    cbar = plt.colorbar(scatter, ax=ax, label='Training Epoch', pad=0.02)
    cbar.ax.tick_params(labelsize=10)
    
    plt.tight_layout()
    
    if config.SAVE_FIGURES:
        fig.savefig(config.OUTPUT_DIR / 'vae_tradeoff.png', dpi=config.FIGURE_DPI, bbox_inches='tight')
    
    plt.close()
    print("✓ Saved: vae_tradeoff.png")


def plot_training_dynamics(history, config):
    """Three-panel plot of VAE training dynamics"""
    print("Generating: Training Dynamics (3-panel)...")
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    epochs = range(len(history['train_loss']))
    
    # Panel 1: Total Loss (ELBO)
    ax1 = axes[0]
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Total', linewidth=2.5, alpha=0.8)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Val Total', linewidth=2.5, alpha=0.8)
    ax1.set_ylabel('Total Loss\n(Lower = Better)', fontsize=12, fontweight='bold')
    ax1.set_title('VAE Training Dynamics: ELBO Components', fontsize=14, fontweight='bold', pad=15)
    ax1.legend(fontsize=11, loc='upper right')
    ax1.grid(alpha=0.3, linestyle='--')
    
    # Panel 2: Reconstruction Loss
    ax2 = axes[1]
    ax2.plot(epochs, history['train_recon'], 'orange', label='Train Recon', linewidth=2.5, alpha=0.8)
    ax2.plot(epochs, history['val_recon'], 'blue', label='Val Recon', linewidth=2.5, alpha=0.8)
    ax2.set_ylabel('Reconstruction Loss\n(MSE)', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=11, loc='upper right')
    ax2.grid(alpha=0.3, linestyle='--')
    
    # Panel 3: KL Divergence
    ax3 = axes[2]
    ax3.plot(epochs, history['train_kl'], 'red', label='Train KL', linewidth=2.5, alpha=0.8)
    ax3.plot(epochs, history['val_kl'], 'blue', label='Val KL', linewidth=2.5, alpha=0.8)
    ax3.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax3.set_ylabel('KL Divergence\n(Semantic Complexity)', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=11, loc='upper right')
    ax3.grid(alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    if config.SAVE_FIGURES:
        fig.savefig(config.OUTPUT_DIR / 'training_dynamics.png', dpi=config.FIGURE_DPI, bbox_inches='tight')
    
    plt.close()
    print("✓ Saved: training_dynamics.png")


def plot_kl_per_dimension(model, dataloader, config):
    """Visualize KL divergence per latent dimension"""
    print("Generating: KL per Dimension analysis...")
    
    model.eval()
    all_mu, all_logvar = [], []
    
    with torch.no_grad():
        for batch_fmri, batch_semantic in dataloader:
            batch_fmri = batch_fmri.to(config.DEVICE)
            batch_semantic = batch_semantic.to(config.DEVICE)
            mu, logvar = model.encode(batch_fmri, batch_semantic)
            all_mu.append(mu.cpu().numpy())
            all_logvar.append(logvar.cpu().numpy())
    
    all_mu = np.vstack(all_mu)
    all_logvar = np.vstack(all_logvar)
    
    # KL per dimension: 0.5 * (mu^2 + exp(logvar) - 1 - logvar)
    kl_per_dim = 0.5 * (all_mu**2 + np.exp(all_logvar) - 1 - all_logvar)
    kl_per_dim_mean = kl_per_dim.mean(axis=0)
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    dims = np.arange(config.LATENT_DIM)
    colors = ['red' if kl < 0.1 else 'steelblue' for kl in kl_per_dim_mean]
    
    bars = ax.bar(dims, kl_per_dim_mean, color=colors, alpha=0.8, edgecolor='black', linewidth=1.2)
    
    ax.axhline(0.1, color='red', linestyle='--', linewidth=2.5,
              label='Inactive Threshold (KL < 0.1)', alpha=0.7)
    
    ax.set_xlabel('Latent Dimension', fontsize=13, fontweight='bold')
    ax.set_ylabel('Mean KL Divergence', fontsize=13, fontweight='bold')
    ax.set_title(f'KL Contribution per Dimension (Active: {(kl_per_dim_mean > 0.1).sum()}/{config.LATENT_DIM})',
                fontsize=15, fontweight='bold', pad=15)
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_xticks(dims)
    ax.set_xticklabels(dims, fontsize=10)
    
    plt.tight_layout()
    
    if config.SAVE_FIGURES:
        fig.savefig(config.OUTPUT_DIR / 'kl_per_dimension.png', dpi=config.FIGURE_DPI, bbox_inches='tight')
    
    plt.close()
    print("✓ Saved: kl_per_dimension.png")
    
    # Print diagnostic info
    active_dims = (kl_per_dim_mean > 0.1).sum()
    print(f"\nLatent Space Diagnostics:")
    print(f" Active dimensions (KL > 0.1): {active_dims}/{config.LATENT_DIM}")
    print(f" Mean KL per dimension: {kl_per_dim_mean.mean():.4f}")
    print(f" Total KL: {kl_per_dim_mean.sum():.4f}")
    
    return kl_per_dim_mean


def plot_latent_space_semantic(latent_codes, semantic_features, feature_name, config):
    """Visualize latent space colored by a specific semantic feature"""
    print(f"Generating: Latent Space visualization for {feature_name}...")
    
    # Dimensionality reduction if needed
    if latent_codes.shape[1] > 2:
        if UMAP_AVAILABLE:
            reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
            latent_2d = reducer.fit_transform(latent_codes)
            method = 'UMAP'
        else:
            reducer = TSNE(n_components=2, random_state=42, perplexity=30)
            latent_2d = reducer.fit_transform(latent_codes)
            method = 't-SNE'
    else:
        latent_2d = latent_codes
        method = 'Native 2D'
    
    # Get feature values for coloring
    feat_idx = config.SEMANTIC_FEATURES.index(feature_name)
    feature_vals = semantic_features[:, feat_idx]
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    scatter = ax.scatter(latent_2d[:, 0], latent_2d[:, 1],
                        c=feature_vals, cmap='viridis',
                        s=70, alpha=0.7, edgecolors='black', linewidth=0.8)
    
    ax.set_xlabel(f'Latent Dimension 1 ({method})', fontsize=13, fontweight='bold')
    ax.set_ylabel(f'Latent Dimension 2 ({method})', fontsize=13, fontweight='bold')
    ax.set_title(f'Latent Space Organization: {feature_name}', fontsize=15, fontweight='bold', pad=15)
    
    cbar = plt.colorbar(scatter, ax=ax, label=f'{feature_name} Value', pad=0.02)
    cbar.ax.tick_params(labelsize=11)
    
    ax.grid(alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    if config.SAVE_FIGURES:
        fig.savefig(config.OUTPUT_DIR / f'latent_space_{feature_name}.png',
                   dpi=config.FIGURE_DPI, bbox_inches='tight')
    
    plt.close()
    print(f"✓ Saved: latent_space_{feature_name}.png")


def plot_rsa_results(rsa_results, config):
    """Visualize RSA results by feature"""
    features = [f for f in config.SEMANTIC_FEATURES if f in rsa_results]
    r_values = [rsa_results[f]['r'] for f in features]
    p_values = [rsa_results[f]['p'] for f in features]
    
    colors = ['red' if p < 0.05 else 'gray' for p in p_values]
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    bars = ax.bar(features, r_values, color=colors, alpha=0.75, edgecolor='black', linewidth=1.2)
    
    ax.axhline(0, color='black', linestyle='-', linewidth=1)
    ax.set_ylabel('RSA Correlation (r)', fontsize=13, fontweight='bold')
    ax.set_title('Semantic Feature-Brain Correlation', fontsize=15, fontweight='bold', pad=15)
    ax.set_ylim([-0.025, 0.025])
    
    plt.xticks(rotation=45, ha='right', fontsize=11)
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    if config.SAVE_FIGURES:
        fig.savefig(config.OUTPUT_DIR / 'rsa_by_feature.png', dpi=config.FIGURE_DPI, bbox_inches='tight')
    
    plt.close()
    print("✓ Saved: rsa_by_feature.png")


def plot_contrast_comparison(contrast_results, rsa_results, config):
    """Visualize how well generated contrasts match real brain patterns"""
    print("Generating: Contrast Comparison plot...")
    
    features = [f for f in contrast_results.keys() if contrast_results[f] is not None]
    
    # Extract correlations
    beta_corrs = [contrast_results[f]['beta_corr'] for f in features]
    t_corrs = [contrast_results[f]['t_corr'] for f in features]
    rsa_rs = [rsa_results[f]['r'] for f in features if f in rsa_results]
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Panel 1: Real-Gen correlation by feature
    ax1 = axes[0]
    x = np.arange(len(features))
    width = 0.35
    
    ax1.bar(x - width/2, beta_corrs, width, label='Beta Maps', alpha=0.85, color='steelblue', edgecolor='black')
    ax1.bar(x + width/2, t_corrs, width, label='T-stat Maps', alpha=0.85, color='coral', edgecolor='black')
    
    ax1.set_xlabel('Semantic Feature', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Real-Generated Correlation', fontsize=12, fontweight='bold')
    ax1.set_title('GLM Contrast Fidelity:\nReal vs. Generated Brain Patterns', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(features, rotation=45, ha='right', fontsize=11)
    ax1.legend(fontsize=11)
    ax1.axhline(0, color='black', linestyle='-', linewidth=1)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Panel 2: RSA correlation vs GLM fidelity
    ax2 = axes[1]
    
    if len(rsa_rs) == len(t_corrs):
        ax2.scatter(rsa_rs, t_corrs, s=150, alpha=0.7, color='purple', edgecolors='black', linewidth=1.5)
        
        for i, feat in enumerate(features[:len(rsa_rs)]):
            ax2.annotate(feat, (rsa_rs[i], t_corrs[i]), fontsize=10, alpha=0.8,
                        xytext=(5, 5), textcoords='offset points')
        
        # Fit line
        if len(rsa_rs) > 2:
            z = np.polyfit(rsa_rs, t_corrs, 1)
            p = np.poly1d(z)
            x_line = np.linspace(min(rsa_rs), max(rsa_rs), 100)
            ax2.plot(x_line, p(x_line), "r--", alpha=0.6, linewidth=2.5,
                    label=f'Fit: y={z[0]:.2f}x+{z[1]:.2f}')
            ax2.legend(fontsize=11)
        
        ax2.set_xlabel('RSA Correlation\n(Latent-Semantic)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('GLM T-stat Correlation\n(Real-Generated)', fontsize=12, fontweight='bold')
        ax2.set_title('Representational Strength vs.\nGenerative Fidelity', fontsize=14, fontweight='bold')
        ax2.grid(alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    if config.SAVE_FIGURES:
        fig.savefig(config.OUTPUT_DIR / 'contrast_comparison.png', dpi=config.FIGURE_DPI, bbox_inches='tight')
    
    plt.close()
    print("✓ Saved: contrast_comparison.png")
