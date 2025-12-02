"""
Analysis Module - RSA, baselines, GLM contrasts, visualizations
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr, pearsonr, ttest_ind
from sklearn.decomposition import PCA, FastICA
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
import nibabel as nib

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False


# ============================================================================
# BASELINE MODELS
# ============================================================================

def run_pca_baseline(fmri_data, trials, config):
    """PCA baseline"""
    print(f"\n{'='*70}\nBASELINE: PCA\n{'='*70}\n")
    
    patterns = np.array([fmri_data[np.clip(t['fmri_tr'], 0, len(fmri_data)-1)] for t in trials])
    semantic = np.array([t['semantic_features'] for t in trials])
    
    pca = PCA(n_components=config.LATENT_DIM)
    latent = pca.fit_transform(patterns)
    
    print(f"✓ Variance explained: {pca.explained_variance_ratio_.sum():.2%}")
    
    brain_rdm = squareform(pdist(latent, metric='correlation'))
    semantic_rdm = squareform(pdist(semantic, metric='euclidean'))
    
    mask = np.triu_indices_from(brain_rdm, k=1)
    r, p = spearmanr(brain_rdm[mask], semantic_rdm[mask])
    
    print(f"✓ RSA: r={r:.3f}, p={p:.3e}\n")
    
    return {'latent_codes': latent, 'rsa_r': r, 'rsa_p': p}


def run_ica_baseline(fmri_data, trials, config):
    """ICA baseline"""
    print(f"\n{'='*70}\nBASELINE: ICA\n{'='*70}\n")
    
    patterns = np.array([fmri_data[np.clip(t['fmri_tr'], 0, len(fmri_data)-1)] for t in trials])
    semantic = np.array([t['semantic_features'] for t in trials])
    
    ica = FastICA(n_components=config.LATENT_DIM, random_state=42, max_iter=500)
    latent = ica.fit_transform(patterns)
    
    brain_rdm = squareform(pdist(latent, metric='correlation'))
    semantic_rdm = squareform(pdist(semantic, metric='euclidean'))
    
    mask = np.triu_indices_from(brain_rdm, k=1)
    r, p = spearmanr(brain_rdm[mask], semantic_rdm[mask])
    
    print(f"✓ RSA: r={r:.3f}, p={p:.3e}\n")
    
    return {'latent_codes': latent, 'rsa_r': r, 'rsa_p': p}


def run_glm_baseline(fmri_data, trials, config):
    """GLM (Ridge Regression) baseline"""
    print(f"\n{'='*70}\nBASELINE: GLM (Ridge Regression)\n{'='*70}\n")
    
    X = np.array([t['semantic_features'] for t in trials])
    Y = np.array([fmri_data[np.clip(t['fmri_tr'], 0, len(fmri_data)-1)] for t in trials])
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = [r2_score(Y[test], Ridge(alpha=1.0).fit(X[train], Y[train]).predict(X[test]), multioutput='variance_weighted')
              for train, test in kf.split(X)]
    
    mean_r2 = np.mean(scores)
    
    print(f"✓ CV R²: {mean_r2:.3f}\n")
    
    glm = Ridge(alpha=1.0)
    glm.fit(X, Y)
    
    return {'r2': mean_r2, 'model': glm}


# ============================================================================
# LATENT CODE EXTRACTION
# ============================================================================

def extract_latent(model, dataloader, config, is_conditional=True):
    """Extract latent codes from model"""
    model.eval()
    model = model.to(config.DEVICE)
    
    latent_codes, semantic_features = [], []
    
    with torch.no_grad():
        for batch_fmri, batch_semantic in dataloader:
            batch_fmri = batch_fmri.to(config.DEVICE)
            batch_semantic = batch_semantic.to(config.DEVICE)
            
            if is_conditional:
                mu, _ = model.encode(batch_fmri, batch_semantic)
                semantic_features.append(batch_semantic.cpu().numpy())
            else:
                mu, _ = model.encode(batch_fmri)
            
            latent_codes.append(mu.cpu().numpy())
    
    return np.vstack(latent_codes), np.vstack(semantic_features) if semantic_features else None


# ============================================================================
# RSA ANALYSIS
# ============================================================================

def rsa_analysis(latent_codes, semantic_features, config):
    """RSA ANALYSIS"""
    print(f"\n{'='*70}\nRSA ANALYSIS\n{'='*70}\n")
    
    brain_rdm = squareform(pdist(latent_codes, metric='correlation'))
    semantic_rdm = squareform(pdist(semantic_features, metric='euclidean'))
    
    mask = np.triu_indices_from(brain_rdm, k=1)
    r_overall, p_overall = spearmanr(brain_rdm[mask], semantic_rdm[mask])
    
    print(f"Overall: r={r_overall:.4f}, p={p_overall:.3e}\n")
    
    rsa_results = {'overall_semantic': {'r': r_overall, 'p': p_overall}}
    
    for i, feat in enumerate(config.SEMANTIC_FEATURES):
        feature_rdm = np.abs(np.subtract.outer(semantic_features[:, i], semantic_features[:, i]))
        r, p = spearmanr(brain_rdm[mask], feature_rdm[mask])
        rsa_results[feat] = {'r': r, 'p': p}
        
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f" {feat:10s}: r={r:7.4f}, p={p:.3e} {sig}")
    
    return brain_rdm, semantic_rdm, rsa_results


# ============================================================================
# GLM CONTRASTS
# ============================================================================

def glm_semantic_contrast_real(fmri_data, trials, feature_name, config):
    """GLM on REAL data"""
    print(f"\nGLM CONTRAST (REAL): {feature_name}")
    
    patterns = []
    feature_vals = []
    feat_idx = config.SEMANTIC_FEATURES.index(feature_name)
    
    for tr in trials:
        tr_idx = int(np.clip(tr['fmri_tr'], 0, fmri_data.shape[0]-1))
        patterns.append(fmri_data[tr_idx])
        feature_vals.append(tr['semantic_features'][feat_idx])
    
    Y = np.vstack(patterns)
    x = np.array(feature_vals)
    X = np.column_stack([np.ones(len(x)), x])
    
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ (X.T @ Y)
    beta_1 = beta[1]
    
    Y_hat = X @ beta
    resid = Y - Y_hat
    dof = X.shape[0] - X.shape[1]
    sigma2 = (resid**2).sum(axis=0) / max(dof, 1)
    
    var_beta = sigma2 * XtX_inv[1, 1]
    se_beta = np.sqrt(var_beta + 1e-12)
    t_vals = beta_1 / se_beta
    
    print(f" Mean beta: {beta_1.mean():.4f}")
    print(f" Mean t: {t_vals.mean():.4f}")
    
    return beta_1, t_vals


def glm_semantic_contrast_generated(model_cvae, feature_name, config, n_points=21):
    """GLM on GENERATED data"""
    print(f"\nGLM CONTRAST (GENERATED): {feature_name}")
    
    model_cvae.eval()
    device = config.DEVICE
    feat_idx = config.SEMANTIC_FEATURES.index(feature_name)
    
    feature_values = np.linspace(-2, 2, n_points).astype(np.float32)
    semantics = np.zeros((n_points, config.SEMANTIC_DIM), dtype=np.float32)
    semantics[:, feat_idx] = feature_values
    
    with torch.no_grad():
        z = torch.randn(n_points, config.LATENT_DIM, device=device)
        c = torch.from_numpy(semantics).to(device)
        recon = model_cvae.decode(z, c)
    
    Y = recon.cpu().numpy()
    x = feature_values
    X = np.column_stack([np.ones(len(x)), x])
    
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ (X.T @ Y)
    beta_1 = beta[1]
    
    Y_hat = X @ beta
    resid = Y - Y_hat
    dof = X.shape[0] - X.shape[1]
    sigma2 = (resid**2).sum(axis=0) / max(dof, 1)
    
    var_beta = sigma2 * XtX_inv[1, 1]
    se_beta = np.sqrt(var_beta + 1e-12)
    t_vals = beta_1 / se_beta
    
    print(f" Mean beta: {beta_1.mean():.4f}")
    print(f" Mean t: {t_vals.mean():.4f}")
    
    return beta_1, t_vals


def save_contrast_map(beta_vector, mask_3d, reference_img, out_path):
    """Save 1D contrast vector as 3D NIfTI"""
    ref_shape = reference_img.shape[:3]
    vol = np.zeros(ref_shape, dtype=np.float32)
    vol[mask_3d] = beta_vector
    
    nii = nib.Nifti1Image(vol, reference_img.affine)
    nib.save(nii, str(out_path))
    
    print(f"✓ Saved: {out_path.name}")


def run_all_glm_contrasts(fmri_data, trials, model_cvae, config, mask, func_img):
    """Generate GLM contrasts for ALL significant semantic features"""
    print(f"\n{'='*70}\nGLM SEMANTIC CONTRASTS - ALL FEATURES\n{'='*70}")
    
    contrast_results = {}
    
    for feature_name in config.GLM_TARGET_FEATURES:
        print(f"\n{'-'*70}")
        print(f"Processing: {feature_name}")
        print(f"{'-'*70}")
        
        try:
            # Real data GLM
            beta_real, t_real = glm_semantic_contrast_real(fmri_data, trials, feature_name, config)
            
            # Save real contrasts
            save_contrast_map(beta_real, mask, func_img,
                            config.OUTPUT_DIR / f'glm_real_{feature_name}_beta.nii.gz')
            save_contrast_map(t_real, mask, func_img,
                            config.OUTPUT_DIR / f'glm_real_{feature_name}_t.nii.gz')
            
            # Generated data GLM
            beta_gen, t_gen = glm_semantic_contrast_generated(model_cvae, feature_name, config)
            
            # Save generated contrasts
            save_contrast_map(beta_gen, mask, func_img,
                            config.OUTPUT_DIR / f'glm_cvae_{feature_name}_beta.nii.gz')
            save_contrast_map(t_gen, mask, func_img,
                            config.OUTPUT_DIR / f'glm_cvae_{feature_name}_t.nii.gz')
            
            # Calculate correlation between real and generated maps
            corr_beta = np.corrcoef(beta_real, beta_gen)[0, 1]
            corr_t = np.corrcoef(t_real, t_gen)[0, 1]
            
            contrast_results[feature_name] = {
                'beta_corr': corr_beta,
                't_corr': corr_t,
                'mean_beta_real': beta_real.mean(),
                'mean_beta_gen': beta_gen.mean(),
                'mean_t_real': t_real.mean(),
                'mean_t_gen': t_gen.mean()
            }
            
            print(f" Real-Gen Beta Correlation: {corr_beta:.4f}")
            print(f" Real-Gen T-stat Correlation: {corr_t:.4f}")
            
        except Exception as e:
            print(f" ⚠️ Error processing {feature_name}: {e}")
            contrast_results[feature_name] = None
    
    # Save summary
    contrast_df = pd.DataFrame(contrast_results).T
    contrast_df.to_csv(config.OUTPUT_DIR / 'contrast_summary.csv')
    
    print(f"\n✓ Saved: contrast_summary.csv")
    
    return contrast_results


# ============================================================================
# MODEL COMPARISON
# ============================================================================

def compare_models(baselines, cvae_rsa, config):
    """Compare all models"""
    print(f"\n{'='*70}\nMODEL COMPARISON\n{'='*70}\n")
    
    comparison = pd.DataFrame({
        'Model': ['PCA', 'ICA', 'Unconditional VAE', 'Conditional VAE'],
        'RSA_r': [
            baselines.get('pca', {}).get('rsa_r', np.nan),
            baselines.get('ica', {}).get('rsa_r', np.nan),
            baselines.get('unconditional', {}).get('rsa_r', np.nan),
            cvae_rsa['overall_semantic']['r']
        ]
    })
    
    print(comparison.to_string(index=False))
    print()
    
    return comparison
