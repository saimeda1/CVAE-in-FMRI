"""
Main Pipeline - Orchestrates all modules for LDT fMRI Semantic CVAE Analysis

Usage:
    python main.py [SUBJECT_ID]
    python main.py D101
    python main.py D104
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import warnings

warnings.filterwarnings('ignore')

from config import Config
from data_loader import DataLoader_LDT, LDT_Dataset
from models import Unconditional_VAE, Conditional_VAE
from training import train_vae
from analysis import (
    run_pca_baseline, run_ica_baseline, run_glm_baseline,
    extract_latent, rsa_analysis, run_all_glm_contrasts,
    compare_models
)
from visualizations import (
    plot_rsa_results, plot_vae_tradeoff, plot_training_dynamics,
    plot_kl_per_dimension, plot_latent_space_semantic, plot_contrast_comparison
)

print("✓ All imports successful\n")


def main():
    """Main LDT fMRI Semantic CVAE Pipeline"""
    
    config = Config()
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print("LDT fMRI - ENHANCED SEMANTIC CVAE PIPELINE")
    print("="*70 + "\n")
    
    try:
        # ====================================================================
        # LOAD DATA
        # ====================================================================
        
        loader = DataLoader_LDT(config)
        fmri_data, mask, func_4d, func_img = loader.load_fmri()
        behav_df = loader.load_behavioral()
        trials = loader.create_trials(behav_df)
        
        if len(trials) < 30:
            print(f"❌ ERROR: Only {len(trials)} trials!")
            return
        
        # ====================================================================
        # CREATE DATASET & DATALOADERS
        # ====================================================================
        
        print(f"\n{'='*70}\nCREATING DATASET\n{'='*70}\n")
        
        dataset = LDT_Dataset(fmri_data, trials)
        n_total = len(dataset)
        n_train = int(config.TRAIN_SPLIT * n_total)
        n_val = int(config.VAL_SPLIT * n_total)
        n_test = n_total - n_train - n_val
        
        train_ds, val_ds, test_ds = torch.utils.data.random_split(
            dataset, [n_train, n_val, n_test],
            generator=torch.Generator().manual_seed(42)
        )
        
        train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE, shuffle=False)
        full_loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=False)
        
        print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}\n")
        
        # ====================================================================
        # RUN BASELINE MODELS
        # ====================================================================
        
        baselines = {}
        
        if config.RUN_PCA:
            baselines['pca'] = run_pca_baseline(fmri_data, trials, config)
        
        if config.RUN_ICA:
            baselines['ica'] = run_ica_baseline(fmri_data, trials, config)
        
        if config.RUN_GLM:
            baselines['glm'] = run_glm_baseline(fmri_data, trials, config)
        
        if config.RUN_UNCONDITIONAL_VAE:
            print(f"\n{'='*70}\nTRAINING: UNCONDITIONAL VAE\n{'='*70}\n")
            
            model_u = Unconditional_VAE(fmri_data.shape[1], config.LATENT_DIM)
            model_u, hist_u = train_vae(model_u, train_loader, val_loader, config, is_conditional=False)
            
            latent_u, _ = extract_latent(model_u, full_loader, config, is_conditional=False)
            semantic_all = np.array([trials[i]['semantic_features'] for i in range(len(trials))])
            
            _, _, rsa_u = rsa_analysis(latent_u, semantic_all, config)
            baselines['unconditional'] = {'rsa_r': rsa_u['overall_semantic']['r']}
        
        # ====================================================================
        # TRAIN CONDITIONAL VAE (MAIN MODEL)
        # ====================================================================
        
        print(f"\n{'='*70}\nTRAINING: CONDITIONAL VAE\n{'='*70}\n")
        
        model_cvae = Conditional_VAE(fmri_data.shape[1], config.SEMANTIC_DIM, config.LATENT_DIM)
        model_cvae, hist_cvae = train_vae(model_cvae, train_loader, val_loader, config, is_conditional=True)
        
        # ====================================================================
        # EXTRACT & ANALYZE LATENT CODES
        # ====================================================================
        
        latent_cvae, semantic_cvae = extract_latent(model_cvae, full_loader, config, is_conditional=True)
        brain_rdm, semantic_rdm, rsa_cvae = rsa_analysis(latent_cvae, semantic_cvae, config)
        
        # ====================================================================
        # GENERATE VISUALIZATIONS
        # ====================================================================
        
        print(f"\n{'='*70}\nGENERATING VISUALIZATIONS\n{'='*70}")
        
        # 1. BCE vs KL Trade-Off
        plot_vae_tradeoff(hist_cvae, config)
        
        # 2. Training Dynamics (3-panel)
        plot_training_dynamics(hist_cvae, config)
        
        # 3. KL per Dimension
        kl_per_dim = plot_kl_per_dimension(model_cvae, full_loader, config)
        
        # 4. Latent Space Semantic Coloring (for significant features)
        for feature in ['VAL', 'DOM', 'AROU']:
            plot_latent_space_semantic(latent_cvae, semantic_cvae, feature, config)
        
        # 5. GLM Contrasts
        if config.RUN_GLM_CONTRASTS:
            contrast_results = run_all_glm_contrasts(
                fmri_data, trials, model_cvae, config, mask, func_img
            )
            
            # Visualize contrast comparisons
            plot_contrast_comparison(contrast_results, rsa_cvae, config)
        
        # 6. RSA by Feature
        plot_rsa_results(rsa_cvae, config)
        
        # ====================================================================
        # COMPARE MODELS
        # ====================================================================
        
        df_comp = compare_models(baselines, rsa_cvae, config)
        
        # ====================================================================
        # SAVE RESULTS
        # ====================================================================
        
        print(f"\n{'='*70}\nSAVING RESULTS\n{'='*70}\n")
        
        np.save(config.OUTPUT_DIR / 'latent_cvae.npy', latent_cvae)
        np.save(config.OUTPUT_DIR / 'brain_rdm.npy', brain_rdm)
        
        df_comp.to_csv(config.OUTPUT_DIR / 'model_comparison.csv', index=False)
        
        rsa_df = pd.DataFrame([
            {'feature': feat, 'r': data['r'], 'p': data['p']}
            for feat, data in rsa_cvae.items() if feat != 'overall_semantic'
        ])
        
        rsa_df.to_csv(config.OUTPUT_DIR / 'rsa_results.csv', index=False)
        
        # ====================================================================
        # COMPLETION MESSAGE
        # ====================================================================
        
        print(f"\n{'='*70}\n✓ PIPELINE COMPLETE!\n{'='*70}\n")
        print(f"Results: {config.OUTPUT_DIR}\n")
        
        print("\n" + "="*70)
        print("GENERATED FILES:")
        print("="*70)
        
        print("\nGLM Contrast Maps (NIfTI):")
        for feature in config.GLM_TARGET_FEATURES:
            print(f" glm_real_{feature}_beta.nii.gz & glm_real_{feature}_t.nii.gz")
            print(f" glm_cvae_{feature}_beta.nii.gz & glm_cvae_{feature}_t.nii.gz")
        
        print("\nVisualization Plots:")
        print(" vae_tradeoff.png (BCE vs KL optimization)")
        print(" training_dynamics.png (3-panel training curves)")
        print(" kl_per_dimension.png (latent space diagnostics)")
        print(" latent_space_VAL.png, latent_space_DOM.png, latent_space_AROU.png")
        print(" contrast_comparison.png (real vs generated GLM)")
        print(" rsa_by_feature.png")
        
        print("\nAnalysis Files:")
        print(" model_comparison.csv, rsa_results.csv, contrast_summary.csv")
        print(" latent_cvae.npy, brain_rdm.npy")
        
        print("\n✓ Use these files for your paper figures and AFNI visualization!\n")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
