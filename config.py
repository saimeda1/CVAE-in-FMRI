"""
Configuration Module - LDT fMRI Semantic CVAE Pipeline
All hyperparameters, paths, and experiment settings in one place
"""

import sys
from pathlib import Path
import torch

class Config:
    """Configuration - Best performing settings"""
    
    SUBJECT_ID = sys.argv[1] if len(sys.argv) > 1 else 'D101'
    HOME = Path.home()
    DATA_ROOT = HOME / 'Desktop' / 'VAE_LDT'
    SUBJECT_DIR = DATA_ROOT / SUBJECT_ID
    OUTPUT_DIR = DATA_ROOT / 'results' / SUBJECT_ID / 'enhanced_cvae'
    
    # File patterns
    FUNC_FILE = f'all_runs.{SUBJECT_ID}+tlrc.BRIK'
    BEHAVIORAL_CSV = f'{SUBJECT_ID}(in).csv'
    
    # AFNI mask patterns (priority order)
    MASK_PATTERNS = [
        f'full_mask.{SUBJECT_ID}+tlrc.BRIK',
        f'{SUBJECT_ID}_brain_mask+tlrc.BRIK',
        f'mask_epi_anat.{SUBJECT_ID}+tlrc.BRIK',
        'full_mask+tlrc.BRIK',
    ]
    
    # Experiment parameters
    TR = 2.0
    HRF_PEAK_DELAY = 5.0
    
    # Semantic features
    SEMANTIC_FEATURES = ['AROU', 'VAL', 'DOM', 'CNC', 'IMAG', 'FAM', 'AOA', 'SIZE', 'GEND', 'MANI', 'ANIM']
    SEMANTIC_DIM = len(SEMANTIC_FEATURES)
    
    # Model hyperparameters - BEST PERFORMING
    LATENT_DIM = 20
    BATCH_SIZE = 32
    EPOCHS = 80
    LEARNING_RATE = 1e-3
    BETA = 1e-3
    
    # Baseline comparisons (full suite)
    RUN_PCA = True
    RUN_ICA = True
    RUN_GLM = True
    RUN_UNCONDITIONAL_VAE = True
    
    # Generative experiments
    RUN_GENERATION = True
    RUN_GLM_CONTRASTS = True
    N_GENERATION_SAMPLES = 50
    
    # Features for extended GLM analysis
    GLM_TARGET_FEATURES = ['VAL', 'DOM', 'AROU', 'AOA', 'GEND', 'CNC', 'IMAG', 'FAM']
    
    # Splits
    TRAIN_SPLIT = 0.6
    VAL_SPLIT = 0.2
    TEST_SPLIT = 0.2
    
    # Device
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Visualization
    SAVE_FIGURES = True
    FIGURE_DPI = 300
