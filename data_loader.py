"""
Data Loading Module - fMRI and behavioral data loading with AFNI masking
"""

import numpy as np
import pandas as pd
import torch
import nibabel as nib
from torch.utils.data import Dataset

from utils import parse_list_value, find_afni_mask, load_afni_mask, create_python_mask


class DataLoader_LDT:
    """Load LDT data with AFNI masking"""
    
    def __init__(self, config):
        self.config = config
    
    def load_fmri(self):
        """Load fMRI with AFNI mask - VECTORIZED"""
        print(f"\n{'='*70}\nLOADING fMRI DATA: {self.config.SUBJECT_ID}\n{'='*70}\n")
        
        func_file = self.config.SUBJECT_DIR / self.config.FUNC_FILE
        
        if not func_file.exists():
            raise FileNotFoundError(f"fMRI file not found: {func_file}")
        
        print(f"✓ Found: {func_file.name}")
        
        func_img = nib.load(str(func_file))
        func_4d = func_img.get_fdata()
        
        print(f" Shape: {func_4d.shape} (X, Y, Z, Time)\n")
        
        # Load mask
        afni_mask_path = find_afni_mask(self.config.SUBJECT_DIR, self.config.MASK_PATTERNS)
        
        if afni_mask_path:
            mask = load_afni_mask(afni_mask_path)
            print(f" AFNI mask voxels: {mask.sum():,}\n")
        else:
            mask = create_python_mask(func_4d)
            print(f" Python mask voxels: {mask.sum():,}\n")
        
        # Vectorized masking
        print(f"Applying mask (vectorized)...")
        X, Y, Z, T = func_4d.shape
        func_reshaped = func_4d.reshape(-1, T)
        mask_flat = mask.flatten()
        func_masked = func_reshaped[mask_flat, :].T
        
        print(f" Extracted: {func_masked.shape}\n")
        
        # Standardize
        print(f"Standardizing per voxel...")
        mean = func_masked.mean(axis=0, keepdims=True)
        std = func_masked.std(axis=0, keepdims=True)
        std[std < 1e-10] = 1.0
        func_2d = (func_masked - mean) / std
        
        print(f" Mean: {func_2d.mean():.8f}")
        print(f" Std: {func_2d.std():.8f}")
        print(f" Final shape: {func_2d.shape}\n")
        
        return func_2d, mask, func_4d, func_img
    
    def load_behavioral(self):
        """Load behavioral CSV"""
        print(f"{'='*70}\nLOADING BEHAVIORAL DATA\n{'='*70}\n")
        
        csv_file = self.config.SUBJECT_DIR / self.config.BEHAVIORAL_CSV
        
        if not csv_file.exists():
            raise FileNotFoundError(f"CSV not found: {csv_file}")
        
        df = pd.read_csv(csv_file, encoding='utf-8-sig')
        df_trials = df[(df['isPractice'] == 0) & (df['null_trial'] == False)].copy()
        
        print(f"✓ Loaded {len(df_trials)} real trials\n")
        
        return df_trials
    
    def create_trials(self, behav_df):
        """Create trial structure for REAL WORDS ONLY"""
        print(f"{'='*70}\nCREATING TRIAL STRUCTURE - REAL WORDS ONLY\n{'='*70}\n")
        
        trials = []
        
        for idx, row in behav_df.iterrows():
            # Only real words (trialType == 1)
            if row['trialType'] != 1:
                continue
            
            word = str(row.get('string', '')).strip()
            
            if not word or pd.isna(word):
                continue
            
            # Extract semantic features
            semantic_features = []
            has_all = True
            
            for feat in self.config.SEMANTIC_FEATURES:
                val = parse_list_value(row.get(feat, np.nan))
                
                if pd.isna(val):
                    has_all = False
                    break
                
                semantic_features.append(val)
            
            if not has_all:
                continue
            
            # Compute fMRI TR (with HRF peak delay)
            word_onset = row['word_on']
            tr_float = (word_onset + self.config.HRF_PEAK_DELAY) / self.config.TR
            fmri_tr = int(np.round(tr_float))
            
            trial = {
                'idx': idx,
                'word': word,
                'semantic_features': np.array(semantic_features, dtype=np.float32),
                'fmri_tr': fmri_tr,
                'rt': parse_list_value(row.get('respTime_FromWord', np.nan)),
                'accuracy': parse_list_value(row.get('resp_corr', np.nan)),
            }
            
            trials.append(trial)
        
        print(f"✓ Created {len(trials)} word trials\n")
        
        return trials


class LDT_Dataset(Dataset):
    """PyTorch Dataset for training"""
    
    def __init__(self, fmri_data, trials):
        self.fmri_data = torch.FloatTensor(fmri_data)
        self.trials = trials
    
    def __len__(self):
        return len(self.trials)
    
    def __getitem__(self, idx):
        trial = self.trials[idx]
        
        # Clip TR to valid range
        fmri_tr = np.clip(trial['fmri_tr'], 0, len(self.fmri_data) - 1)
        fmri_pattern = self.fmri_data[fmri_tr]
        
        # Semantic features
        semantic_features = torch.FloatTensor(trial['semantic_features'])
        
        return fmri_pattern, semantic_features
