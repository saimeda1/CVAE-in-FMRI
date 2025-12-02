"""
Utility Functions - AFNI masking, data parsing, preprocessing
"""

import numpy as np
import pandas as pd
import nibabel as nib
from scipy.ndimage import label
import ast

def parse_list_value(val):
    """Parse PsychoPy list format"""
    if pd.isna(val):
        return np.nan
    
    val_str = str(val).strip()
    
    if val_str.startswith('[') and val_str.endswith(']'):
        try:
            parsed = ast.literal_eval(val_str)
            if isinstance(parsed, (list, tuple)) and len(parsed) > 0:
                return parsed[0]
            return parsed
        except:
            return np.nan
    
    try:
        return float(val_str)
    except:
        return np.nan


def find_afni_mask(subject_dir, mask_patterns):
    """Find AFNI-generated brain mask"""
    print("Searching for AFNI brain mask...")
    
    for pattern in mask_patterns:
        mask_path = subject_dir / pattern
        if mask_path.exists():
            print(f" ✓ Found AFNI mask: {mask_path.name}")
            return mask_path
    
    print(" ⚠️ No AFNI mask found. Will use fallback Python masking.")
    return None


def load_afni_mask(mask_path):
    """Load AFNI mask file and squeeze 4D→3D if needed"""
    print(f"Loading AFNI mask: {mask_path.name}")
    
    mask_img = nib.load(str(mask_path))
    mask_data = mask_img.get_fdata().astype(bool)
    
    # Handle AFNI 4D masks (singleton time dimension)
    if mask_data.ndim == 4:
        print(f" Original shape: {mask_data.shape} (4D)")
        mask_data = np.squeeze(mask_data, axis=3)
        print(f" Squeezed to: {mask_data.shape}")
    
    if mask_data.ndim != 3:
        raise ValueError(f"Unexpected mask shape: {mask_data.shape}")
    
    return mask_data


def create_python_mask(func_4d, percentile_threshold=20):
    """Fallback Python-side masking"""
    print(f"Creating Python mask (percentile={percentile_threshold})...")
    
    mean_vol = func_4d.mean(axis=3)
    brain_like = mean_vol[mean_vol > 0]
    threshold = np.percentile(brain_like, percentile_threshold)
    mask = mean_vol > threshold
    
    # Keep largest component
    labeled_array, num_features = label(mask)
    
    if num_features > 0:
        sizes = np.bincount(labeled_array.ravel())
        sizes[0] = 0
        largest_component = np.argmax(sizes)
        mask = labeled_array == largest_component
    
    return mask
