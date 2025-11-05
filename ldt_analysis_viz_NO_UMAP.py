"""
LDT MANIFOLD ANALYSIS - POST-PROCESSING & VISUALIZATION
Manifold Visualization + Behavioral Prediction

CORRECTED FOR YOUR FILE STRUCTURE:
~/Desktop/VAE_LDT/
├── D101/  (data folder)
├── results/
│   └── D101/  (results folder - this is where files are)
│       ├── latent_codes.npy
│       ├── labels.npy
│       ├── trials.pkl
│       ├── training_history.csv
│       ├── brain_rdm.npy
│       ├── rsa_results.csv
│       └── figures/  (output goes here)

FIXED: Correct UMAP import from umap package (not sklearn)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Try to import UMAP, but make it optional
try:
    from umap import UMAP
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("⚠️  UMAP not installed. Skipping UMAP visualization.")
    print("   Install with: pip install umap-learn")

# ============================================================================
# CONFIGURATION
# ============================================================================

SUBJECT_ID = 'D101'  # CHANGE THIS FOR EACH SUBJECT
HOME = Path.home()

# YOUR STRUCTURE: results are in Desktop/VAE_LDT/results/SUBJECT_ID/
RESULTS_DIR = HOME / 'Desktop' / 'VAE_LDT' / 'results' / SUBJECT_ID
FIGURES_DIR = RESULTS_DIR / 'figures'
FIGURES_DIR.mkdir(exist_ok=True, parents=True)

# Plotting
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11
DPI = 300

print(f"\nResults directory: {RESULTS_DIR}")
print(f"Figures directory: {FIGURES_DIR}")

# ============================================================================
# LOAD DATA
# ============================================================================

print(f"\n{'='*70}")
print(f"LOADING RESULTS FOR {SUBJECT_ID}")
print(f"{'='*70}\n")

# Check files exist
required_files = [
    'latent_codes.npy',
    'labels.npy', 
    'brain_rdm.npy',
    'training_history.csv',
    'trials.pkl'
]

missing_files = []
for file in required_files:
    if not (RESULTS_DIR / file).exists():
        missing_files.append(file)

if missing_files:
    print(f"❌ ERROR: Missing files:")
    for f in missing_files:
        print(f"  - {RESULTS_DIR / f}")
    exit(1)

# Load main outputs
latent_codes = np.load(RESULTS_DIR / 'latent_codes.npy')
labels = np.load(RESULTS_DIR / 'labels.npy')
brain_rdm = np.load(RESULTS_DIR / 'brain_rdm.npy')
training_history = pd.read_csv(RESULTS_DIR / 'training_history.csv', index_col=0)

with open(RESULTS_DIR / 'trials.pkl', 'rb') as f:
    trials = pickle.load(f)

print(f"✓ Loaded data:")
print(f"  Latent codes: {latent_codes.shape}")
print(f"  Labels: {labels.shape}")
print(f"  Brain RDM: {brain_rdm.shape}")
print(f"  Trials: {len(trials)}")

# Extract behavioral data from trials
rt_values = np.array([trial['rt'] for trial in trials])
accuracy_values = np.array([trial['accuracy'] for trial in trials])

# Remove NaN RTs
valid_rt_mask = ~np.isnan(rt_values)
rt_valid = rt_values[valid_rt_mask]
latent_valid_rt = latent_codes[valid_rt_mask]

print(f"\n  Valid RTs: {len(rt_valid)}/{len(rt_values)}")
print(f"  RT range: {rt_valid.min():.3f} - {rt_valid.max():.3f} sec")
print(f"  Accuracy: {accuracy_values.mean()*100:.1f}%")

# Condition mapping
condition_names = {1: 'Word', 2: 'Pseudoword', 3: 'NoGo'}
condition_colors = {1: '#2E86AB', 2: '#A23B72', 3: '#F18F01'}  # Blue, Purple, Orange

# ============================================================================
# 1. TRAINING CONVERGENCE
# ============================================================================

print(f"\n{'='*70}")
print(f"1. TRAINING CONVERGENCE")
print(f"{'='*70}\n")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Loss curves
axes[0].plot(training_history.index, training_history['train_loss'], 
             label='Train', linewidth=2, color='#2E86AB')
axes[0].plot(training_history.index, training_history['val_loss'], 
             label='Val', linewidth=2, color='#A23B72')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss (normalized)')
axes[0].set_title('VAE Loss Convergence')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Reconstruction vs KL
axes[1].plot(training_history.index, training_history['train_recon'], 
             label='Train Recon', linewidth=2, color='#2E86AB', linestyle='-')
axes[1].plot(training_history.index, training_history['train_kl'], 
             label='Train KL', linewidth=2, color='#F18F01', linestyle='-')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss (normalized)')
axes[1].set_title('Reconstruction vs KL Divergence')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(FIGURES_DIR / '01_training_convergence.png', dpi=DPI, bbox_inches='tight')
print(f"✓ Saved: 01_training_convergence.png")
plt.close()

# ============================================================================
# 2. MANIFOLD VISUALIZATION - PCA
# ============================================================================

print(f"\n{'='*70}")
print(f"2. MANIFOLD VISUALIZATION - PCA")
print(f"{'='*70}\n")

pca = PCA(n_components=3)
latent_pca = pca.fit_transform(latent_codes)

print(f"PCA variance explained: {pca.explained_variance_ratio_[:3].sum():.1%}")

# 2D PCA scatter
fig, ax = plt.subplots(figsize=(10, 8))

for label in [1, 2]:
    mask = labels == label
    ax.scatter(latent_pca[mask, 0], latent_pca[mask, 1], 
               label=condition_names[label], 
               alpha=0.6, s=80,
               color=condition_colors[label],
               edgecolors='black', linewidth=0.5)

ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
ax.set_title(f'{SUBJECT_ID}: Latent Manifold - PCA')
ax.legend(fontsize=12, loc='best')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(FIGURES_DIR / '02_manifold_pca_2d.png', dpi=DPI, bbox_inches='tight')
print(f"✓ Saved: 02_manifold_pca_2d.png")
plt.close()

# 3D PCA scatter
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')

for label in [1, 2]:
    mask = labels == label
    ax.scatter(latent_pca[mask, 0], latent_pca[mask, 1], latent_pca[mask, 2],
               label=condition_names[label],
               alpha=0.6, s=80,
               color=condition_colors[label],
               edgecolors='black', linewidth=0.5)

ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
ax.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]:.1%})')
ax.set_title(f'{SUBJECT_ID}: Latent Manifold - PCA 3D')
ax.legend(fontsize=12)
plt.tight_layout()
plt.savefig(FIGURES_DIR / '03_manifold_pca_3d.png', dpi=DPI, bbox_inches='tight')
print(f"✓ Saved: 03_manifold_pca_3d.png")
plt.close()

# ============================================================================
# 3. MANIFOLD VISUALIZATION - UMAP (OPTIONAL)
# ============================================================================

if HAS_UMAP:
    print(f"\n{'='*70}")
    print(f"3. MANIFOLD VISUALIZATION - UMAP")
    print(f"{'='*70}\n")
    
    print("Computing UMAP (this takes ~30 seconds)...")
    umap_reducer = UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    latent_umap = umap_reducer.fit_transform(latent_codes)
    print("✓ UMAP complete")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for label in [1, 2]:
        mask = labels == label
        ax.scatter(latent_umap[mask, 0], latent_umap[mask, 1],
                   label=condition_names[label],
                   alpha=0.6, s=80,
                   color=condition_colors[label],
                   edgecolors='black', linewidth=0.5)
    
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.set_title(f'{SUBJECT_ID}: Latent Manifold - UMAP')
    ax.legend(fontsize=12, loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / '04_manifold_umap.png', dpi=DPI, bbox_inches='tight')
    print(f"✓ Saved: 04_manifold_umap.png")
    plt.close()
else:
    print(f"\n{'='*70}")
    print(f"3. SKIPPING UMAP (not installed)")
    print(f"{'='*70}\n")
    latent_umap = None

# ============================================================================
# 4. BEHAVIORAL PREDICTION - RT
# ============================================================================

print(f"\n{'='*70}")
print(f"4. BEHAVIORAL PREDICTION - REACTION TIME")
print(f"{'='*70}\n")

# Linear Regression
lr_model = LinearRegression()
lr_cv_scores = cross_val_score(lr_model, latent_valid_rt, rt_valid, 
                               cv=5, scoring='r2')
lr_model.fit(latent_valid_rt, rt_valid)
lr_pred = lr_model.predict(latent_valid_rt)
lr_r2 = r2_score(rt_valid, lr_pred)
lr_rmse = np.sqrt(mean_squared_error(rt_valid, lr_pred))

print(f"Linear Regression:")
print(f"  R² (full): {lr_r2:.4f}")
print(f"  R² (5-fold CV): {lr_cv_scores.mean():.4f} ± {lr_cv_scores.std():.4f}")
print(f"  RMSE: {lr_rmse:.4f} sec")

# Random Forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_cv_scores = cross_val_score(rf_model, latent_valid_rt, rt_valid,
                               cv=5, scoring='r2')
rf_model.fit(latent_valid_rt, rt_valid)
rf_pred = rf_model.predict(latent_valid_rt)
rf_r2 = r2_score(rt_valid, rf_pred)
rf_rmse = np.sqrt(mean_squared_error(rt_valid, rf_pred))

print(f"\nRandom Forest:")
print(f"  R² (full): {rf_r2:.4f}")
print(f"  R² (5-fold CV): {rf_cv_scores.mean():.4f} ± {rf_cv_scores.std():.4f}")
print(f"  RMSE: {rf_rmse:.4f} sec")

# Feature importance (Random Forest)
feature_importance = rf_model.feature_importances_
top_features = np.argsort(feature_importance)[-5:]

print(f"\nTop 5 Important Latent Dimensions (RF):")
for rank, dim in enumerate(top_features[::-1], 1):
    print(f"  {rank}. Dimension {dim}: {feature_importance[dim]:.4f}")

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Linear Regression scatter
axes[0].scatter(rt_valid, lr_pred, alpha=0.5, s=40, color='#2E86AB', edgecolors='black', linewidth=0.3)
axes[0].plot([rt_valid.min(), rt_valid.max()], [rt_valid.min(), rt_valid.max()], 
             'r--', linewidth=2, label='Perfect')
axes[0].set_xlabel('Actual RT (sec)')
axes[0].set_ylabel('Predicted RT (sec)')
axes[0].set_title(f'Linear Regression\nR² = {lr_r2:.4f}')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Random Forest scatter
axes[1].scatter(rt_valid, rf_pred, alpha=0.5, s=40, color='#A23B72', edgecolors='black', linewidth=0.3)
axes[1].plot([rt_valid.min(), rt_valid.max()], [rt_valid.min(), rt_valid.max()],
             'r--', linewidth=2, label='Perfect')
axes[1].set_xlabel('Actual RT (sec)')
axes[1].set_ylabel('Predicted RT (sec)')
axes[1].set_title(f'Random Forest\nR² = {rf_r2:.4f}')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(FIGURES_DIR / '05_rt_prediction.png', dpi=DPI, bbox_inches='tight')
print(f"\n✓ Saved: 05_rt_prediction.png")
plt.close()

# ============================================================================
# 5. BEHAVIORAL PREDICTION - ACCURACY
# ============================================================================

print(f"\n{'='*70}")
print(f"5. BEHAVIORAL PREDICTION - ACCURACY")
print(f"{'='*70}\n")

# Logistic Regression
lr_clf = LogisticRegression(random_state=42, max_iter=1000)
lr_clf_cv = cross_val_score(lr_clf, latent_codes, accuracy_values, cv=5, scoring='accuracy')
lr_clf.fit(latent_codes, accuracy_values)
lr_clf_pred = lr_clf.predict(latent_codes)
lr_clf_acc = accuracy_score(accuracy_values, lr_clf_pred)

print(f"Logistic Regression:")
print(f"  Accuracy (full): {lr_clf_acc:.4f}")
print(f"  Accuracy (5-fold CV): {lr_clf_cv.mean():.4f} ± {lr_clf_cv.std():.4f}")

# Random Forest Classifier
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_clf_cv = cross_val_score(rf_clf, latent_codes, accuracy_values, cv=5, scoring='accuracy')
rf_clf.fit(latent_codes, accuracy_values)
rf_clf_pred = rf_clf.predict(latent_codes)
rf_clf_acc = accuracy_score(accuracy_values, rf_clf_pred)

print(f"\nRandom Forest Classifier:")
print(f"  Accuracy (full): {rf_clf_acc:.4f}")
print(f"  Accuracy (5-fold CV): {rf_clf_cv.mean():.4f} ± {rf_clf_cv.std():.4f}")

# ============================================================================
# 6. SUMMARY STATISTICS
# ============================================================================

print(f"\n{'='*70}")
print(f"6. SUMMARY STATISTICS")
print(f"{'='*70}\n")

summary_df = pd.DataFrame({
    'Model': ['Linear Regression (RT)', 'Random Forest (RT)', 
              'Logistic Regression (Acc)', 'Random Forest (Acc)'],
    'Metric': [f'{lr_r2:.4f}', f'{rf_r2:.4f}', 
               f'{lr_clf_acc:.4f}', f'{rf_clf_acc:.4f}'],
    'CV_Mean': [f'{lr_cv_scores.mean():.4f}', f'{rf_cv_scores.mean():.4f}',
                f'{lr_clf_cv.mean():.4f}', f'{rf_clf_cv.mean():.4f}'],
    'CV_Std': [f'{lr_cv_scores.std():.4f}', f'{rf_cv_scores.std():.4f}',
               f'{lr_clf_cv.std():.4f}', f'{rf_clf_cv.std():.4f}']
})

print(summary_df.to_string(index=False))

# Save summary
summary_df.to_csv(RESULTS_DIR / 'prediction_summary.csv', index=False)
print(f"\n✓ Saved: prediction_summary.csv")

# ============================================================================
# 7. CONDITION SEPARATION ANALYSIS
# ============================================================================

print(f"\n{'='*70}")
print(f"7. CONDITION SEPARATION ANALYSIS")
print(f"{'='*70}\n")

# Compute center of mass for each condition
centers = {}
for label in [1, 2]:
    mask = labels == label
    centers[label] = latent_codes[mask].mean(axis=0)

# Distance between centers
center_dist = np.linalg.norm(centers[1] - centers[2])
print(f"Distance between condition centers: {center_dist:.4f}")

# Within-condition variance
within_var = {}
for label in [1, 2]:
    mask = labels == label
    within_var[label] = np.mean(np.linalg.norm(latent_codes[mask] - centers[label], axis=1))

print(f"Mean within-condition distance:")
print(f"  Word:       {within_var[1]:.4f}")
print(f"  Pseudoword: {within_var[2]:.4f}")

# Separation metric (between / within)
separation = center_dist / ((within_var[1] + within_var[2]) / 2)
print(f"\nSeparation Index: {separation:.4f}")
print(f"  (Higher = better separation)")

# ============================================================================
# 8. FINAL SUMMARY PLOT
# ============================================================================

print(f"\n{'='*70}")
print(f"8. CREATING SUMMARY REPORT")
print(f"{'='*70}\n")

fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Title
fig.suptitle(f'{SUBJECT_ID}: LDT Manifold Analysis Summary', fontsize=16, fontweight='bold')

# 1. PCA scatter
ax1 = fig.add_subplot(gs[0, 0])
for label in [1, 2]:
    mask = labels == label
    ax1.scatter(latent_pca[mask, 0], latent_pca[mask, 1], alpha=0.5, s=30,
                label=condition_names[label], color=condition_colors[label])
ax1.set_xlabel('PC1')
ax1.set_ylabel('PC2')
ax1.set_title('PCA Manifold')
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)

# 2. UMAP scatter (if available)
ax2 = fig.add_subplot(gs[0, 1])
if HAS_UMAP and latent_umap is not None:
    for label in [1, 2]:
        mask = labels == label
        ax2.scatter(latent_umap[mask, 0], latent_umap[mask, 1], alpha=0.5, s=30,
                    label=condition_names[label], color=condition_colors[label])
    ax2.set_xlabel('UMAP1')
    ax2.set_ylabel('UMAP2')
    ax2.set_title('UMAP Manifold')
else:
    ax2.text(0.5, 0.5, 'UMAP\nnot installed', ha='center', va='center', fontsize=12)
    ax2.set_title('UMAP Manifold (N/A)')
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)

# 3. Training loss
ax3 = fig.add_subplot(gs[0, 2])
ax3.plot(training_history.index, training_history['train_loss'], label='Train', linewidth=1.5)
ax3.plot(training_history.index, training_history['val_loss'], label='Val', linewidth=1.5)
ax3.set_xlabel('Epoch')
ax3.set_ylabel('Loss')
ax3.set_title('Training Convergence')
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.3)

# 4. RT prediction (LR)
ax4 = fig.add_subplot(gs[1, 0])
ax4.scatter(rt_valid, lr_pred, alpha=0.4, s=30, color='#2E86AB')
ax4.plot([rt_valid.min(), rt_valid.max()], [rt_valid.min(), rt_valid.max()], 'r--', linewidth=1)
ax4.set_xlabel('Actual RT')
ax4.set_ylabel('Predicted RT')
ax4.set_title(f'LR RT Prediction\nR²={lr_r2:.3f}')
ax4.grid(True, alpha=0.3)

# 5. RT prediction (RF)
ax5 = fig.add_subplot(gs[1, 1])
ax5.scatter(rt_valid, rf_pred, alpha=0.4, s=30, color='#A23B72')
ax5.plot([rt_valid.min(), rt_valid.max()], [rt_valid.min(), rt_valid.max()], 'r--', linewidth=1)
ax5.set_xlabel('Actual RT')
ax5.set_ylabel('Predicted RT')
ax5.set_title(f'RF RT Prediction\nR²={rf_r2:.3f}')
ax5.grid(True, alpha=0.3)

# 6. Model comparison
ax6 = fig.add_subplot(gs[1, 2])
models = ['LR\n(RT)', 'RF\n(RT)', 'LogReg\n(Acc)', 'RF\n(Acc)']
scores = [lr_r2, rf_r2, lr_clf_acc, rf_clf_acc]
colors_bar = ['#2E86AB', '#A23B72', '#2E86AB', '#A23B72']
ax6.bar(models, scores, color=colors_bar, alpha=0.7, edgecolor='black')
ax6.set_ylabel('Score (R² or Acc)')
ax6.set_title('Model Performance')
ax6.set_ylim([0, 1])
ax6.grid(True, alpha=0.3, axis='y')

# 7. Condition statistics
ax7 = fig.add_subplot(gs[2, 0])
condition_sizes = [np.sum(labels == 1), np.sum(labels == 2)]
ax7.bar(['Word', 'Pseudoword'], condition_sizes, color=['#2E86AB', '#A23B72'], alpha=0.7, edgecolor='black')
ax7.set_ylabel('N Trials')
ax7.set_title('Trial Counts')
ax7.grid(True, alpha=0.3, axis='y')

# 8. Feature importance
ax8 = fig.add_subplot(gs[2, 1])
top_5_dims = np.argsort(feature_importance)[-5:]
ax8.barh(range(5), feature_importance[top_5_dims], color='#F18F01', alpha=0.7, edgecolor='black')
ax8.set_yticks(range(5))
ax8.set_yticklabels([f'Dim {d}' for d in top_5_dims])
ax8.set_xlabel('Importance')
ax8.set_title('Top 5 Dimensions (RF)')
ax8.grid(True, alpha=0.3, axis='x')

# 9. Text summary
ax9 = fig.add_subplot(gs[2, 2])
ax9.axis('off')
summary_text = f"""
ANALYSIS SUMMARY

Subject: {SUBJECT_ID}
Trials: {len(trials)}
- Words: {np.sum(labels == 1)}
- Pseudowords: {np.sum(labels == 2)}

RT Prediction:
- LR R²: {lr_r2:.4f}
- RF R²: {rf_r2:.4f}

Accuracy Prediction:
- LogReg: {lr_clf_acc:.4f}
- RF: {rf_clf_acc:.4f}

Separation Index: {separation:.4f}
"""
ax9.text(0.1, 0.5, summary_text, fontsize=9, family='monospace',
         verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.savefig(FIGURES_DIR / '06_summary_report.png', dpi=DPI, bbox_inches='tight')
print(f"✓ Saved: 06_summary_report.png")
plt.close()

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print(f"\n{'='*70}")
print(f"ANALYSIS COMPLETE!")
print(f"{'='*70}")
print(f"\nAll figures saved to: {FIGURES_DIR}")
print(f"\nGenerated files:")
print(f"  • 01_training_convergence.png")
print(f"  • 02_manifold_pca_2d.png")
print(f"  • 03_manifold_pca_3d.png")
if HAS_UMAP:
    print(f"  • 04_manifold_umap.png")
print(f"  • 05_rt_prediction.png")
print(f"  • 06_summary_report.png")
print(f"  • prediction_summary.csv")
print(f"\n✓ Ready for next subject or group analysis!")
