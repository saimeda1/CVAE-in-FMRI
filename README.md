# Neural Manifold Analysis of Lexical Processing

**Deep learning approach to understanding how the brain represents words**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

---

This project applies **Variational Autoencoders (VAEs)** to fMRI data from an auditory lexical decision task in an attempt to discover low-dimensional neural manifolds that organize lexical representations.

## Results

### 1. Manifold Discovery
Successfully compressed ~390k voxels into a 15D latent space that captures task-relevant brain dynamics.

![UMAP Visualization](figures/04_manifold_umap.jpg)

### 2. Distinct Brain States
Identified 6-8 clusters representing different cognitive states during word processing:

![Cluster Analysis](figures/06_summary_report.jpg)

### 3. Behavioral Prediction
Neural manifold position predicts reaction time with high accuracy (R² = 0.85 with Random Forest).

![RT Prediction](figures/05_rt_prediction.jpg)

### 4. Training Stability
VAE converged smoothly with balanced reconstruction and regularization.

![Training](figures/01_training_convergence.jpg)

---

## Methods

**Data**: Auditory lexical decision fMRI (3T, TR=2s, 684 trials)

**Model**: Conditional Variational Autoencoder
- Encoder: fMRI (390k voxels) → 15D latent space
- Decoder: 15D latent → reconstructed fMRI
- Loss: MSE reconstruction + KL divergence

**Analysis**:
- Dimensionality reduction (PCA, UMAP)
- Cluster analysis (K-means)
- Behavioral prediction (Random Forest, Ridge Regression)
- Semantic feature encoding (cross-validated R²)

---

## 📈 Findings

### What I Discovered
**Neural manifolds organize by task states** (attention, arousal, difficulty)  
**Behavioral variability is decodable** from neural state  
**Distinct clustering patterns** emerge without supervision  
**Trial-order effects** indicate attentional drift (r=0.12, p<0.01)

### What I Tested
**Semantic feature encoding**: Manifold did not capture word-level semantic properties (all R² < 0)  
**Lexical category separation**: Words vs. pseudowords heavily overlapped  

**Interpretation**: Lexical decision tasks may access phonological/orthographic representations rather than deep semantic meaning.

