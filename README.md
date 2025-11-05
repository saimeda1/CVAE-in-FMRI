# Neural Manifold Analysis of Lexical Processing

**Deep learning approach to understanding how the brain represents words**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

---

This project applies **Variational Autoencoders (VAEs)** to fMRI data from an auditory lexical decision task in an attempt to discover low-dimensional neural manifolds that organize lexical representations.

## Results

### 1. Manifold Discovery
Successfully compressed ~390k voxels into a 15D latent space that captures task-relevant brain dynamics.

<img width="984" height="782" alt="Screenshot 2025-11-05 at 2 19 52 PM" src="https://github.com/user-attachments/assets/7eae1cc4-921a-4ba2-ba20-1d9250ff4506" />

### 2. Distinct Brain States
Identified 6-8 clusters representing different cognitive states during word processing:

<img width="1109" height="785" alt="Screenshot 2025-11-05 at 2 20 37 PM" src="https://github.com/user-attachments/assets/59c614ab-6bd9-44ed-8086-b78097a4a370" />


### 3. Behavioral Prediction
Neural manifold position predicts reaction time with high accuracy (R² = 0.85 with Random Forest).

<img width="1476" height="520" alt="Screenshot 2025-11-05 at 2 20 55 PM" src="https://github.com/user-attachments/assets/982b7df5-9623-4fd4-adc8-25cd7abbb52d" />


### 4. Training Stability
VAE converged smoothly with balanced reconstruction and regularization.

<img width="1479" height="501" alt="Screenshot 2025-11-05 at 2 21 16 PM" src="https://github.com/user-attachments/assets/51d542d4-0256-48fd-a186-efbdce9e59b7" />


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

## Findings

### What I Discovered
**Neural manifolds organize by task states** (attention, arousal, difficulty)  
**Behavioral variability is decodable** from neural state  
**Distinct clustering patterns** emerge without supervision  
**Trial-order effects** indicate attentional drift (r=0.12, p<0.01)

### What I Tested
**Semantic feature encoding**: Manifold did not capture word-level semantic properties (all R² < 0)  
**Lexical category separation**: Words vs. pseudowords heavily overlapped  

**Interpretation**: Lexical decision tasks may access phonological/orthographic representations rather than deep semantic meaning.

