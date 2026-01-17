# HAD-Net: Unsupervised Retinal Anomaly Detection using Hybrid Texture-Semantic Features

This repository contains the official implementation of **HAD-Net** (Hybrid Anomaly Detection Network), a novel unsupervised framework designed for identifying pathologies in retinal fundus images.

## 🌟 Research Highlights
- **Unsupervised Learning:** The model is trained exclusively on "healthy" samples to learn a robust representation of normality, enabling detection without expensive pathological annotations.
- **Hybrid Feature Fusion:** Fuses high-level semantic features (ResNet18) with interpretable low-level texture biomarkers (Haralick and Local Binary Patterns).
- **Superior Performance:** Achieved an **AUROC of 0.9435** on the APTOS 2019 dataset, outperforming the deep-only baseline by **+4.41%**.

## 📊 Dataset: APTOS 2019
Evaluation was performed on the **APTOS 2019 Blindness Detection** dataset (provided by the Asia Pacific Tele-Ophthalmology Society).
- **Training Set:** 1,434 healthy images (Class 0).
- **Test Set:** 366 mixed images (Healthy Class 0 vs. Anomalous Classes 1-4).

## 🔬 Methodology: Why Hybrid?
Standard Deep Learning models often prioritize macroscopic structures while overlooking subtle micro-textural variations (e.g., tiny hemorrhages or exudates). HAD-Net bridges this gap by integrating:
1. **Semantic Stream:** Pre-trained ResNet18 backbone for global geometry.
2. **Texture Stream:** Specialized branch extracting 16 handcrafted texture biomarkers (GLCM & LBP).
3. **One-Class Center Loss:** A contrastive objective mapping samples into a compact 128-D latent hypersphere.

## 📈 Ablation Study & Results
| Method | Texture Features | AUROC | Improvement |
| :--- | :---: | :---: | :---: |
| ResNet18 (Baseline) | ❌ | 0.9036 | - |
| **HAD-Net (Ours)** | ✅ | **0.9435** | **+4.41%** |

### Visual Analysis
- **ROC Curve:** The **green curve** (HAD-Net) demonstrates higher sensitivity at low false-positive rates compared to the red baseline.
- **t-SNE Visualization:** Healthy samples form a tight, dense cluster, while diseased samples are effectively mapped to different regions of the latent space.

## 📂 Project Structure
- **`preprocessing/`**: Implementation of the image enhancement pipeline (Ben Graham's method and CLAHE filters).
- **`features/`**: Extraction scripts for handcrafted biomarkers (GLCM/Haralick and Local Binary Patterns).
- **`models/`**: PyTorch implementations of the proposed **HAD-Net** and the **Deep-Only Baseline**.
- **`data/`**: Custom Dataset classes for multi-modal loading of images and texture vectors.
- **`utils/`**: Core logic for hypersphere center initialization and SVDD training loops.
- **`scripts/`**: Automated data preparation and large-scale feature extraction pipelines.
- **`notebooks/`**: Comprehensive ablation study, performance comparison, and visualization.
- **`main.py`**: Main entry point for training and model persistence.

## 🚀 How to Run
1. **Preprocess Data:** Run `python scripts/prepare_data.py` to extract texture biomarkers and filter images.
2. **Train Model:** Run `python main.py` to train the HAD-Net model.
3. **Compare Results:** Open `notebooks/Ablation_Study_Comparison.ipynb` to visualize the ROC curves and performance gain.
   
## 🎓 Citation
If you find this work useful, please cite our paper:
> **S. M. Kaddour**, et al. "Unsupervised Retinal Anomaly Detection using Hybrid Texture-Semantic Features." To appear in Proc. 1st International Conference on Data Analytics and Intelligent Systems (DAIS), **Springer CCIS**, 2026.
