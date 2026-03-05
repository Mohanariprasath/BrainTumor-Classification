# BrainScan AI: High-Performance Tumor Classification


<div align="center">

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-orange.svg)](https://pytorch.org/)
[![Flask](https://img.shields.io/badge/Flask-Web%20Framework-lightgrey.svg)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**An advanced deep learning framework for precise Brain Tumor detection, featuring state-of-the-art Model Explainability (Grad-CAM).**

[Overview](#-overview) • [Dataset Insight](#-dataset-insight) • [Preprocessing](#%EF%B8%8F-preprocessing) • [Performance Metrics](#-performance-metrics) • [Quick Start](#-quick-start)

</div>

---

##  Overview

**BrainScan AI** is designed to assist medical professionals by providing rapid, AI-driven preliminary scans of MRI data. Our system doesn't just predict; it **explains**. By integrating **Grad-CAM (Gradient-weighted Class Activation Mapping)**, the model highlights the specific regions in an MRI scan that contributed most to its classification, ensuring transparency and trust in the diagnostic process.

###  Key Features
- **Multi-Class Detection**: Accurate classification of **Glioma**, **Meningioma**, and **Pituitary** tumors, or **No Tumor**.
- **Explainable Insights**: Real-time heatmap generation to visualize tumor localization.
- **Dual Pipeline**: Optimized implementations in both **PyTorch** (Inference) and **TensorFlow** (Robust Training).
- **Interactive UI**: A modern, responsive web application for seamless image uploads and result viewing.

---

##  Dataset Insight

Our model is trained on a comprehensive collection of MRI scans spanning four distinct classes.

### 1. Data Distribution
The dataset is well-balanced to ensure robust learning across all classes:
<p align="center">
  <img src="assets/dataset_distribution.png" width="50%" alt="Dataset Distribution" />
</p>

### 2. Sample MRI Scans
Diverse MRI modalities are utilized to generalize the model's feature extraction:
<p align="center">
  <img src="assets/sample_mris.png" width="70%" alt="Sample MRI Images" />
</p>

---

##  Preprocessing

To maximize the model's focus on brain tissue and minimize noise, we employ automated cropping techniques that remove unnecessary dark background regions from the MRI scans.

<p align="center">
  <img src="assets/preprocessing.png" width="70%" alt="Original vs Cropped Image Comparison" />
</p>

---

##  Performance Metrics

Our models have been rigorously tested to ensure clinical-grade reliability.

###  Training & Validation Dynamics
The following plots illustrate the accuracy and loss curves during training. The stable convergence demonstrates strong generalization without significant overfitting.

<p align="center">
  <img src="assets/training_curves.png" width="80%" alt="Training and Validation Curves" />
</p>

###  Classification Results
The confusion matrix below highlights strong class separation, verifying that the model achieves high precision across all tumor classes and healthy scans.

<p align="center">
  <img src="assets/new_confusion_matrix.png" width="60%" alt="Confusion Matrix" />
</p>

| Model Architecture | Accuracy | F1-Score | Precision |
| :--- | :---: | :---: | :---: |
| **ResNet-18 (PyTorch)** | 94.2% | 0.94 | 0.95 |
| **ResNet-50 (TensorFlow)** | **96.1%** | **0.96** | **0.96** |

---

##  Explainable AI (Grad-CAM Results)

Understanding *why* a model makes a decision is paramount in healthcare. Below are localized heatmaps demonstrating the model's focus during successful classifications.

<p align="center">
  <img src="assets/grad_cam_1.png" width="45%" style="border-radius: 10px; margin: 10px;" />
  <img src="assets/grad_cam_2.png" width="45%" style="border-radius: 10px; margin: 10px;" />
</p>

| Patient Case | View Type | Analysis Outcome |
| :--- | :--- | :--- |
| **Case Study 01** | Sagittal View | **Focal Activation** in the primary tumor region (Red zone). |
| **Case Study 02** | Axial View | **Symmetric Comparison** with clear lateral tumor highlighting. |

---

##  License & Disclaimer

- **License**: MIT License. See `LICENSE` for details.
- **Disclaimer**: *This tool is intended for research and educational purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment.*

---
