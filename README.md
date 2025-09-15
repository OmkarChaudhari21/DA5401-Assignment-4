# DA5401 Assignment 4 — GMM-Based Synthetic Sampling for Imbalanced Fraud Detection

**Course**: DA5401 — Data Analytics Lab  
**Assignment**: A4 — GMM-Based Oversampling  
**Name**: Omkar Chaudhari  
**Roll No**: NA22B059  
**Date**: 15th September 2025  
**Dataset**: [Credit Card Fraud Detection (Kaggle)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

---

## Objective

This project applies a **Gaussian Mixture Model (GMM)** to address the challenge of extreme class imbalance in fraud detection. The goal is to generate **realistic synthetic minority class samples** and build classifiers that detect frauds more effectively than baseline models.

---

## Key Concepts

- **Class Imbalance Problem**: Only 0.17% of transactions are fraudulent.
- **GMM-Based Oversampling**: Learns a probabilistic model of the minority class and samples new points from the learned mixture.
- **Clustering-Based Undersampling (CBU)**: Reduces the majority class while preserving its structure using KMeans.
- **Performance Evaluation**: Precision, Recall, F1-score, ROC–AUC, and bootstrapped confidence intervals.

---

## Repository Structure
├── a4.ipynb # Original notebook
├── a4_improved.ipynb # Enhanced notebook with bonus features
├── README.md # This file
└── creditcard.csv # Input dataset (not included; download from Kaggle)


---

## Experiments Conducted

| Model              | Precision (Fraud) | Recall (Fraud) | F1-score (Fraud) |
|-------------------|-------------------|----------------|------------------|
| Baseline (LR)     | 0.8476            | 0.6014         | 0.7036           |
| GMM → CBU (V1)     | 0.0878            | 0.8581         | 0.1592           |
| CBU → GMM (V2)     | 0.0872            | 0.8581         | 0.1583           |

> GMM-based oversampling drastically improved recall, enabling the model to detect 86% of frauds.
Final Recommendation

GMM-based oversampling is a powerful, structure-aware, and practical technique for improving recall in fraud detection, especially when missing fraudulent cases is unacceptable.

