# DA5401 A4 – GMM-Based Synthetic Sampling for Imbalanced Data  
**Name:** Omkar Chaudhari  
**Roll No.:** NA22B059  

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

## Procedure

1. **Baseline (on original, imbalanced data)**
   - Loaded the `creditcard.csv` dataset and confirmed severe fraud–nonfraud imbalance.
   - Split the data into training and test sets, ensuring the **test set preserves the imbalance**.
   - Trained a **Logistic Regression** model on the imbalanced training data.
   - Evaluated on the test set using **precision, recall, F1-score for the fraud class**, since accuracy is misleading in imbalanced settings.

2. **GMM Training**
   - Trained a **Gaussian Mixture Model** on only the **minority class** (frauds).
   - Chose the number of components **k** using the **Bayesian Information Criterion (BIC)** to avoid overfitting.

3. **Synthetic Sample Generation**
   - Sampled new fraud points from the GMM by:
     - Drawing a component according to its weight.
     - Sampling from its multivariate normal distribution.
     - Inversely transforming the data back to the original space.
   - Generated enough fraud samples to match the majority class.

4. **Clustering-Based Undersampling (CBU)**
   - Reduced the majority class using KMeans-based undersampling.
   - Balanced the training set by combining the GMM-generated frauds with the reduced majority set.

5. **Two Versions of Balancing**
   - **V1 (GMM → CBU)**: Generated synthetic frauds first, then undersampled the majority.
   - **V2 (CBU → GMM)**: Undersampled majority first, then matched minority count with GMM-sampled frauds.

6. **Model Retraining and Evaluation**
   - Retrained Logistic Regression models on both balanced versions.
   - Evaluated all models (Baseline, V1, V2) on the **same imbalanced test set** for consistency.

---

## Experiments Conducted

| Model              | Precision (Fraud) | Recall (Fraud) | F1-score (Fraud) |
|-------------------|-------------------|----------------|------------------|
| Baseline (LR)     | 0.8476            | 0.6014         | 0.7036           |
| GMM → CBU (V1)    | 0.0878            | 0.8581         | 0.1592           |
| CBU → GMM (V2)    | 0.0872            | 0.8581         | 0.1583           |

> **Insight**: GMM-based oversampling drastically improved **recall**, enabling the model to detect ~86% of frauds, but at the cost of precision.

---

## Conclusion

- **Baseline model** performed well in terms of precision, but missed many frauds (lower recall).
- **GMM-balanced models** (V1 & V2) dramatically increased recall but suffered a **drop in precision**, leading to more false positives.
- The **ROC–AUC** score improved slightly (≈ 0.955 → 0.968–0.969), showing better ranking of fraud likelihood.

### Final Recommendation

**GMM-based oversampling** is a powerful, structure-aware, and practical technique for improving **recall** in fraud detection — especially when **missing fraudulent transactions is more costly** than raising false alarms.  

To mitigate the drop in precision:
- Use **threshold tuning**, **probability calibration**, or **cost-sensitive learning** in production.
- Consider GMM as a **complementary technique** rather than a standalone fix.

---
