# Neural Network Optimization, Normalization, and Regularization with Scikit-learn and PyTorch

This module investigates the core training dynamics, optimization algorithms, internal normalization schemes, and variance regularization methods for Multi-Layer Perceptrons (MLPs) using PyTorch and Scikit-learn.

---

# Problem 1 — Comparing Optimization Algorithms for MLPClassifier

## Objective

Train and benchmark Multi-Layer Perceptrons using different first-order and adaptive optimization algorithms on the **Breast Cancer Wisconsin dataset**, analyzing their loss trajectories, convergence rates, computational overhead, and generalization capabilities.

## Key Tasks & Implementations

- **Data Standardization**: Standardize features using StandardScaler fitted strictly on training data (70/30 split).
- **Optimization Algorithms**:
  - **Vanilla Stochastic Gradient Descent (SGD)**: Fixed learning rate (eta = 0.01).
  - **SGD with Classical Momentum**: Momentum parameter mu = 0.9 to dampen oscillations in high-curvature directions.
  - **Adam (Adaptive Moment Estimation)**: First and second raw moment estimation with bias correction.
  - **AdamW (Decoupled Weight Decay)**: Decoupling weight decay (lambda = 0.01) from gradient moments to improve generalization.
- **Learning Rate Scheduling**: Benchmark Constant Learning Rate vs. Cosine Annealing on SGD.

## Summary of Results

| Optimizer / Schedule        |  Accuracy  | Precision  | Recall  |  F1-Score  | Training Time (s) |     Epochs      |
| :-------------------------- | :--------: | :--------: | :-----: | :--------: | :---------------: | :-------------: |
| **SGD (Vanilla)**           |   89.47%   |   88.03%   | 96.26%  |   0.9196   |      0.1400       | 35 (early stop) |
| **SGD + Momentum (mu=0.9)** |   95.32%   |   93.04%   | 100.00% |   0.9640   |      0.0618       | 39 (early stop) |
| **Adam**                    |   95.91%   |   94.64%   | 99.07%  |   0.9680   |      0.0925       | 39 (early stop) |
| **AdamW (Decoupled Decay)** | **97.08%** | **99.04%** | 96.26%  | **0.9763** |      1.0172       |       226       |
| **SGD (Constant LR)**       |   96.49%   |   96.33%   | 98.13%  |   0.9722   |      4.9277       |      1000       |
| **SGD (Cosine Annealing)**  |   95.32%   |   94.59%   | 98.13%  |   0.9633   |      4.0934       |      1000       |

---

# Problem 2 — Investigating the Impact of Normalization and Regularization Techniques

## Objective

Evaluate the empirical effects of **Batch Normalization** and **Regularization techniques** (Dropout, L2 Weight Decay, and their Combination) on multiclass classification of the **Scikit-learn Digits dataset** (1,797 samples, 64 pixel features, 10 handwritten digit classes).

## Architectural Methodology

- **Input Normalization vs. Internal Batch Normalization**:
  The Baseline MLP receives **raw, unscaled pixel inputs** in the interval [0, 16]. Standardizing inputs upfront would mask the distinct impact of internal Batch Normalization. Normalization is therefore applied internally using `nn.BatchNorm1d` within the network layers.
- **Loss Function**: Multiclass Cross-Entropy Loss with Softmax logits.
- **Data Partitioning**: Stratified 80/20 train/test split, with a nested 90/10 stratified validation split from training samples for live convergence tracking.

## Configuration Benchmarks

| Configuration         | Normalization | Dropout |  Weight Decay  | Test Accuracy | Test Log-Loss | Gen Gap (Loss) |  Status   |
| :-------------------- | :-----------: | :-----: | :------------: | :-----------: | :-----------: | :------------: | :-------: |
| **1. Baseline**       |     None      |  None   |  lambda = 0.0  |    97.22%     |    0.0777     |     0.0776     | Completed |
| **2. BatchNorm Only** |  BatchNorm1d  |  None   |  lambda = 0.0  |  **99.17%**   |  **0.0388**   |     0.0375     | Completed |
| **3. Dropout Only**   |     None      | p = 0.3 |  lambda = 0.0  |    98.33%     |    0.0780     |     0.0601     | Completed |
| **4. L2 Reg Only**    |     None      |  None   | lambda = 0.001 |    98.06%     |    0.0608     |     0.0592     | Completed |
| **5. Combined**       |  BatchNorm1d  | p = 0.3 | lambda = 0.001 |    98.61%     |    0.0464     |   **0.0197**   | Completed |

## Key Insights

1. **Overfitting Mitigation**: The Baseline exhibits the largest Generalization Gap (0.0776). The Combined architecture (BatchNorm + Dropout + L2) achieves the smallest Generalization Gap (0.0197), proving the synergy of multi-faceted regularization in combating overfitting.
2. **Probability Calibration**: Batch Normalization dramatically reduces Test Log-Loss from 0.0777 down to 0.0388 (a 50% relative reduction), achieving the highest Test Accuracy (99.17%) and lowest misclassification rate.
3. **Feature Independence**: Dropout successfully prevents feature co-adaptation, ensuring validation loss does not diverge even without weight decay.

---

# References

- Kingma, D. P., & Ba, J. (2014). _Adam: A Method for Stochastic Optimization_. arXiv:1412.6980.
- Loshchilov, I., & Hutter, F. (2017). _Decoupled Weight Decay Regularization (AdamW)_. arXiv:1711.05101.
- Ioffe, S., & Szegedy, C. (2015). _Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift_. ICML 2015.
- Srivastava, N., et al. (2014). _Dropout: A Simple Way to Prevent Neural Networks from Overfitting_. JMLR.
