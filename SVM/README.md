# Support Vector Machines (SVM)

This module provides a rigorous theoretical investigation and empirical benchmark of **Support Vector Machines (SVM)** in Python using `scikit-learn`. It explores the fundamental transition from linear hyperplanes to non-linear reproducing kernel Hilbert spaces (RKHS), evaluating maximum-margin classification, duality, Mercer kernels, support vector sparsity, and hyperparameter sensitivity.

---

## Table of Contents

- [Module Overview and Status](#module-overview-and-status)
- [Module Architecture and Artifacts](#module-architecture-and-artifacts)
- [Problem 1: Comparison of Linear and Non-linear SVM](#problem-1-comparison-of-linear-and-non-linear-svm)
  - [Mathematical Formulations and Duality](#mathematical-formulations-and-duality)
  - [The Kernel Trick and Mercer Kernels](#the-kernel-trick-and-mercer-kernels)
  - [Experimental Setup and Preprocessing Protocol](#experimental-setup-and-preprocessing-protocol)
  - [Empirical Benchmark Results](#empirical-benchmark-results)
  - [Support Vector Sparsity and Inference Latency](#support-vector-sparsity-and-inference-latency)
  - [Visual Diagnostics and Decision Boundaries](#visual-diagnostics-and-decision-boundaries)
  - [Comprehensive Kernel Comparison Matrix](#comprehensive-kernel-comparison-matrix)
  - [Scenario-Based Engineering Selection Guidelines](#scenario-based-engineering-selection-guidelines)
- [Problem 2: Study on the Effect of SVM Parameters](#problem-2-study-on-the-effect-of-svm-parameters)
  - [Theoretical Foundations of SVM Hyperparameters](#theoretical-foundations-of-svm-hyperparameters)
  - [Effect of Regularization Parameter C (Task a)](#effect-of-regularization-parameter-c-task-a)
  - [Effect of RBF Kernel Parameter Gamma (Task b)](#effect-of-rbf-kernel-parameter-gamma-task-b)
  - [Hard-Margin vs. Soft-Margin Under Overlapping Noise (Task c)](#hard-margin-vs-soft-margin-under-overlapping-noise-task-c)
  - [Hyperparameter Optimization via GridSearchCV (Task d)](#hyperparameter-optimization-via-gridsearchcv-task-d)
  - [Visual Diagnostics and Parameter Influence Dashboard](#visual-diagnostics-and-parameter-influence-dashboard)
  - [Comprehensive Hyperparameter Engineering Matrix](#comprehensive-hyperparameter-engineering-matrix)
- [Execution and Reproduction Guide](#execution-and-reproduction-guide)
- [References and Academic Literature](#references-and-academic-literature)

---

## Module Overview and Status

| Problem | Notebook | Domain / Dataset | Key Techniques & Algorithms | Status |
| :---: | :--- | :--- | :--- | :---: |
| **1** | [`Linear_vs_Nonlinear_SVM.ipynb`](./Linear_vs_Nonlinear_SVM.ipynb) | Synthetic 2D ($N=500$) & Iris Binary ($N=100$) | Maximum Margin, Primal/Dual, KKT Conditions, `LinearSVC`, RBF, Poly ($d=3$), Sigmoid, Support Vector Sparsity | `Completed` |
| **2** | [`Study_SVM_Parameters.ipynb`](./Study_SVM_Parameters.ipynb) | Synthetic 2D ($N=500$) & Overlapping Blobs ($N=300$) | Hyperparameter Sweeps ($C, \gamma$), Margin Width ($2/\Vert \mathbf{w} \Vert_2$), Hard vs. Soft Margin, `GridSearchCV`, Bias-Variance Diagnostics | `Completed` |

---

## Module Architecture and Artifacts

```text
SVM/
├── SVM_Lab.pdf                     # Official coursework specification
├── Linear_vs_Nonlinear_SVM.ipynb   # Problem 1: Comparison of Linear and Non-linear SVM
├── Study_SVM_Parameters.ipynb      # Problem 2: Study on the Effect of SVM Parameters
└── README.md                       # Comprehensive module documentation
```

---

# Problem 1: Comparison of Linear and Non-linear SVM

## Mathematical Formulations and Duality

Support Vector Machines optimize the structural risk bound of Vapnik-Chervonenkis (VC) theory by discovering the separating hyperplane that maximizes the geometric margin $\gamma = \frac{2}{\Vert \mathbf{w} \Vert}$.

### 1. Primal Soft-Margin Formulation
For a binary dataset $\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^N$ with $\mathbf{x}_i \in \mathbb{R}^D$ and $y_i \in \{-1, +1\}$:

$$\min_{\mathbf{w}, b, \boldsymbol{\xi}} \frac{1}{2} \Vert \mathbf{w} \Vert^2 + C \sum_{i=1}^N \xi_i \quad \text{subject to} \quad y_i (\mathbf{w}^T \mathbf{x}_i + b) \ge 1 - \xi_i, \quad \xi_i \ge 0, \quad \forall i$$

where $C > 0$ controls the trade-off between margin width and slack penalty $\xi_i$.

### 2. Dual Lagrangian Formulation
Formulating the Lagrangian with multipliers $\alpha_i \ge 0$ and $\beta_i \ge 0$ and applying the KKT saddle-point optimality conditions ($\nabla_{\mathbf{w}} L = 0 \implies \mathbf{w} = \sum_{i=1}^N \alpha_i y_i \mathbf{x}_i$, $\nabla_b L = 0 \implies \sum_{i=1}^N \alpha_i y_i = 0$):

$$\max_{\boldsymbol{\alpha}} \sum_{i=1}^N \alpha_i - \frac{1}{2} \sum_{i=1}^N \sum_{j=1}^N \alpha_i \alpha_j y_i y_j K(\mathbf{x}_i, \mathbf{x}_j) \quad \text{subject to} \quad 0 \le \alpha_i \le C, \quad \sum_{i=1}^N \alpha_i y_i = 0$$

### 3. KKT Complementary Slackness & Support Vectors
- **Non-Support Vectors** ($\alpha_i = 0$): Points strictly outside the margin ($y_i f(\mathbf{x}_i) > 1$, $\xi_i = 0$). Zero influence on the boundary.
- **Free Margin Support Vectors** ($0 < \alpha_i < C$): Points lying precisely on the margin boundary ($y_i f(\mathbf{x}_i) = 1$, $\xi_i = 0$).
- **Bounded Support Vectors** ($\alpha_i = C$): Margin-violating points lying inside the margin or misclassified ($y_i f(\mathbf{x}_i) < 1$, $\xi_i > 0$).

The decision function depends only on the active support vectors:

$$f(\mathbf{x}) = \text{sign}\left( \sum_{i \in \text{SV}} \alpha_i y_i K(\mathbf{x}_i, \mathbf{x}) + b \right)$$

---

## The Kernel Trick and Mercer Kernels

By Mercer's theorem, any continuous, symmetric, positive semi-definite kernel function $K(\mathbf{x}, \mathbf{x}')$ implicitly computes the inner product in a transformed Hilbert feature space $\mathcal{H}$ without explicit feature mapping:

$$K(\mathbf{x}, \mathbf{x}') = \langle \Phi(\mathbf{x}), \Phi(\mathbf{x}') \rangle_\mathcal{H}$$

1. **Linear Kernel**: $K(\mathbf{x}, \mathbf{x}') = \mathbf{x}^T \mathbf{x}'$
2. **Radial Basis Function (RBF / Gaussian) Kernel**: $K(\mathbf{x}, \mathbf{x}') = \exp\left( -\gamma \Vert \mathbf{x} - \mathbf{x}' \Vert^2 \right)$ (Infinite-dimensional RKHS, universal approximator)
3. **Polynomial Kernel**: $K(\mathbf{x}, \mathbf{x}') = (\gamma \mathbf{x}^T \mathbf{x}' + r)^d$ ($d=3$ cubic interactions)
4. **Sigmoid Kernel**: $K(\mathbf{x}, \mathbf{x}') = \tanh(\gamma \mathbf{x}^T \mathbf{x}' + r)$ (Conditionally positive semi-definite)

---

## Experimental Setup and Preprocessing Protocol

Two distinct evaluation domains were configured to test linear and non-linear boundaries:
- **Synthetic 2D Dataset (`make_classification`)**: $N = 500$ samples, 2 informative features, 2 classes, moderately non-linear separation landscape.
- **Real-World Iris Dataset Binary Formulation**: $N = 100$ samples (Versicolor vs. Virginica, classes 1 vs. 2), evaluated on the 2 most discriminative morphological attributes: Petal Length and Petal Width.

### Preprocessing Protocol:
- **Stratified Partitioning**: 70% Training ($N_{\text{tr}} = 350$ synthetic, $70$ Iris), 30% Testing ($N_{\text{te}} = 150$ synthetic, $30$ Iris) via `train_test_split(test_size=0.30, random_state=42, stratify=y)`.
- **Z-Score Standardization**: `StandardScaler()` fitted strictly on the training partitions to eliminate data leakage.

---

## Empirical Benchmark Results

### 1. Synthetic 2D Classification Benchmark ($N = 500$, Test $N = 150$)

| Model / Kernel Architecture | Test Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | Support Vectors ($N_{\text{SV}}$) | Sparsity Ratio (%) | Train Time (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Linear SVM (`LinearSVC`)** | 88.67 | 88.16 | 89.33 | 88.74 | 233 | 66.57 | 3.37 |
| **Linear SVM (`SVC(linear)`)** | 92.00 | 90.91 | 93.33 | 92.11 | 119 | 34.00 | 6.21 |
| **RBF Kernel (`SVC(rbf)`)** | **94.67** | **94.67** | **94.67** | **94.67** | **101** | **28.86** | 4.60 |
| **Polynomial Kernel (`SVC(poly, d=3)`)** | 83.33 | 85.29 | 80.00 | 82.57 | 198 | 56.57 | 6.40 |
| **Sigmoid Kernel (`SVC(sigmoid)`)** | 91.33 | 91.89 | 90.67 | 91.28 | 98 | 28.00 | 5.42 |

### 2. Real-World Iris Dataset Benchmark ($N = 100$, Test $N = 30$)

| Model / Kernel Architecture | Test Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | Support Vectors ($N_{\text{SV}}$) | Sparsity Ratio (%) | Train Time (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Linear SVM (`LinearSVC`)** | 86.67 | 86.67 | 86.67 | 86.67 | 17 | 24.29 | 2.36 |
| **Linear SVM (`SVC(linear)`)** | 86.67 | 86.67 | 86.67 | 86.67 | 12 | 17.14 | 2.15 |
| **RBF Kernel (`SVC(rbf)`)** | **90.00** | **87.50** | **93.33** | **90.32** | 15 | 21.43 | 2.22 |
| **Polynomial Kernel (`SVC(poly, d=3)`)** | 76.67 | 88.89 | 66.67 | 76.19 | 28 | 40.00 | 2.69 |
| **Sigmoid Kernel (`SVC(sigmoid)`)** | **90.00** | **87.50** | **93.33** | **90.32** | 15 | 21.43 | 2.67 |

---

## Support Vector Sparsity and Inference Latency

- **Sparsity Ratio ($\rho_{\text{SV}} = N_{\text{SV}} / N_{\text{tr}}$)**:
  - **RBF SVM** achieves the optimal balance between high predictive accuracy and sparsity ($28.86\%$ on Synthetic, $21.43\%$ on Iris).
  - **Polynomial SVM ($d=3$)** exhibits high support vector density ($56.57\%$ on Synthetic), indicating broad margin violations and elevated inference computational overhead.
- **Primal vs. Dual Inference Speed**:
  - `LinearSVC` discards individual support vectors post-training, storing only the weight vector $\mathbf{w} \in \mathbb{R}^D$ and scalar bias $b \in \mathbb{R}$. Test inference requires only $\mathcal{O}(D)$ operations.
  - Non-linear kernels (`SVC`) require evaluating $K(\mathbf{x}_i, \mathbf{x})$ against all $N_{\text{SV}}$ active support vectors, scaling as $\mathcal{O}(N_{\text{SV}} \cdot D)$.

---

## Visual Diagnostics and Decision Boundaries

The notebook embeds 5 high-DPI base64 visual figures:

1. **Figure 1: Exploratory Feature Landscape and Partition Diagnostics**:
   - 2-panel scatter plots displaying standardized feature distributions and stratified train/test partitions for both synthetic and Iris datasets.
2. **Figure 2: Empirical Decision Boundaries & Margins on Synthetic Dataset**:
   - 4-panel $2 \times 2$ grid displaying continuous contour decision gradients, solid decision boundaries ($f(\mathbf{x}) = 0$), dashed margin bounds ($f(\mathbf{x}) = \pm 1$), and highlighted green support vectors.
3. **Figure 3: Empirical Decision Boundaries & Margins on Real-World Iris Dataset**:
   - 4-panel $2 \times 2$ grid illustrating the geometric adaptation of each kernel to the overlapping Versicolor vs. Virginica petal measurements.
4. **Figure 4: Comprehensive Multi-Metric Comparative Diagnostics Dashboard**:
   - Panel A: Grouped bar chart comparing Test Accuracy across all kernels and datasets.
   - Panel B: Support Vector counts and sparsity ratios.
   - Panel C: Wall-clock training latency (ms).
   - Panel D: Receiver Operating Characteristic (ROC) curves with AUC scores.
5. **Figure 5: Geometric Intuition of the Kernel Trick (3D Projection)**:
   - 2D non-linearly separable concentric circles mapped via $\Phi(\mathbf{x}) = (x_1, x_2, x_1^2 + x_2^2)^T$ into a 3D feature space, separated cleanly by a planar slice $z = h$.

---

## Comprehensive Kernel Comparison Matrix

| Kernel Architecture | Mathematical Formula | Feature Space Dimension | Key Hyperparameters | Primary Strengths | Potential Vulnerabilities |
| :--- | :--- | :---: | :---: | :--- | :--- |
| **Linear SVM** (`LinearSVC` / `SVC(linear)`) | $K(\mathbf{x}, \mathbf{x}') = \mathbf{x}^T \mathbf{x}'$ | $D$ | $C$ | Blazing fast $\mathcal{O}(N \cdot D)$ coordinate descent via LIBLINEAR; test latency $\mathcal{O}(D)$ independent of $N_{\text{SV}}$; resistant to overfitting. | Constrained by hyperplanar bias; cannot model curved or non-linear boundaries. |
| **RBF (Gaussian) Kernel** | $K(\mathbf{x}, \mathbf{x}') = \exp(-\gamma \Vert \mathbf{x} - \mathbf{x}' \Vert^2)$ | $\infty$ | $C, \gamma$ | Universal function approximator; smooth localized boundary curvature; top empirical benchmark accuracy. | Slower inference $\mathcal{O}(N_{\text{SV}} \cdot D)$; sensitive to feature scaling and choice of $\gamma$. |
| **Polynomial Kernel** | $K(\mathbf{x}, \mathbf{x}') = (\gamma \mathbf{x}^T \mathbf{x}' + r)^d$ | $\binom{D+d}{d}$ | $C, \gamma, r, d$ | Naturally models combinatorial feature interactions and physical polynomial dynamics. | High risk of boundary oscillations far from training data (Runge's phenomenon); sensitive to degree $d$. |
| **Sigmoid Kernel** | $K(\mathbf{x}, \mathbf{x}') = \tanh(\gamma \mathbf{x}^T \mathbf{x}' + r)$ | Varied | $C, \gamma, r$ | Functional proxy for multi-layer perceptron neural networks; smooth decision surfaces. | Conditionally positive semi-definite; violates Mercer's condition if parameters are untuned, causing optimization failure. |

---

## Scenario-Based Engineering Selection Guidelines

1. **High-Dimensional Feature Regimes ($D \gg N$)**:
   - *Applications*: Text classification (Bag-of-Words / TF-IDF), genomics, mass spectrometry.
   - *Selection*: **Linear SVM (`LinearSVC`)**.
   - *Rationale*: Samples in high-dimensional spaces are almost always linearly separable. Kernel projections add computational overhead without accuracy benefits.
2. **Non-Linear Tabular Datasets ($N \ge D$ or Complex Manifolds)**:
   - *Applications*: Clinical diagnostics, sensor analytics, risk scoring.
   - *Selection*: **RBF Kernel (`SVC(kernel='rbf')`)**.
   - *Rationale*: Default gold standard. Requires tuning only $C$ and $\gamma$, and reliably fits non-linear manifolds.
3. **Ultra-Low Latency Embedded / Edge Deployment**:
   - *Applications*: Microcontrollers, automotive real-time ECUs, mobile on-device classification.
   - *Selection*: **Linear SVM (`LinearSVC`)**.
   - *Rationale*: Pre-computes $\mathbf{w}$ and $b$; requires zero kernel evaluations and minimal RAM.
4. **Physical Systems with Explicit Polynomial Dynamics**:
   - *Applications*: Aerodynamics, kinematic trajectories, chemical reaction kinetics.
   - *Selection*: **Polynomial Kernel (`SVC(kernel='poly', degree=2)`)**.
   - *Rationale*: Directly captures degree-$d$ interactions without manual feature engineering.

---

# Problem 2: Study on the Effect of SVM Parameters

## Theoretical Foundations of SVM Hyperparameters

The performance, boundary geometry, and generalization capacity of Support Vector Machines are governed by two fundamental hyperparameters: the regularization penalty $C$ and the kernel bandwidth parameter $\gamma$.

### 1. Mathematical Mechanics of the Regularization Parameter C
In soft-margin SVM, the primal optimization objective balances maximum margin separation against empirical slack penalties $\xi_i$:

$$\min_{\mathbf{w}, b, \boldsymbol{\xi}} \frac{1}{2} \Vert \mathbf{w} \Vert^2 + C \sum_{i=1}^N \xi_i \quad \text{subject to} \quad y_i (\mathbf{w}^T \mathbf{x}_i + b) \ge 1 - \xi_i, \quad \xi_i \ge 0, \quad \forall i$$

- **Geometric Margin Width**:
$$M = \frac{2}{\Vert \mathbf{w} \Vert_2}$$
- **Primal-Dual Mechanics**:
  - In the dual objective, $C$ imposes an upper bound on the Lagrange multipliers: $0 \le \alpha_i \le C$.
  - When $C \to 0$, individual slack penalties are virtually unpenalized. The optimizer prioritizes maximizing margin width $M = 2/\Vert \mathbf{w} \Vert_2$ (minimizing $\Vert \mathbf{w} \Vert_2$), yielding a wide, flat margin where nearly all samples violate the margin condition and become bounded support vectors (high bias, low variance, underfitting).
  - When $C \to \infty$, slack penalties are magnified infinitely. The optimizer strictly forbids margin violations ($\xi_i \to 0$). The margin contracts dramatically, forcing the decision boundary to contort around noise and outliers (low bias, high variance, approaching hard-margin overfitting).

### 2. Mathematical Mechanics of the RBF Kernel Parameter Gamma
The Radial Basis Function (RBF) kernel is parameterized by $\gamma > 0$:

$$K(\mathbf{x}, \mathbf{x}') = \exp\left(-\gamma \Vert \mathbf{x} - \mathbf{x}' \Vert^2\right) = \exp\left(-\frac{\Vert \mathbf{x} - \mathbf{x}' \Vert^2}{2\sigma^2}\right), \quad \gamma = \frac{1}{2\sigma^2}$$

- **Gaussian Influence Radius $\sigma = 1 / \sqrt{2\gamma}$**:
  - **Small $\gamma$ ($\gamma \to 0$, large $\sigma$)**: Each support vector casts a broad, gradual Gaussian bell across the entire feature space. The decision surface behaves nearly linearly, incapable of capturing local non-linear structure (underfitting).
  - **Large $\gamma$ ($\gamma \to \infty$, small $\sigma$)**: Each support vector casts an extremely sharp, localized Gaussian spike. The decision boundary fragments into isolated "islands" encircling individual training points, memorizing empirical training noise while degrading global generalization (overfitting).

### 3. Hard-Margin vs. Soft-Margin Under Data Overlap
- **Hard-Margin Formulation**: Constrains all training points to lie strictly on or outside the margin boundaries ($y_i(\mathbf{w}^T\mathbf{x}_i + b) \ge 1$). In real-world noisy regimes with overlapping class distributions, no separating hyperplane exists, or the margin collapses towards zero, rendering the model hypersensitive to outliers.
- **Soft-Margin Formulation**: Introduces slack variables $\xi_i = \max(0, 1 - y_i f(\mathbf{x}_i))$ to tolerate controlled margin penetrations and misclassifications, preserving a broad geometric margin and robust decision stability.

---

## Effect of Regularization Parameter C (Task a)

An empirical sweep across 7 orders of magnitude ($C \in [0.001, 0.01, 0.1, 1, 10, 100, 1000]$) was evaluated on the standardized 2D synthetic dataset ($N_{\text{tr}} = 350, N_{\text{te}} = 150$) with a fixed linear kernel.

### Empirical Parameter C Benchmark Table

| Parameter $C$ | Train Accuracy (%) | Test Accuracy (%) | Test F1-Score (%) | Geometric Margin ($2/\Vert\mathbf{w}\Vert$) | Weight Norm $\Vert\mathbf{w}\Vert_2$ | Support Vectors ($N_{\text{SV}}$) | Sparsity Ratio (%) | Margin Violations ($\xi_i > 0$) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0.001** | 51.43 | 50.00 | 66.67 | **8.185** | 0.244 | 348 | 99.43 | 348 |
| **0.010** | 86.29 | 84.00 | 86.21 | 2.103 | 0.951 | 248 | 70.86 | 248 |
| **0.100** | 89.14 | 90.67 | 91.46 | 1.114 | 1.795 | 148 | 42.29 | 146 |
| **1.000** | **90.00** | **92.00** | **92.59** | 0.831 | 2.407 | 119 | 34.00 | 117 |
| **10.000** | **90.00** | **92.00** | **92.59** | 0.799 | 2.503 | 114 | 32.57 | 112 |
| **100.000** | **90.00** | **92.00** | **92.59** | 0.795 | 2.516 | **112** | **32.00** | 110 |
| **1000.000** | **90.00** | **92.00** | **92.59** | **0.795** | 2.516 | **112** | **32.00** | 110 |

### Analytical Insights:
1. **Margin Width Contraction**: As $C$ increases from $0.001$ to $1000$, the margin width contracts by **$10.3\times$** (from $8.185$ down to $0.795$), while the weight norm $\Vert \mathbf{w} \Vert_2$ grows from $0.244$ to $2.516$.
2. **Support Vector Dynamics**: At $C = 0.001$, virtually all training samples ($348/350 = 99.43\%$) violate the margin condition and serve as support vectors. As $C \ge 1.0$, the support vector set stabilizes at $\sim 32\%$ sparsity ($112$ active vectors).
3. **Underfitting vs. Overfitting Regimes**:
   - $C \le 0.01$ causes severe underfitting: the margin is excessively wide, and the boundary fails to separate classes effectively ($50.00\% - 84.00\%$ accuracy).
   - $C \in [1.0, 10.0]$ represents the optimal trade-off zone, balancing margin stability with classification fidelity ($92.00\%$ test accuracy).
   - $C \ge 100.0$ approaches the asymptotic hard-margin limit where the boundary becomes rigid.

---

## Effect of RBF Kernel Parameter Gamma (Task b)

An empirical sweep across 6 values of $\gamma \in [0.001, 0.01, 0.1, 1, 10, 100]$ was performed using an RBF SVM with fixed $C = 1.0$.

### Empirical RBF Gamma Benchmark Table

| Parameter $\gamma$ | Effective Radius $\sigma$ | Train Accuracy (%) | Test Accuracy (%) | Test F1-Score (%) | Support Vectors ($N_{\text{SV}}$) | Sparsity Ratio (%) | Boundary Regime |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **0.001** | 22.361 | 74.29 | 74.67 | 79.79 | 342 | 97.71 | Severe Underfitting (Flat bell) |
| **0.010** | 7.071 | 86.86 | 84.00 | 86.21 | 211 | 60.29 | High Bias (Pseudo-linear) |
| **0.100** | 2.236 | 89.71 | 90.67 | 91.46 | 130 | 37.14 | Moderate Curvature |
| **1.000** | **0.707** | **93.43** | **95.33** | **95.54** | **94** | **26.86** | **Optimal Generalization Sweet Spot** |
| **10.000** | 0.224 | 97.14 | **96.67** | **96.77** | 132 | 37.71 | High Variance (Localized pockets) |
| **100.000** | 0.071 | **99.14** | **96.67** | 96.73 | 252 | 72.00 | Severe Overfitting (Isolated islands) |

### Analytical Insights:
1. **Effective Radius Progression**: As $\gamma$ increases from $0.001$ to $100.0$, the Gaussian kernel influence radius $\sigma = 1/\sqrt{2\gamma}$ drops precipitously from $22.36$ down to $0.07$.
2. **Sparsity "U-Curve"**: Support vector count follows a distinct U-shaped trajectory:
   - At $\gamma = 0.001$, the model requires $342$ support vectors ($97.7\%$) due to excessive margin width.
   - At $\gamma = 1.0$, support vector count reaches its global minimum of **$94$ vectors** ($26.86\%$ sparsity), achieving maximum geometric compression.
   - At $\gamma = 100.0$, support vector count balloons back to **$252$ vectors** ($72.00\%$) as individual training points are memorized as isolated Gaussian kernels.
3. **Generalization Sweet Spot**: $\gamma^* = 1.0$ is the optimal operating point, maximizing test accuracy ($95.33\%$) while preserving boundary smoothness and structural sparsity.

---

## Hard-Margin vs. Soft-Margin Under Overlapping Noise (Task c)

To rigorously evaluate soft-margin tolerance against hard-margin brittleness, a synthetic dataset with substantial class overlap was synthesized via `make_blobs(n_samples=300, centers=2, cluster_std=1.5, random_state=42)`. An RBF SVM was evaluated under two contrasting regimes:
- **Soft-Margin SVM ($C = 1.0$)**: High slack tolerance.
- **Hard-Margin SVM ($C = 1000.0$)**: Severe slack penalization.

### Empirical Hard-Margin vs. Soft-Margin Benchmark Table

| Model Regime | Regularization $C$ | Train Accuracy (%) | Test Accuracy (%) | Margin Width ($2/\Vert\mathbf{w}\Vert$) | Weight Norm $\Vert\mathbf{w}\Vert_2$ | Support Vectors ($N_{\text{SV}}$) | Sparsity (%) | Margin Violations ($\xi_i > 0$) | Misclassified Points |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Soft-Margin SVM** | 1.0 | 91.43 | 91.11 | **0.849** | **2.357** | 48 | 22.86 | 46 | 18 |
| **Hard-Margin SVM** | 1000.0 | 91.43 | 91.11 | **0.765** | **2.616** | 42 | 20.00 | 41 | 18 |

### Analytical Insights:
1. **Margin Geometric Contraction**: In the presence of overlapping clusters, increasing $C$ from $1$ to $1000$ forces the margin width to shrink from $0.849$ down to $0.765$ (a **$9.9\%$ contraction**), driving $\Vert \mathbf{w} \Vert_2$ up from $2.357$ to $2.616$.
2. **Slack Tolerance**: Soft-margin ($C=1$) tolerates $46$ margin penetrations ($\xi_i > 0$), resulting in a wider, more robust buffer zone that absorbs stochastic noise. Hard-margin ($C=1000$) aggressively penalizes slack, driving the boundary closer to the noisy interface.
3. **Generalization Safeguard**: When overlapping distributions make perfect linear or kernel separation mathematically impossible, soft-margin regularization prevents the optimization solver from producing an erratic, ill-conditioned boundary.

---

## Hyperparameter Optimization via GridSearchCV (Task d)

A systematic $5 \times 5$ grid search was executed over the full Cartesian product of candidate hyperparameters:
- $C \in [0.1, 1.0, 10.0, 100.0, 1000.0]$
- $\gamma \in [0.001, 0.01, 0.1, 1.0, 10.0]$
- **Cross-Validation Protocol**: Stratified 5-Fold Cross-Validation (25 total candidate models, 125 total fit evaluations).

### Benchmark Results: Baseline vs. Optimized Estimator

| Model Configuration | Parameter $C^*$ | Parameter $\gamma^*$ | 5-Fold CV Accuracy (%) | Test Accuracy (%) | Test Precision (%) | Test Recall (%) | Test F1-Score (%) | Support Vectors ($N_{\text{SV}}$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline Default RBF** | 1.0 | `scale` (0.50) | 95.71 | 94.67 | 90.36 | **100.00** | 94.94 | **101** |
| **Optimized RBF (`GridSearchCV`)** | **10.0** | **10.0** | **97.43** | **96.67** | **93.75** | **100.00** | **96.77** | 102 |

### Analytical Insights:
1. **Performance Lift**: Grid search optimization elevates 5-fold cross-validation accuracy from $95.71\%$ to **$97.43\%$** (+1.72% gain) and test accuracy from $94.67\%$ to **$96.67\%$** (+2.00% gain).
2. **Precision Enhancement**: Precision increases from $90.36\%$ to **$93.75\%$** while maintaining a flawless **$100.00\%$ recall**, producing a superior F1-score of **$96.77\%$**.
3. **Bias-Variance Convergence (Learning Curve)**: The learning curve diagnostic confirms that as training samples increase from $35$ to $280$, the training score converges stably towards the cross-validation score (~97%), demonstrating healthy generalization without high-variance divergence.

---

## Visual Diagnostics and Parameter Influence Dashboard

The notebook `Study_SVM_Parameters.ipynb` embeds 6 high-DPI base64 visual figures:

1. **Figure 1: Regularization Parameter C Influence Curves**:
   - 3-panel horizontal diagnostic displaying:
     - Panel A: Train Accuracy vs. Test Accuracy across $\log_{10}(C)$, clearly marking the underfitting zone ($C < 0.1$).
     - Panel B: Geometric Margin Width $2/\Vert\mathbf{w}\Vert_2$ contracting from $8.18$ down to $0.79$.
     - Panel C: Support Vector count dropping from $348$ down to $112$.
2. **Figure 2: Empirical Decision Boundaries Across Parameter C**:
   - 4-panel visual grid ($C = 0.001, 0.1, 1.0, 100.0$) displaying decision surfaces ($f(\mathbf{x}) = 0$), dashed margin bounds ($f(\mathbf{x}) = \pm 1$), and highlighted support vectors demonstrating margin tightening.
3. **Figure 3: RBF Decision Boundary Complexity Progression Across Gamma**:
   - 6-panel $2 \times 3$ visual grid ($\gamma \in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]$) illustrating the transition from flat linear underfitting ($\gamma=0.001$) to smooth optimal boundaries ($\gamma=1.0$) and fragmented island memorization ($\gamma=100.0$).
4. **Figure 4: RBF Kernel Gamma Validation Curve**:
   - Single-panel plot displaying mean train and 5-fold cross-validation accuracy with shaded $\pm 1\sigma$ variance error bands, highlighting the optimal sweet spot at $\gamma^* = 1.0$.
5. **Figure 5: Soft-Margin vs. Hard-Margin Comparative Diagnostics on Noisy Blobs**:
   - 2-panel comparative layout:
     - Panel A: Soft-Margin ($C=1$) with wide margin ($M=0.849$), tolerating 46 slack margin violations without boundary distortion.
     - Panel B: Hard-Margin ($C=1000$) with contracted margin ($M=0.765$), showing rigid boundary adaptation around noisy border points.
6. **Figure 6: Hyperparameter Optimization Diagnostic Dashboard**:
   - Panel A: 2D 5-Fold Cross-Validation Accuracy Heatmap across the $5 \times 5$ grid of $(C, \gamma)$, pinpointing the global optimum $(C=10, \gamma=10)$.
   - Panel B: Bias-Variance Learning Curve plotting train and CV scores across expanding training set sizes ($N=35$ to $280$).

---

## Comprehensive Hyperparameter Engineering Matrix

| Hyperparameter | Low Value Regime | High Value Regime | Optimal Range | Diagnostic Signature of Failure |
| :--- | :--- | :--- | :---: | :--- |
| **Regularization $C$** | Wide margin, high slack tolerance, high bias (underfitting). | Narrow margin, low slack tolerance, high variance (overfitting). | $0.1 \le C \le 10$ | Large gap between train and test score indicates $C$ is too high; low train and test scores indicate $C$ is too low. |
| **RBF Kernel $\gamma$** | Broad Gaussian radius $\sigma$, smooth flat boundary, high bias. | Sharp Gaussian radius $\sigma$, fragmented boundary islands, high variance. | $0.1 \le \gamma \le 1.0$ (or $1/(D \cdot \text{Var}(X))$) | Fragmented island contours or exploding support vector counts indicate $\gamma$ is too high. |
| **Optimization Method** | Manual heuristic parameter guessing. | Systematic exhaustive search over logarithmic grid via `GridSearchCV`. | Log-space grid (e.g., $[10^{-3}, 10^3]$) | Grid edges containing the best score indicate search range must be expanded. |

---

## Execution and Reproduction Guide

### Dependencies
Ensure the following packages are installed:
```bash
pip install numpy pandas matplotlib scikit-learn
```

### Reproducing the Experiments

#### Problem 1: Comparison of Linear and Non-linear SVM
```bash
jupyter notebook Linear_vs_Nonlinear_SVM.ipynb
```
Key verification checkpoints:
- **Cell 4**: Generates synthetic data and loads Iris, confirming stratified partitions.
- **Cell 5**: Renders Figure 1 (Exploratory Data Analysis).
- **Cell 7**: Trains and outputs benchmark table for synthetic data.
- **Cell 8**: Renders Figure 2 (Synthetic Decision Boundaries & Margins).
- **Cell 10**: Trains and outputs benchmark table for Iris data.
- **Cell 11**: Renders Figure 3 (Iris Decision Boundaries & Margins).
- **Cell 13**: Renders Figure 4 (Multi-Metric Diagnostics Dashboard).
- **Cell 15**: Renders Figure 5 (3D Kernel Trick Transformation).

#### Problem 2: Study on the Effect of SVM Parameters
```bash
jupyter notebook Study_SVM_Parameters.ipynb
```
Key verification checkpoints:
- **Cell 2**: Generates 2D synthetic dataset ($N=500$) with stratified 70/30 split.
- **Cell 3**: Runs parameter $C$ sweep across $[0.001, 1000]$ and prints quantitative benchmark table.
- **Cell 4**: Renders Figure 1 (3-panel curves: Accuracy, Margin Width, Support Vectors vs. $\log_{10}(C)$).
- **Cell 5**: Renders Figure 2 (4-panel visual decision boundaries across $C$).
- **Cell 7**: Runs RBF parameter $\gamma$ sweep across $[0.001, 100]$ and prints benchmark table.
- **Cell 8**: Renders Figure 3 (6-panel RBF boundary complexity progression).
- **Cell 9**: Computes and renders Figure 4 (Validation Curve for $\gamma$ with $\pm 1\sigma$ bands).
- **Cell 11**: Synthesizes noisy overlapping blobs (`cluster_std=1.5`) and prints Hard vs. Soft margin benchmark.
- **Cell 12**: Renders Figure 5 (Visual comparison of Soft $C=1$ vs. Hard $C=1000$ margin violations).
- **Cell 14**: Executes 5-fold `GridSearchCV` over $(C, \gamma)$ and prints Baseline vs. Optimized benchmark.
- **Cell 15**: Renders Figure 6 (2D Cross-Validation Heatmap and Bias-Variance Learning Curve).

---

## References and Academic Literature

- Vapnik, V. N. (1998). *Statistical Learning Theory*. John Wiley & Sons.
- Cortes, C., & Vapnik, V. (1995). Support-vector networks. *Machine Learning*, 20(3), 273-297.
- Scholkopf, B., & Smola, A. J. (2002). *Learning with Kernels: Support Vector Machines, Regularization, Optimization, and Beyond*. MIT Press.
- Chang, C. C., & Lin, C. J. (2011). LIBSVM: A library for support vector machines. *ACM Transactions on Intelligent Systems and Technology*, 2(3), 1-27.
- Fan, R. E., Chang, K. W., Hsieh, C. J., Wang, X. R., & Lin, C. J. (2008). LIBLINEAR: A library for large linear classification. *Journal of Machine Learning Research*, 9, 1871-1874.
- Course Lecture Notes: `SVM_Lab.pdf` (HCMUT Machine Learning Coursework).
