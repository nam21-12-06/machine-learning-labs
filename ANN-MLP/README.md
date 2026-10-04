# Artificial Neural Networks & Multi-Layer Perceptrons (ANN-MLP)

This directory contains laboratory coursework on **Linear Models, Multi-Output Regression, and Multi-Layer Perceptron (MLP) Training Dynamics** using Scikit-Learn and PyTorch.

---

## Sub-Modules Overview

```text
ANN-MLP/
├── Linear_Model/                  # Linear, Ridge, LASSO, Logistic & Multi-output
│   ├── MLP_Programming_HW_EL.pdf
│   ├── linear_regresion.ipynb
│   ├── logistic_regression_binary_classification.ipynb
│   ├── multi-output_regression.ipynb
│   └── README.md
│
└── Training_ANN-Optimizer/        # Optimizers, BatchNorm, Dropout & L2 Regularization
    ├── ANN_Part2_Programming_HW_EL.pdf
    ├── Optimization_Algorithm.ipynb
    ├── Normalization_Regularization_MLP.ipynb
    └── README.md
```

---

## Progress and Summary of Labs

| Module | Problem | Notebook | Dataset | Key Techniques | Status |
| :--- | :---: | :--- | :--- | :--- | :---: |
| **Linear Models** | **1** | [`linear_regresion.ipynb`](./Linear_Model/linear_regresion.ipynb) | Diabetes (442 samples, 10 features) | OLS, Ridge ($L_2$), LASSO ($L_1$), Alpha sweep | `Completed` |
| **Linear Models** | **2** | [`logistic_regression_binary_classification.ipynb`](./Linear_Model/logistic_regression_binary_classification.ipynb) | Breast Cancer (569 samples, 30 features) | Logistic Regression, $L_1$ vs $L_2$, ROC-AUC, $C$ sweep | `Completed` |
| **Linear Models** | **3** | [`multi-output_regression.ipynb`](./Linear_Model/multi-output_regression.ipynb) | Linnerud (20 samples, 3 targets) | `MultiOutputRegressor`, Decoupled OLS, Ridge | `Completed` |
| **ANN Optimizers** | **1** | [`Optimization_Algorithm.ipynb`](./Training_ANN-Optimizer/Optimization_Algorithm.ipynb) | Breast Cancer (569 samples, 30 features) | SGD, Momentum, Adam, AdamW, Cosine Annealing | `Completed` |
| **ANN Optimizers** | **2** | [`Normalization_Regularization_MLP.ipynb`](./Training_ANN-Optimizer/Normalization_Regularization_MLP.ipynb) | Digits (1,797 samples, 10 classes) | BatchNorm, Dropout, $L_2$ Decay, Combined Network | `Completed` |

---

## Module Details

### 1. [Linear Models & Multi-Output Regression](./Linear_Model/)
- **Specification**: [`MLP_Programming_HW_EL.pdf`](./Linear_Model/MLP_Programming_HW_EL.pdf)
- **Topics**:
  - Regularization trade-offs: $L_2$ parameter shrinkage vs. $L_1$ feature sparsity.
  - Multicollinearity resolution in physiological data.
  - Probability calibration and ROC discrimination in binary cancer classification.
  - Multi-task regression and mathematical decoupling of Frobenius norm objectives.

### 2. [Neural Network Training, Optimization & Regularization](./Training_ANN-Optimizer/)
- **Specification**: [`ANN_Part2_Programming_HW_EL.pdf`](./Training_ANN-Optimizer/ANN_Part2_Programming_HW_EL.pdf)
- **Topics**:
  - Convergence acceleration: First-order SGD vs. Momentum vs. Adam vs. AdamW.
  - Decoupled weight decay dynamics for adaptive optimizers.
  - Learning rate decay: Constant schedule vs. Cosine Annealing.
  - Internal covariate shift stabilization via Batch Normalization (`nn.BatchNorm1d`).
  - Co-adaptation mitigation via Dropout ($p = 0.3$).
  - Overfitting reduction and generalization gap analysis.

---

## References

- [Scikit-learn Documentation](https://scikit-learn.org/)
- [PyTorch Documentation](https://pytorch.org/)
- Course Lecture Slides: `3. LinearRegression.pdf`, `LogisticRegression.pdf`, `MLP.pdf`
