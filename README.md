# HCMUT — Machine Learning Hub

<div align="center">

![University](https://img.shields.io/badge/University-HCMUT-0052CC?style=for-the-badge)
![Course](https://img.shields.io/badge/Course-Machine_Learning-FF6F00?style=for-the-badge)
![Semester](https://img.shields.io/badge/Semester-Fall_2026-green?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Active_Development-brightgreen?style=for-the-badge)

<p align="center">
  A curated repository of <b>Machine Learning Labs and Coursework</b> at <b>Ho Chi Minh City University of Technology (HCMUT)</b>.<br>
  Combining rigorous theoretical mathematical formulations with clean, reproducible Scikit-Learn and PyTorch implementations.
</p>

---

### Tech Stack & Core Libraries

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat-square)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4+-F7931E?style=flat-square)
![NumPy](https://img.shields.io/badge/NumPy-1.26+-013243?style=flat-square)
![Pandas](https://img.shields.io/badge/Pandas-2.2+-150458?style=flat-square)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.8+-11557c?style=flat-square)
![Jupyter](https://img.shields.io/badge/Jupyter-Lab%20%2F%20Notebook-F37626?style=flat-square)

</div>

---

## Repository Overview

This repository serves as a comprehensive **Machine Learning Hub**. Each topic contains:

- **Problem Specification (PDF)**: Official assignment requirements and hints.
- **Self-contained Notebooks**: Theory presentation with LaTeX mathematics, step-by-step EDA, model training, evaluation, and visualization.
- **Dedicated Module README**: Summary of objectives, tasks, models, and reference links.

|   #    | Topic / Module                                                     | Notebooks                                                                                                                                                                                                                                                                                           | Dataset                                   | Key Techniques & Algorithms                                                                 |   Status    |
| :----: | :----------------------------------------------------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :---------------------------------------- | :------------------------------------------------------------------------------------------ | :---------: |
| **01** | [**Decision Tree**](./Decision_Tree/)                              | [`DT_Classification.ipynb`](./Decision_Tree/DT_Classification.ipynb)<br>[`DT_Regression.ipynb`](./Decision_Tree/DT_Regression.ipynb)                                                                                                                                                                | Iris<br>Diabetes                          | ID3 (Entropy), Gini (CART), Log-Loss, Pruning, From-scratch implementation, `SimpleImputer` | `Completed` |
| **02** | [**Linear Models & Regression**](./ANN-MLP/Linear_Model/)          | [`linear_regresion.ipynb`](./ANN-MLP/Linear_Model/linear_regresion.ipynb)<br>[`logistic_regression_binary_classification.ipynb`](./ANN-MLP/Linear_Model/logistic_regression_binary_classification.ipynb)<br>[`multi-output_regression.ipynb`](./ANN-MLP/Linear_Model/multi-output_regression.ipynb) | Diabetes<br>Breast Cancer<br>Linnerud     | OLS, Ridge ($L_2$), LASSO ($L_1$), Logistic Regression, ROC-AUC, `MultiOutputRegressor`     | `Completed` |
| **03** | [**ANN Optimizers & Training**](./ANN-MLP/Training_ANN-Optimizer/) | [`Optimization_Algorithm.ipynb`](./ANN-MLP/Training_ANN-Optimizer/Optimization_Algorithm.ipynb)<br>[`Normalization_Regularization_MLP.ipynb`](./ANN-MLP/Training_ANN-Optimizer/Normalization_Regularization_MLP.ipynb)                                                                              | Breast Cancer<br>Digits (Handwritten 0-9) | SGD, Momentum, Adam, AdamW, Cosine Annealing, Batch Normalization, Dropout, L2 Decay        | `Completed` |

---

## Detailed Lab Modules

### Module 1 — Decision Trees

<details open>
<summary><b>Click to expand Module 1 details</b></summary>

**Folder**: [`Decision_Tree/`](./Decision_Tree/) | **Specification**: [`DecisionTree_Programming_EL.pdf`](./Decision_Tree/DecisionTree_Programming_EL.pdf)

- **Problem 1 — Decision Tree Classification** ([`DT_Classification.ipynb`](./Decision_Tree/DT_Classification.ipynb))
  - **Dataset**: Iris Dataset (150 samples, 4 features, 3 classes).
  - **Algorithms**: `DecisionTreeClassifier` with Entropy (ID3), Gini Impurity (CART), and Log Loss.
  - **From-scratch Implementation**: Information Gain calculation, recursive splitting, and tree inference.
  - **Evaluation**: Cross-validation, tree boundary visualization, and accuracy comparison.

- **Problem 2 — Decision Tree Regression & Missing Data** ([`DT_Regression.ipynb`](./Decision_Tree/DT_Regression.ipynb))
  - **Dataset**: Diabetes Dataset (442 samples, 10 features).
  - **Algorithms**: `DecisionTreeRegressor` with squared error criterion; hyperparameter tuning across tree depths ($d \in [1, 10]$).
  - **Missing Value Handling**: Randomly simulates 10% missing values; benchmarks Mean Imputation vs. `SimpleImputer`.
  - **Evaluation**: MAE and RMSE across preprocessing strategies.

</details>

---

### Module 2 — Linear Models & Multi-Output Regression

<details open>
<summary><b>Click to expand Module 2 details</b></summary>

**Folder**: [`ANN-MLP/Linear_Model/`](./ANN-MLP/Linear_Model/) | **Specification**: [`MLP_Programming_HW_EL.pdf`](./ANN-MLP/Linear_Model/MLP_Programming_HW_EL.pdf) | **Documentation**: [`README.md`](./ANN-MLP/Linear_Model/README.md)

- **Problem 1 — Comparing Linear Regression, Ridge, and LASSO** ([`linear_regresion.ipynb`](./ANN-MLP/Linear_Model/linear_regresion.ipynb))
  - **Dataset**: Scikit-learn Diabetes Dataset (442 samples, 10 physiological predictors).
  - **Concepts**: Ordinary Least Squares (OLS), Ridge ($L_2$ shrinkage), and LASSO ($L_1$ sparsity).
  - **Highlights**: Multicollinearity diagnosis in serum variables (`s1` vs. `s2`), automated feature elimination (LASSO zeroes out 3 features), regularization parameter sweep ($\alpha \in [10^{-3}, 10^2]$).
  - **Best Metric**: LASSO ($\alpha = 0.1$) achieves minimum $\text{Test MSE} = 2775.17$ with 7/10 active features.

- **Problem 2 — Binary Classification with Logistic Regression** ([`logistic_regression_binary_classification.ipynb`](./ANN-MLP/Linear_Model/logistic_regression_binary_classification.ipynb))
  - **Dataset**: Breast Cancer Wisconsin Diagnostic (569 samples, 30 features).
  - **Concepts**: Sigmoid formulation, Log-loss, inverse regularization hyperparameter $C$, data leakage prevention via `StandardScaler`.
  - **Highlights**: ROC Curve ($\text{AUC} = 0.9981$), confusion matrix, regularization paths comparing $L_1$ vs. $L_2$ over $C \in [10^{-3}, 10^2]$.
  - **Best Metric**: $L_2$ at $C = 1.0$ achieves $\text{Accuracy} = 98.83\%$; $L_1$ at $C = 1.0$ achieves $\text{Accuracy} = 97.66\%$ using only 14/30 features (53% compression).

- **Problem 3 — Multi-tasking with Multi-output Regression** ([`multi-output_regression.ipynb`](./ANN-MLP/Linear_Model/multi-output_regression.ipynb))
  - **Dataset**: Linnerud Dataset (20 samples, 3 exercise inputs $\implies$ 3 physiological targets: `Weight`, `Waist`, `Pulse`).
  - **Concepts**: Frobenius norm multi-task objective, correlation matrix heatmap, architectural equivalence proof.
  - **Highlights**: Mathematical and empirical verification that `MultiOutputRegressor(LinearRegression())` and standalone 1D regressors yield identical predictions ($\Delta < 10^{-14}$); Ridge regression effectively mitigates high variance on small sample size.
  - **Best Metric**: Multi-output Ridge improves $R^2$ and lowers RMSE across all 3 targets simultaneously.

</details>

---

### Module 3 — Neural Network Optimization, Normalization & Regularization

<details open>
<summary><b>Click to expand Module 3 details</b></summary>

**Folder**: [`ANN-MLP/Training_ANN-Optimizer/`](./ANN-MLP/Training_ANN-Optimizer/) | **Specification**: [`ANN_Part2_Programming_HW_EL.pdf`](./ANN-MLP/Training_ANN-Optimizer/ANN_Part2_Programming_HW_EL.pdf) | **Documentation**: [`README.md`](./ANN-MLP/Training_ANN-Optimizer/README.md)

- **Problem 1 — Comparing Optimization Algorithms for MLPClassifier** ([`Optimization_Algorithm.ipynb`](./ANN-MLP/Training_ANN-Optimizer/Optimization_Algorithm.ipynb))
  - **Dataset**: Breast Cancer Wisconsin Diagnostic (569 samples, 30 features).
  - **Optimizers**: Vanilla SGD, SGD with Momentum ($\mu = 0.9$), Adam (Adaptive Moment Estimation), and AdamW (Decoupled Weight Decay).
  - **Learning Rate Schedules**: Constant Learning Rate vs. Cosine Annealing schedule on SGD.
  - **Highlights**: Convergence trajectory comparison, wall-clock time vs. iteration counts, and empirical superiority of decoupled weight decay.
  - **Best Metric**: AdamW achieves $\text{Accuracy} = 97.08\%$, $\text{Precision} = 99.04\%$, and $\text{F1} = 0.9763$.

- **Problem 2 — Impact of Normalization and Regularization Techniques** ([`Normalization_Regularization_MLP.ipynb`](./ANN-MLP/Training_ANN-Optimizer/Normalization_Regularization_MLP.ipynb))
  - **Dataset**: Scikit-learn Digits Dataset (1,797 samples, 64 pixel features, 10 handwritten digit classes).
  - **Architectural Design**: 2-hidden layer MLP ($64 \to 128 \to 64 \to 10$) with ReLU activations and Cross-Entropy Loss.
  - **Benchmarking Configurations**:
    1. Baseline (Unnormalized raw pixels, unregularized)
    2. Batch Normalization only (`nn.BatchNorm1d`)
    3. Dropout only ($p = 0.3$)
    4. $L_2$ Regularization only (weight decay $\lambda = 0.001$)
    5. Combined Architecture (BatchNorm + Dropout + $L_2$)
  - **Best Metric**: BatchNorm achieves $\text{Test Accuracy} = 99.17\%$ and $\text{Test Log-Loss} = 0.0388$; Combined model achieves lowest Generalization Gap ($\Delta_{\text{loss}} = 0.0197$).

</details>

---

## Directory Structure

```text
Machine-Learning-Labs/
│
├── Decision_Tree/
│   ├── DecisionTree_Programming_EL.pdf                 # Problem specification
│   ├── DT_Classification.ipynb                         # Iris classification & from-scratch tree
│   ├── DT_Regression.ipynb                             # Diabetes regression & missing data handling
│   └── README.md                                       # Decision Tree module documentation
│
├── ANN-MLP/
│   ├── Linear_Model/
│   │   ├── MLP_Programming_HW_EL.pdf                   # Problem specification
│   │   ├── linear_regresion.ipynb                      # Problem 1: Linear, Ridge, and LASSO
│   │   ├── logistic_regression_binary_classification.ipynb # Problem 2: Logistic Regression (L1 vs L2)
│   │   ├── multi-output_regression.ipynb               # Problem 3: Multi-output Regression (Linnerud)
│   │   └── README.md                                   # Linear Models module documentation
│   │
│   ├── Training_ANN-Optimizer/
│   │   ├── ANN_Part2_Programming_HW_EL.pdf             # Problem specification
│   │   ├── Optimization_Algorithm.ipynb                # Problem 1: SGD, Momentum, Adam, AdamW
│   │   ├── Normalization_Regularization_MLP.ipynb       # Problem 2: Normalization & Regularization
│   │   └── README.md                                   # ANN Optimizers module documentation
│   │
│   └── README.md                                       # ANN-MLP module hub documentation
│
├── agents/
│   └── rules.md                                        # Working principles and development guidelines
│
├── requirements.txt                                    # Python environment dependencies
└── README.md                                           # Root repository hub documentation
```

---

## Getting Started

### Prerequisites

- Python `3.10` or higher
- `pip` package manager

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
```

### 2. Set Up a Virtual Environment

```bash
# Windows
python -m venv .env
.env\Scripts\activate

# Linux / macOS
python3 -m venv .env
source .env/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Launch Jupyter Notebook

```bash
jupyter notebook
```

---

## References & Lecture Materials

- **Scikit-learn Documentation**: [scikit-learn.org](https://scikit-learn.org/)
- **PyTorch Documentation**: [pytorch.org](https://pytorch.org/)
- **Course Lecture Slides**:
  - `1. ML-Introduction.pdf`
  - `2. Decision_Tree.pdf`
  - `3. LinearRegression.pdf`
  - `LogisticRegression.pdf`
  - `MLP.pdf`
- **Book Reference**: _Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow_ by Aurélien Géron.

---

<div align="center">
  <sub>HCMUT Machine Learning Coursework • Maintained with standard coding practices and mathematical rigor.</sub>
</div>
