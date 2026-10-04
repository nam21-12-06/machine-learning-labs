# Linear Models and Multi-output Regression with Scikit-learn

This module explores **linear models for regression and classification**, including regularization techniques ($L_1$ and $L_2$) and multi-output regression using standard datasets from `scikit-learn`.

---

## Module Overview & Status

| Problem | Notebook                                                                                               | Dataset                                  | Key Techniques                                          |   Status    |
| :-----: | :----------------------------------------------------------------------------------------------------- | :--------------------------------------- | :------------------------------------------------------ | :---------: |
|  **1**  | [`linear_regresion.ipynb`](./linear_regresion.ipynb)                                                   | Diabetes (442 samples, 10 features)      | OLS, Ridge ($L_2$), LASSO ($L_1$), Alpha sweep          | `Completed` |
|  **2**  | [`logistic_regression_binary_classification.ipynb`](./logistic_regression_binary_classification.ipynb) | Breast Cancer (569 samples, 30 features) | Logistic Regression, $L_1$ vs $L_2$, ROC-AUC, $C$ sweep | `Completed` |
|  **3**  | [`multi-output_regression.ipynb`](./multi-output_regression.ipynb)                                     | Linnerud (20 samples, 3 targets)         | `MultiOutputRegressor`, Decoupled OLS, Ridge            | `Completed` |

---

# Problem 1 — Comparing Linear Regression, Ridge, and LASSO

## Objective

Apply Linear Regression, Ridge, and LASSO on the **Diabetes dataset** to investigate the effect of regularization on predictive performance, coefficient shrinkage, and feature selection.

## Tasks

- Prepare and inspect the Diabetes dataset (distributions and summary statistics).
- Split the dataset into training and testing sets (70/30 ratio).
- Train three baseline models:
  - Ordinary Linear Regression (OLS)
  - Ridge Regression ($\alpha = 0.1$)
  - LASSO Regression ($\alpha = 0.1$)
- Evaluate performance using Mean Squared Error (MSE) on the test set.
- Compare the number of selected features (non-zero weights) across models.
- Visualize and compare the learned weights (coefficients) using plots and comparison tables.
- Experiment with different regularization strengths ($\alpha \in [10^{-3}, 10^2]$) and analyze their impact on prediction error and sparsity.


## Summary of Results

  | Model | Penalty | Hyperparameter | Test MSE | Active Features | Status |
  | :--- | :---: | :---: | :---: | :---: | :---: |
  | **OLS Baseline** | None | - | 2821.75 | 10/10 | Completed |
  | **Ridge Regression** | $L_2$ | $\alpha = 0.1$ | 2805.40 | 10/10 | Completed |
  | **LASSO Regression** | $L_1$ | $\alpha = 0.1$ | **2775.17** | **7/10** | Completed |

- **Multicollinearity**: OLS suffers from large opposing weights on collinear features `s1` ($-901.96$) and `s2` ($+506.76$). Ridge shrinks these to $-108.81$ and $-70.58$.
- **Feature Selection**: LASSO ($\alpha = 0.1$) achieves the lowest test error while eliminating 3 redundant features (`age`, `s2`, `s4`).

---

# Problem 2 — Binary Classification with Logistic Regression

## Objective

Apply Logistic Regression on the **Breast Cancer Wisconsin dataset** to evaluate classification performance, analyze the effect of inverse regularization strength ($C$), and compare $L_1$ versus $L_2$ penalties.

## Tasks

- Load and explore the Breast Cancer dataset (class distributions and feature summaries).
- Partition the data into training (70%) and testing (30%) sets using stratified sampling.
- Perform feature scaling using `StandardScaler` (fitted strictly on the training set to prevent data leakage).
- Train baseline Logistic Regression with $L_2$ penalty ($C = 1.0$) using the `liblinear` solver.
- Evaluate performance using:
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - Classification report and Confusion Matrix
- Plot the Receiver Operating Characteristic (ROC) curve and compute Area Under the Curve (AUC).
- Analyze the effect of the regularization strength hyperparameter ($C \in [10^{-3}, 10^2]$) on:
  - Model accuracy (Train vs. Test to detect underfitting/overfitting)
  - Number of selected features (non-zero weights)
  - Regularization paths (weight shrinkage trajectories)
- Systematically compare $L_1$ versus $L_2$ regularization trade-offs.


## Summary of Results

  | Model | Penalty | Hyperparameter ($C$) | Test Accuracy | Precision | Recall | F1-Score | ROC-AUC | Active Features | Status |
  | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
  | **Logistic Regression ($L_2$)** | $L_2$ | $C = 1.0$ | **98.83%** | 98.15% | 100.00% | 0.9907 | **0.9981** | 30/30 | Completed |
  | **Logistic Regression ($L_1$)** | $L_1$ | $C = 1.0$ | 97.66% | 97.25% | 99.07% | 0.9815 | 0.9950 | **14/30** | Completed |

- **Optimal Regularization**: $C \in [0.1, 1.0]$ delivers the highest generalization without overfitting.
- **Sparsity**: $L_1$ at $C = 1.0$ compresses feature space by 53% (retaining 14 of 30 features) with minimal accuracy penalty ($97.66\%$ vs $98.83\%$).

---

# Problem 3 — Multi-tasking with Multi-output Regression

## Objective

Apply multi-output regression on the **Linnerud dataset** to model multiple continuous physiological target variables simultaneously from exercise measurements.

## Tasks

- Load the Linnerud dataset and explore Pearson correlations among multiple target variables (`Weight`, `Waist`, `Pulse`).
- Split the dataset into training (70%) and testing (30%) sets, followed by feature standardization.
- Train multi-output regression models using:
  - Multi-output Linear Regression (`MultiOutputRegressor(LinearRegression())`)
  - Multi-output Ridge Regression (`MultiOutputRegressor(Ridge(alpha=1.0))`)
- Evaluate performance individually for each target variable using $R^2$ score, RMSE, and MAE.
- Compare multi-output regression results against training individual models for each target variable separately, verifying theoretical equivalence.
- Visualize predicted versus actual values using scatter plots with identity reference lines ($y = x$).

## Summary of Results

  | Target Variable | Linear Regression $R^2$ | Ridge Regression $R^2$ | Linear RMSE | Ridge RMSE | Linear MAE | Ridge MAE |
  | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
  | **Weight** | -0.27 | **0.08** | 25.16 | **21.94** | 19.34 | **17.65** |
  | **Waist** | 0.50 | **0.56** | 1.72 | **1.59** | 1.57 | **1.45** |
  | **Pulse** | -0.56 | **-0.48** | 9.87 | **9.61** | 8.16 | **7.78** |

- **Variance Mitigation**: Ridge regression ($\alpha = 1.0$) outperforms unregularized OLS across all 3 targets on small sample size ($n = 20$).
- **Architectural Equivalence**: Verified zero numerical divergence ($\Delta < 10^{-14}$) between `MultiOutputRegressor` and standalone 1D models.

---

# References

- [Scikit-learn Generalized Linear Models](https://scikit-learn.org/stable/modules/linear_model.html)
- [Scikit-learn Multi-output Regressor Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.multioutput.MultiOutputRegressor.html)
- Course Lecture Slides: `3. LinearRegression.pdf`, `LogisticRegression.pdf`, `MLP.pdf`
