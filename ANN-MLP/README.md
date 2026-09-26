# Linear Models and Multi-output Regression with Scikit-learn

This lab explores **linear models for regression and classification**, including regularization techniques ($L_1$ and $L_2$) and multi-output regression using datasets from `scikit-learn`.

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

## Libraries

- `scikit-learn`
- `pandas`
- `numpy`
- `matplotlib`

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

## Libraries

- `scikit-learn`
- `pandas`
- `numpy`
- `matplotlib`

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

## Libraries

- `scikit-learn`
- `pandas`
- `numpy`
- `matplotlib`

---

# References

- [Scikit-learn Generalized Linear Models](https://scikit-learn.org/stable/modules/linear_model.html)
- [Scikit-learn Multi-output Regressor Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.multioutput.MultiOutputRegressor.html)
- Course Lecture Slides: `3. LinearRegression.pdf`, `LogisticRegression.pdf`, `MLP.pdf`
