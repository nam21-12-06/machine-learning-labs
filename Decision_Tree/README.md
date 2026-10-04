# Decision Trees with Scikit-learn

This module explores **Decision Tree algorithms** for both classification and regression tasks using `scikit-learn` and custom implementations from scratch.

---

## Module Overview & Status

| Problem | Notebook                                               | Dataset                             | Key Techniques                                          |   Status    |
| :-----: | :----------------------------------------------------- | :---------------------------------- | :------------------------------------------------------ | :---------: |
|  **1**  | [`DT_Classification.ipynb`](./DT_Classification.ipynb) | Iris (150 samples, 4 features)      | ID3 (Entropy), CART (Gini), Log-Loss, From-Scratch Tree | `Completed` |
|  **2**  | [`DT_Regression.ipynb`](./DT_Regression.ipynb)         | Diabetes (442 samples, 10 features) | Squared Error, Depth Tuning, Missing Value Imputation   | `Completed` |

---

# Problem 1 — Decision Tree Classification

## Objective

Apply Decision Tree classification on the **Iris dataset** and compare different splitting criteria.

## Tasks


* Prepare and preprocess the Iris dataset.
* Train `DecisionTreeClassifier` using:
  - Entropy (ID3)
  - Log Loss
  - Gini Impurity (CART)
* Compare model performance and decision boundaries.
* Visualize and interpret the resulting decision tree architectures.
* Evaluate models using cross-validation and test accuracy.
## Optional — From Scratch Implementation

Implement a basic Decision Tree classifier **from scratch** without relying on `DecisionTreeClassifier`:

- Entropy and Information Gain calculation.
- Optimal split threshold search across continuous features.
- Recursive tree construction and terminal node assignment.
- Inference pipeline using the custom constructed tree.
- Validation and benchmark against the `scikit-learn` implementation.

---

# Problem 2 — Decision Tree Regression and Handling Missing Values

## Objective

Apply Decision Tree Regression on the **Diabetes dataset** and investigate the effect of tree depth and missing-value handling on model performance.

## Tasks

* Train `DecisionTreeRegressor` using the squared error criterion.
* Experiment with hyperparameter tuning across tree depths ($d \in [1, 10]$).
* Evaluate performance using MAE (Mean Absolute Error) and RMSE (Root Mean Squared Error).
* Simulate missing data by randomly removing **10% of the values in one feature**.
* Benchmark missing value imputation strategies:
  - Mean imputation
  - `SimpleImputer` from `scikit-learn`
* Compare regression results and analyze the impact of missing data handling on model generalization.
---

# References

* [Scikit-learn Decision Trees Documentation](https://scikit-learn.org/stable/modules/tree.html)
* Course Lecture Slides: `2. Decision_Tree.pdf`
