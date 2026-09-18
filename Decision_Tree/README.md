# Decision Trees with Scikit-learn

This lab explores **Decision Tree models** for both classification and regression using datasets from `scikit-learn`.

---

# Problem 1 — Decision Tree Classification

## Objective

Apply Decision Tree classification on the **Iris dataset** and compare different splitting criteria.

## Tasks

* Prepare and preprocess the dataset.
* Train `DecisionTreeClassifier` using:

  * Entropy (ID3)
  * Log Loss
  * Gini (CART)
* Compare the models.
* Visualize and interpret the decision tree.
* Evaluate the models using cross-validation and accuracy.

## Optional — From Scratch

As an optional extension, implement a basic Decision Tree classifier **from scratch** without using `DecisionTreeClassifier`.

The implementation can include:

* Entropy calculation
* Information Gain
* Finding the best split
* Recursive tree construction
* Prediction using the constructed tree
* Evaluation and comparison with the `scikit-learn` implementation

This section is intended to help understand how Decision Tree classification works internally.

## Libraries

* `scikit-learn`
* `pandas`
* `numpy`
* `matplotlib`

---

# Problem 2 — Decision Tree Regression and Handling Missing Values

## Objective

Apply Decision Tree Regression on the **Diabetes dataset** and investigate the effect of tree depth and missing-value handling on model performance.

## Tasks

* Train `DecisionTreeRegressor` using the **squared error** criterion.
* Experiment with different tree depths.
* Evaluate the models using:

  * MAE (Mean Absolute Error)
  * RMSE (Root Mean Squared Error)
* Simulate missing data by randomly removing **10% of the values in one feature**.
* Handle missing values using:

  * Mean imputation
  * `SimpleImputer` from `scikit-learn`
* Compare the results and analyze the impact of missing data handling on model quality.

## Libraries

* `scikit-learn`
* `pandas`
* `numpy`
* `matplotlib`

---

# References

* [Scikit-learn Decision Trees Documentation](https://scikit-learn.org/stable/modules/tree.html)
* Lecture slides: **DecisionTree.pdf**
