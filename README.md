# HCMUT Machine Learning Labs

Welcome to my Machine Learning Labs repository! This repository is a collection of lab exercises and projects completed during my **Machine Learning course** at Ho Chi Minh City University of Technology (HCMUT).

## Repository Structure

The repository is organized by machine learning topics. Each folder contains Jupyter Notebooks that explore both **library-based implementations** and, where applicable, **from-scratch implementations** to better understand the underlying concepts.

### Current Labs

1. **[Decision Tree](./Decision_Tree/)**

   * `DT_Classification.ipynb`: Decision Tree classification on the **Iris dataset**. It covers Entropy (ID3), Log Loss, and Gini (CART), together with tree visualization and cross-validation. An optional from-scratch implementation is also included.

   * `DT_Regression.ipynb`: Decision Tree regression on the **Diabetes dataset** using the squared error criterion. It explores different tree depths and evaluates the models using MAE and RMSE. The notebook also investigates missing-value handling using mean imputation and `SimpleImputer`.

   *(Additional labs and topics will be added as the course progresses.)*

## Getting Started

To run these notebooks locally and experiment with the code, Python is required. Using a virtual environment is highly recommended.

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
```

### 2. Install Dependencies

Install the required Python packages using the provided `requirements.txt`:

```bash
pip install -r requirements.txt
```

### 3. Run Jupyter Notebook

Launch Jupyter Notebook:

```bash
jupyter notebook
```

## Technologies & Libraries Used

* **Python 3** — Main programming language.
* **NumPy & Pandas** — Data manipulation, preprocessing, and numerical operations.
* **Matplotlib** — Data visualization.
* **Scikit-Learn** — Dataset loading, model implementation, cross-validation, preprocessing, and evaluation.

## About

* **Course:** Machine Learning
* **Semester:** 2026–2027 (1st semester)
* **University:** Ho Chi Minh City University of Technology (HCMUT)
