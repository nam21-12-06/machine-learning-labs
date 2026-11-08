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
![pgmpy](https://img.shields.io/badge/pgmpy-0.1.25+-007ACC?style=flat-square)
![NetworkX](https://img.shields.io/badge/NetworkX-3.2+-2B7A78?style=flat-square)
![NLTK](https://img.shields.io/badge/NLTK-3.8+-1B4965?style=flat-square)

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
| **04** | [**Bayesian Learning**](./Bayesian/)                               | [`MLE-MAP_in_Linear_Regression.ipynb`](./Bayesian/MLE-MAP_in_Linear_Regression.ipynb)<br>[`Bayesian_Classification_Naive_Bayes.ipynb`](./Bayesian/Bayesian_Classification_Naive_Bayes.ipynb)                                                                                                          | Synthetic Gaussian ($N=100$)<br>20 Newsgroups (1,977 articles) | MLE (OLS), MAP (Ridge), MAP-Ridge Equivalence ($\alpha = \sigma^2/\tau^2$), Naive Bayes (Gaussian, Multinomial, Bernoulli, Complement), TF-IDF, Laplace Smoothing | `Completed` |
| **05** | [**Genetic Algorithms**](./GA/) | [`GA_Structure_Reproduction_Operators.ipynb`](./GA/GA_Structure_Reproduction_Operators.ipynb)<br>[`GA_Rule_Representation_Hyperparameter_Optimization.ipynb`](./GA/GA_Rule_Representation_Hyperparameter_Optimization.ipynb) | OneMax Bit-String ($L=20$, $p=50$)<br>Synthetic Rule Classification ($N=120$) | Roulette Wheel Selection, Single/Two-point/Uniform Crossover, Bit Mutation, GABIL Rule Induction, Modulo-8 Offset-Aligned Crossover | `Completed` |
| **06** | [**Bayesian Networks**](./Bayesian_Network/) | [`Bayesian_Network_Inference.ipynb`](./Bayesian_Network/Bayesian_Network_Inference.ipynb)<br>[`Text_Classification_Naive_Bayes.ipynb`](./Bayesian_Network/Text_Classification_Naive_Bayes.ipynb) | Medical Pulmonary Diagnosis ($S, L, C, X$)<br>SMS Spam Collection (5,572 messages) | DAG Topology, Factorization, Variable Elimination, Belief Propagation, From-scratch Naive Bayes, Laplace Smoothing, 8 NLP Pipelines, LLR | `Completed` |
| **07** | [**Support Vector Machines**](./SVM/) | [`Linear_vs_Nonlinear_SVM.ipynb`](./SVM/Linear_vs_Nonlinear_SVM.ipynb)<br>[`Study_SVM_Parameters.ipynb`](./SVM/Study_SVM_Parameters.ipynb) | Synthetic 2D ($N=500$)<br>Iris Binary ($N=100$)<br>Overlapping Blobs ($N=300$) | Maximum Margin, Primal/Dual, KKT Conditions, Kernel Trick, Margin Width ($2/\Vert\mathbf{w}\Vert_2$), Parameter Sweeps ($C, \gamma$), Hard vs. Soft Margin, `GridSearchCV` | `Completed` |

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

### Module 4 — Bayesian Learning

<details open>
<summary><b>Click to expand Module 4 details</b></summary>

**Folder**: [`Bayesian/`](./Bayesian/) | **Specification**: [`BayesianLearning_Programming_HW_EL.pdf`](./Bayesian/BayesianLearning_Programming_HW_EL.pdf) | **Documentation**: [`README.md`](./Bayesian/README.md)

- **Problem 1 — Comparing MLE vs. MAP in Linear Regression** ([`MLE-MAP_in_Linear_Regression.ipynb`](./Bayesian/MLE-MAP_in_Linear_Regression.ipynb))
  - **Dataset**: Synthetic Gaussian linear dataset ($N=100$, true relationship $y = 2X + 1 + \epsilon$ with $\epsilon \sim \mathcal{N}(0, 0.5)$).
  - **Paradigms**: Frequentist Maximum Likelihood Estimation (MLE via Ordinary Least Squares) vs. Bayesian Maximum A Posteriori (MAP via Ridge Regression).
  - **Core Theoretical Link**: Mathematical proof showing that MAP estimation under a zero-mean Gaussian weight prior $w \sim \mathcal{N}(0, \tau^2)$ and Gaussian noise $\epsilon \sim \mathcal{N}(0, \sigma^2)$ is identical to Ridge regression with regularization strength $\alpha = \sigma^2 / \tau^2$.
  - **Highlights**: Hyperparameter sweep across $\alpha \in [0.1, 1.0, 10.0, 100.0]$, weight shrinkage trajectories, and robustness evaluation under 5% extreme noise contamination.
  - **Best Metric**: MLE achieves Test MSE = 0.4269 ($\hat{w} = 1.8872$, $\hat{b} = 1.0657$); Ridge with $\alpha = 0.1$ closely matches with Test MSE = 0.4271.

- **Problem 2 — Bayesian Classification and Naive Bayes** ([`Bayesian_Classification_Naive_Bayes.ipynb`](./Bayesian/Bayesian_Classification_Naive_Bayes.ipynb))
  - **Dataset**: 20 Newsgroups (`sci.med` vs. `sci.space`, $N = 1977$ articles, 80/20 train/test split: $n_{\text{train}} = 1581$, $n_{\text{test}} = 396$).
  - **Feature Extraction**: `TfidfVectorizer(max_features=5000, stop_words='english')` yielding sparse term-frequency inverse-document matrices (>98.5% sparsity).
  - **Bayesian Architectures**: Comparative evaluation of GaussianNB, MultinomialNB, BernoulliNB, and ComplementNB with inference latency profiling.
  - **Laplace Smoothing Sweep**: Log-scale exploration of additive smoothing parameter $\alpha \in [10^{-4}, 10.0]$ to combat the Zero-Probability dilemma.
  - **Feature Attribution**: Extraction of top category-discriminative vocabulary using Log-Likelihood Ratio (LLR) and Bayes' theorem.
  - **Best Metric**: BernoulliNB achieves top Test Accuracy = 98.74% and F1-Score = 0.9874; ComplementNB delivers ultra-low inference latency (0.72 ms).

</details>

---

### Module 5 — Genetic Algorithms

<details open>
<summary><b>Click to expand Module 5 details</b></summary>

**Folder**: [`GA/`](./GA/) | **Specification**: [`GA.pdf`](./GA/GA.pdf) | **Documentation**: [`README.md`](./GA/README.md)

- **Problem 1 — Genetic Algorithm Structure and Reproduction Operators** ([`GA_Structure_Reproduction_Operators.ipynb`](./GA/GA_Structure_Reproduction_Operators.ipynb))
  - **Problem Domain**: Binary bit-string optimization (OneMax problem, $L = 20$ bits, population size $p = 50$).
  - **Selection Scheme**: Fitness-proportionate Roulette Wheel Selection with direct retention of $(1 - r) \cdot p = 10$ individuals ($r = 0.8$).
  - **Reproduction Operators**: Comparative implementation of Single-point, Two-point, and Uniform Crossover across $(r/2) \cdot p = 20$ parent pairs producing 40 offspring.
  - **Mutation Operator**: Bit-flip mutation applied with rate $m = 5\%$ to candidate individuals.
  - **Statistical Validation**: Monte Carlo benchmark over $N = 30$ independent runs tracking maximum fitness, population mean fitness, diversity depletion, and convergence speed.
  - **Best Metric**: Two-point Crossover achieves 100.0% success rate with fastest average convergence in 34.20 +/- 16.82 generations (vs. 53.03 generations for Single-point).

- **Problem 2 — Rule Representation and GA Hyperparameter Optimization** ([`GA_Rule_Representation_Hyperparameter_Optimization.ipynb`](./GA/GA_Rule_Representation_Hyperparameter_Optimization.ipynb))
  - **Problem Domain**: Propositional rule induction (GABIL framework, $N = 120$ samples, attributes $A_1 \in \{0, 1, 2\}, A_2, A_3 \in \{0, 1\}$).
  - **Encoding Scheme**: Fixed 8-bit chromosome $[A_1(3) \mid A_2(2) \mid A_3(2) \mid C(1)]$ with internal disjunction and external conjunction.
  - **Fitness Metric**: Non-linear squared fitness $\text{Fitness}(h) = (\text{correct}(h))^2$ providing super-linear selection amplification.
  - **Hyperparameter Grid Search**: $4 \times 4$ factorial search across $r \in [0.2, 0.5, 0.8, 1.0]$ and $m \in [0.01, 0.05, 0.1, 0.2]$ over 80 runs; reveals single-rule conjunctive ceiling at 83.33%.
  - **Variable-Length Crossover (Task c)**: GABIL modulo-8 offset-aligned crossover ($pt_1 \equiv pt_2 \pmod 8$) with parsimony pressure ($\lambda = 0.5$); achieves 100.0% accuracy with concise DNF rule sets.
  - **Best Metric**: Variable-length GA achieves 100.0% accuracy across 10/10 independent Monte Carlo trials (mean convergence: 28.6 generations).

</details>

---

### Module 6 — Bayesian Networks & Probabilistic Graphical Models

<details open>
<summary><b>Click to expand Module 6 details</b></summary>

**Folder**: [`Bayesian_Network/`](./Bayesian_Network/) | **Specification**: [`Bayesian_Network_HW_EL.pdf`](./Bayesian_Network/Bayesian_Network_HW_EL.pdf) | **Documentation**: [`README.md`](./Bayesian_Network/README.md)

- **Problem 1 — Implementing Inference in Bayesian Networks** ([`Bayesian_Network_Inference.ipynb`](./Bayesian_Network/Bayesian_Network_Inference.ipynb))
  - **Problem Domain**: Medical pulmonary diagnostic reasoning across four clinical variables: $\text{Smoking} \in \{0, 1\}$, $\text{LungCancer} \in \{0, 1\}$, $\text{Cough} \in \{0, 1\}$, and $\text{XRay} \in \{0, 1\}$.
  - **Network Topology & Factorization**: Directed Acyclic Graph encoding local Markov properties with joint factorization: $P(S, L, C, X) = P(S) \cdot P(L \mid S) \cdot P(C \mid L, S) \cdot P(X \mid L)$.
  - **Multi-Paradigm Probabilistic Inference**:
    - **Causal (Predictive)**: $P(\text{Cough}=\text{Yes} \mid \text{Smoking}=\text{Yes}) = 44.00\%$ (vs. $10.42\%$ non-smokers; $+33.58\%$ risk gain).
    - **Diagnostic (Explanatory)**: $P(\text{Smoking}=\text{Yes} \mid \text{XRay}=\text{Abnormal}) = 49.72\%$ (up from $30.00\%$ prior; $+19.72\%$ abductive shift).
    - **Intercausal (Explaining Away)**: $P(\text{LungCancer}=\text{Yes} \mid \text{Cough}=\text{Yes}, \text{XRay}=\text{Abnormal}) = 74.58\%$ (synergistic leap from $3.70\%$ prior).
  - **Exact Mathematical Validation**: Full 16-state joint probability enumeration proving machine-precision equivalence between manual marginalization and algorithmic Variable Elimination ($\Delta < 10^{-15}$).
  - **Clinical Visualization**: High-resolution NetworkX DAG architecture diagram and 4-panel diagnostic reasoning dashboard.

- **Problem 2 — Text Classification with Naive Bayes** ([`Text_Classification_Naive_Bayes.ipynb`](./Bayesian_Network/Text_Classification_Naive_Bayes.ipynb))
  - **Problem Domain**: SMS Spam detection on Kaggle SMS Spam Collection ($N = 5,572$ messages: 4,825 Ham, 747 Spam; 80/20 stratified split into 4,457 train and 1,115 test samples).
  - **From-scratch Implementation**: Multinomial Naive Bayes classifier built from scratch in NumPy (`ScratchMultinomialNB`) with pure-Python tokenization/vocabulary builder (`ScratchTextPreprocessor`), MLE priors, Laplace smoothing ($\alpha = 1.0$), and vectorized log-space inference with Log-Sum-Exp normalization.
  - **Scikit-Learn Head-to-Head Benchmark**: Identical 100.00% prediction match across all 1,115 test samples (Accuracy: 98.74%, Precision: 98.56%, Recall: 91.95%, F1-Score: 95.14%; 0 divergent classifications; $4.9\times$ faster CPU training).
  - **8 NLP Preprocessing Pipelines**: Evaluated Count-based BoW vs. Normalized TF-IDF combined with stopword removal, Porter stemming, and WordNet lemmatization. BoW yields peak balanced detection ($F_1 = 95.14\%$); TF-IDF delivers **100.00% Precision** (0 false alarms out of 966 test Ham messages).
  - **Vocabulary Scaling Sensitivity**: Benchmarked capacity constraints $\vert V \vert \in [50, 7668]$; performance saturates at $\vert V \vert = 2,000$ (98.74% accuracy, 95.17% F1), reducing memory footprint by 73.91% with zero predictive loss.
  - **Lexical Feature Attribution**: Ranked most discriminative tokens using Log-Likelihood Ratios ($\text{LLR}(w) = \ln \frac{P(w \mid \text{Spam})}{P(w \mid \text{Ham})}$); top Spam tokens: `claim` (+5.73), `prize` (+5.39), `150p` (+5.18); top Ham tokens: `gt` (-4.39), `lt` (-4.38), `he` (-4.00), `da` (-3.74).
  - **Local Prediction Explainability**: Interactive inspector (`explain_message_prediction`) decomposing messages into token-level log-odds contributions $\Delta \text{Log-Odds}(w) = \text{LLR}(w)$, verified on 4 real-world test communications.
  - **5 Embedded High-DPI Visualizations**: Figure 1 (EDA Dashboard), Figure 2 (Confusion Matrix Heatmap & ROC/PR Curves), Figure 3 (4-panel NLP Diagnostics), Figure 4 (Vocabulary Scaling Trajectories), and Figure 5 (LLR Dual Bar Charts).
  - **Best Metric**: BoW + Scratch / Scikit-Learn MultinomialNB achieves Test Accuracy = 98.74% and F1-Score = 95.14%; TF-IDF variants achieve 100.00% Precision.

</details>

---

### Module 7 — Support Vector Machines (SVM)

<details open>
<summary><b>Click to expand Module 7 details</b></summary>

**Folder**: [`SVM/`](./SVM/) | **Specification**: [`SVM_Lab.pdf`](./SVM/SVM_Lab.pdf) | **Documentation**: [`README.md`](./SVM/README.md)

- **Problem 1 — Comparison of Linear and Non-linear SVM** ([`Linear_vs_Nonlinear_SVM.ipynb`](./SVM/Linear_vs_Nonlinear_SVM.ipynb))
  - **Datasets & Preprocessing**: 2D Synthetic dataset (`make_classification`, $N=500$ samples, 2 classes) and real-world benchmark (Fisher's Iris dataset: Versicolor vs. Virginica on petal length and petal width, $N=100$ samples). Standardized via `StandardScaler` under a strict zero-leakage 70/30 stratified train/test split.
  - **4 Model Architectures**: Evaluated `LinearSVC` (primal coordinate descent via LIBLINEAR), `SVC(kernel='linear')`, `SVC(kernel='rbf')`, `SVC(kernel='poly', degree=3)`, and `SVC(kernel='sigmoid')`.
  - **Benchmark Performance**: RBF kernel achieves top generalization accuracy across both testbeds (**94.67%** on Synthetic, **90.00%** on Iris); Linear models achieve **88.67% - 92.00%** on Synthetic and **86.67%** on Iris; Polynomial ($d=3$) achieves **83.33%** and **76.67%**.
  - **Support Vector Sparsity**: RBF requires only **28.86%** active support vectors on Synthetic ($101/350$) and **21.43%** on Iris ($15/70$). `LinearSVC` discards all support vectors post-training, enabling $\mathcal{O}(D)$ microsecond inference.
  - **5 Embedded High-DPI Visualizations**: Figure 1 (EDA and feature distributions), Figure 2 (Synthetic decision boundaries, margins $f(\mathbf{x})=\pm 1$, and support vectors), Figure 3 (Iris decision boundaries and margins), Figure 4 (Multi-metric comparative diagnostics, SV counts, latency, and ROC curves), and Figure 5 (3D geometric intuition of the kernel trick mapping 2D non-separable concentric data to a linearly separable 3D space).
  - **Scenario-Based Engineering Guidelines**: Comprehensive guide mapping data regimes ($D \gg N$, non-linear tabular, edge low-latency, and polynomial dynamics) to optimal kernel selection.
  - **Best Metric**: RBF Kernel (`SVC(kernel='rbf')`) achieves Test Accuracy = 94.67% (Synthetic) and 90.00% (Iris) with 28.86% support vector sparsity.

- **Problem 2 — Study on the Effect of SVM Parameters** ([`Study_SVM_Parameters.ipynb`](./SVM/Study_SVM_Parameters.ipynb))
  - **Datasets & Hyperparameter Sweeps**: Standardized 2D synthetic dataset ($N=500$) and noisy overlapping isotropic Gaussian clusters (`make_blobs`, $N=300$, `cluster_std=1.5`).
  - **Regularization Parameter $C$ Analysis**: Evaluated $C \in [0.001, 1000]$. Geometric margin width contracts from $8.185$ to $0.795$ ($>10\times$ contraction) and support vector count decreases from $348$ down to $112$ ($99.4\%$ down to $32.0\%$ sparsity). Rigorously quantifies the transition from high-bias underfitting ($C \le 0.01$) to the optimal margin-penalty trade-off ($C \in [1, 10]$).
  - **RBF Kernel Parameter $\gamma$ Dynamics**: Evaluated $\gamma \in [0.001, 100]$ corresponding to Gaussian influence radii $\sigma \in [22.36, 0.07]$. Demonstrates smooth linear underfitting for small $\gamma$, an optimal generalization sweet spot at $\gamma^* = 1.0$ (95.33% accuracy, 94 SVs), and severe island overfitting for $\gamma \ge 10.0$ (Train accuracy 99.14%, 252 SVs).
  - **Hard vs. Soft Margin on Overlapping Noise**: Under cluster overlap, soft-margin ($C=1$) preserves a wider geometric margin ($0.849$ vs. $0.765$) and tolerates 46 margin penetrations without boundary distortion, whereas hard-margin ($C=1000$) forces margin shrinkage and exhibits high boundary rigidity.
  - **GridSearchCV Hyperparameter Optimization**: Exhaustive 5-fold cross-validation over $(C, \gamma)$ discovers optimal configuration ($C^*=10, \gamma^*=10$), improving cross-validation accuracy to $97.43\%$ and test accuracy to $96.67\%$ (F1-score $96.77\%$).
  - **6 Embedded High-DPI Visualizations**: Figure 1 (3-panel curves: Accuracy, Margin Width, and SVs vs. $\log_{10}(C)$), Figure 2 (4-panel visual decision boundaries across $C$), Figure 3 (6-panel RBF boundary complexity progression across $\gamma$), Figure 4 (Validation curve for $\gamma$ with $\pm 1\sigma$ bands), Figure 5 (Soft vs. hard margin on noisy blobs highlighting slack violations), and Figure 6 (2D GridSearchCV heatmap and bias-variance learning curve).
  - **Best Metric**: Optimized RBF via `GridSearchCV` ($C^*=10.0, \gamma^*=10.0$) achieves 97.43% 5-Fold CV Accuracy, 96.67% Test Accuracy, and 96.77% Test F1-Score.

</details>

---

## Directory Structure

```text
Machine-Learning-Labs/
│
├── Decision_Tree/
│   ├── DecisionTree_Programming_EL.pdf                             # Problem specification
│   ├── DT_Classification.ipynb                                     # Iris classification & from-scratch tree
│   ├── DT_Regression.ipynb                                         # Diabetes regression & missing data handling
│   └── README.md                                                   # Decision Tree module documentation
│
├── ANN-MLP/
│   ├── Linear_Model/
│   │   ├── MLP_Programming_HW_EL.pdf                               # Problem specification
│   │   ├── linear_regresion.ipynb                                  # Problem 1: Linear, Ridge, and LASSO
│   │   ├── logistic_regression_binary_classification.ipynb         # Problem 2: Logistic Regression (L1 vs L2)
│   │   ├── multi-output_regression.ipynb                           # Problem 3: Multi-output Regression (Linnerud)
│   │   └── README.md                                               # Linear Models module documentation
│   ├── Training_ANN-Optimizer/
│   │   ├── ANN_Part2_Programming_HW_EL.pdf                         # Problem specification
│   │   ├── Optimization_Algorithm.ipynb                            # Problem 1: SGD, Momentum, Adam, AdamW
│   │   ├── Normalization_Regularization_MLP.ipynb                  # Problem 2: Normalization & Regularization
│   │   └── README.md                                               # ANN Optimizers module documentation
│   └── README.md                                                   # ANN-MLP module hub documentation
│
├── Bayesian/
│   ├── BayesianLearning_Programming_HW_EL.pdf                      # Problem specification
│   ├── MLE-MAP_in_Linear_Regression.ipynb                          # Problem 1: Comparing MLE vs MAP in Linear Regression
│   ├── Bayesian_Classification_Naive_Bayes.ipynb                   # Problem 2: Bayesian Classification and Naive Bayes
│   └── README.md                                                   # Bayesian Learning module documentation
│
├── GA/
│   ├── GA.pdf                                                      # Problem specification
│   ├── GA_Structure_Reproduction_Operators.ipynb                   # Problem 1: GA Structure and Reproduction Operators
│   ├── GA_Rule_Representation_Hyperparameter_Optimization.ipynb    # Problem 2: GA Rule Representation and Hyperparameters
│   └── README.md                                                   # Genetic Algorithms module documentation
│
├── Bayesian_Network/
│   ├── Bayesian_Network_HW_EL.pdf                                  # Problem specification
│   ├── Bayesian_Network_Inference.ipynb                            # Problem 1: Bayesian Network Inference (Causal, Diagnostic, Intercausal)
│   ├── Text_Classification_Naive_Bayes.ipynb                       # Problem 2: Text Classification with Scratch & Scikit-Learn Naive Bayes
│   ├── spam.csv                                                    # SMS Spam Collection dataset (5,572 messages)
│   └── README.md                                                   # Bayesian Networks module documentation
│
├── SVM/
│   ├── SVM_Lab.pdf                                                 # Problem specification
│   ├── Linear_vs_Nonlinear_SVM.ipynb                               # Problem 1: Comparison of Linear and Non-linear SVM
│   ├── Study_SVM_Parameters.ipynb                                  # Problem 2: Study on the Effect of SVM Parameters
│   └── README.md                                                   # Support Vector Machines module documentation
│
├── agents/
│   └── rules.md                                                    # Working principles and development guidelines
│
├── requirements.txt                                                # Python environment dependencies
└── README.md                                                       # Root repository hub documentation
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
  - `BayesianLearning.pdf`
  - `GA.pdf`
  - `Bayesian_Network_HW_EL.pdf`
  - `SVM_Lab.pdf`
  - `1.BayesNet - Intro.pdf`
  - `5.NaiveBayes.pdf`
- **Book References**:
  - _Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow_ by Aurélien Géron.
  - _Pattern Recognition and Machine Learning_ by Christopher M. Bishop.
  - _Machine Learning: A Probabilistic Perspective_ by Kevin P. Murphy.
  - _An Introduction to Genetic Algorithms_ by Melanie Mitchell.
  - _Genetic Algorithms in Search, Optimization, and Machine Learning_ by David E. Goldberg.
  - _Probabilistic Graphical Models: Principles and Techniques_ by Daphne Koller and Nir Friedman.
  - _Probabilistic Reasoning in Intelligent Systems_ by Judea Pearl.
  - _Statistical Learning Theory_ by Vladimir N. Vapnik.
  - _Learning with Kernels_ by Bernhard Schölkopf and Alexander J. Smola.

---

<div align="center">
  <sub>HCMUT Machine Learning Coursework • Maintained with standard coding practices and mathematical rigor.</sub>
</div>
