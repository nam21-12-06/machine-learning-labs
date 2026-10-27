# Bayesian Learning with Scikit-learn

This module explores **Bayesian principles, probabilistic parameter estimation, and Naive Bayes text classification**, covering both continuous regression (MLE vs. MAP) and discrete multi-class text categorization using Scikit-Learn.

---

## Module Overview & Status

| Problem | Notebook | Dataset | Key Techniques | Status |
| :---: | :--- | :--- | :--- | :---: |
| **1** | [`MLE-MAP_in_Linear_Regression.ipynb`](./MLE-MAP_in_Linear_Regression.ipynb) | Synthetic Gaussian Data ($N=100$) | MLE (OLS), MAP (Ridge), $\alpha = \frac{\sigma^2}{\tau^2}$, Outlier Robustness | `Completed` |
| **2** | [`Bayesian_Classification_Naive_Bayes.ipynb`](./Bayesian_Classification_Naive_Bayes.ipynb) | 20 Newsgroups (`sci.med` vs `sci.space`) | GaussianNB, MultinomialNB, BernoulliNB, ComplementNB, TF-IDF, Laplace Smoothing | `Completed` |

---

# Problem 1 — Comparing MLE vs. MAP in Linear Regression

## Objective

Investigate the mathematical equivalence and empirical differences between **Maximum Likelihood Estimation (MLE)** and **Maximum A Posteriori (MAP)** estimation on a synthetic linear regression problem:

$$
y = 2X + 1 + \epsilon, \quad X \sim \mathcal{N}(0, 1), \quad \epsilon \sim \mathcal{N}(0, \sigma^2) \quad (\sigma^2 = 0.5)
$$

## Tasks & Methodology

- **Synthetic Data Generation**: Generate 100 random samples and partition into 70% training ($n=70$) and 30% test ($n=30$) sets.
- **MLE Estimation**: Train `LinearRegression` (Ordinary Least Squares) to estimate slope $w$ and intercept $b$ under uniform flat prior assumption.
- **MAP Estimation (Ridge Regression)**: Train `Ridge` across $\alpha \in [0.1, 1.0, 10.0, 100.0]$ and derive the theoretical correspondence:

$$
\alpha = \frac{\sigma^2}{\tau^2}
$$

  where $\sigma^2$ is the observation noise variance and $\tau^2$ is the variance of the zero-mean isotropic Gaussian prior $p(w) \sim \mathcal{N}(0, \tau^2)$.
- **Evaluation & Visual Comparison**: Evaluate Test MSE, compare estimated coefficients against true parameters ($w^*=2, b^*=1$), and plot regression trajectories.
- **Outlier Robustness Experiment**: Inject large noise ($\eta \sim \mathcal{N}(0, 10^2)$) into 5% of training samples ($n=3$) to evaluate the sensitivity of squared-loss estimators.

## Summary of Results

### 1. Clean Data Benchmark (Task d)

| Method | Regularization ($\alpha$) | Estimated Slope ($w$) | Intercept ($b$) | Test MSE | Shrinkage vs. MLE |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **True Model** | - | 2.0000 | 1.0000 | - | - |
| **MLE (LinearRegression)** | None ($\tau^2 \to \infty$) | 1.8872 | 1.0657 | **0.4269** | 0.00% |
| **MAP (Ridge $\alpha=0.1$)** | 0.1 | 1.8843 | 1.0653 | 0.4271 | 0.15% |
| **MAP (Ridge $\alpha=1.0$)** | 1.0 | 1.8586 | 1.0620 | 0.4289 | 1.52% |
| **MAP (Ridge $\alpha=10.0$)** | 10.0 | 1.6356 | 1.0336 | 0.4754 | 13.33% |
| **MAP (Ridge $\alpha=100.0$)** | 100.0 | 0.7435 | 0.9198 | 1.2153 | 60.60% |

### 2. Outlier Robustness Benchmark (Task e)

| Training Condition | Estimator | Regularization ($\alpha$) | Estimated Slope ($w$) | Intercept ($b$) | Test MSE |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Clean Baseline** | MLE | None | 1.8872 | 1.0657 | 0.4269 |
| **5% Outliers Injected** | MLE | None | 2.0323 | 0.8244 | 0.3858 |
| **5% Outliers Injected** | MAP (Ridge) | 0.1 | 2.0292 | 0.8240 | 0.3856 |
| **5% Outliers Injected** | MAP (Ridge) | 1.0 | 2.0015 | 0.8204 | 0.3843 |
| **5% Outliers Injected** | MAP (Ridge) | 10.0 | 1.7614 | 0.7898 | 0.4084 |
| **5% Outliers Injected** | MAP (Ridge) | 100.0 | 0.8007 | 0.6672 | 1.1468 |

---

# Problem 2 — Bayesian Classification and Naive Bayes

## Objective

Perform binary text classification using probabilistic Bayesian methods on the **20 Newsgroups dataset**, classifying articles into `sci.med` (medicine) and `sci.space` (space exploration).

## Tasks & Methodology

- **Data Preparation**: Fetch 1,977 articles across `sci.med` (990) and `sci.space` (987); perform stratified 80/20 train/test split ($n_{\text{train}} = 1,581, n_{\text{test}} = 396$).
- **Text Preprocessing**: Transform unstructured text into numerical feature vectors using `TfidfVectorizer(max_features=5000, stop_words='english')` fitted strictly on training data.
- **Model Formulations**:
  - **GaussianNB**: Models continuous features using class-conditional Gaussian distributions.
  - **MultinomialNB**: Models word count and TF-IDF intensities via multinomial distributions.
  - **BernoulliNB**: Multivariate binary model capturing word presence and absence.
  - **ComplementNB**: Inverted complement-class estimator designed to mitigate text classification biases.
- **Performance Evaluation**: Benchmark Accuracy, Precision, Recall, F1-score, training time, and inference latency; visualize confusion matrices.
- **Laplace Smoothing Sweep**: Investigate the role of smoothing hyperparameter $\alpha \in [10^{-4}, 10.0]$ in resolving the Zero-Probability problem ($P(w \mid c) = 0$).
- **Feature Importance**: Identify the most discriminative words for each category using class log-probabilities and the Log-Likelihood Ratio (LLR).

## Summary of Results

### 1. Model Benchmark Comparison (Task d)

| Model | Accuracy | Precision | Recall | F1-Score | Train Time (ms) | Inference Time (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **GaussianNB** | 96.97% | 96.99% | 96.97% | 0.9697 | 254.44 | 87.40 |
| **MultinomialNB** | 97.73% | 97.76% | 97.73% | 0.9773 | 5.39 | 1.41 |
| **BernoulliNB** | **98.74%** | **98.74%** | **98.74%** | **0.9874** | 7.34 | 2.43 |
| **ComplementNB** | 97.73% | 97.76% | 97.73% | 0.9773 | **3.22** | **0.72** |

### 2. Laplace Smoothing Impact on MultinomialNB (Task e)

| Alpha ($\alpha$) | Test Accuracy | Test F1-Score | Behavior & Stability |
| :---: | :---: | :---: | :--- |
| **0.0001** | **97.98%** | **0.9798** | High sensitivity; slight risk of zero-frequency penalties |
| **0.0010** | **97.98%** | **0.9798** | Stable high performance |
| **0.0100** | **97.98%** | **0.9798** | Stable high performance |
| **0.1000** | **97.98%** | **0.9798** | Optimal balance between data fidelity and regularization |
| **1.0000** | 97.73% | 0.9773 | Standard Laplace smoothing |
| **10.0000** | 97.22% | 0.9722 | Over-smoothing (probabilities pulled toward uniform distribution) |

### 3. Top Discriminative Vocabulary (Task f)

- **`sci.space` Indicators (Highest LLR)**: `space`, `nasa`, `orbit`, `moon`, `launch`, `satellite`, `flight`, `shuttle`, `solar`, `earth`.
- **`sci.med` Indicators (Lowest LLR)**: `doctor`, `patients`, `disease`, `medical`, `treatment`, `pain`, `medicine`, `cancer`, `symptoms`, `clinical`.

---

## Key Insights

1. **Equivalence of MAP and Ridge**: Assuming a Gaussian prior $w \sim \mathcal{N}(0, \tau^2)$ under Gaussian noise $\epsilon \sim \mathcal{N}(0, \sigma^2)$ transforms the MAP objective into an $L_2$-penalized least squares problem with regularization strength $\alpha = \sigma^2 / \tau^2$.
2. **Prior Variance Dynamics**: As $\tau^2 \to \infty$ (complete prior uncertainty), $\alpha \to 0$, causing the MAP estimator to converge asymptotically to the MLE solution. When $\tau^2$ is small, the prior dominates and shrinks $w$ heavily toward zero.
3. **Outlier Sensitivity of Gaussian Models**: Both MLE and standard Gaussian MAP minimize quadratic residuals $(y_i - \hat{y}_i)^2$, making both estimators vulnerable to extreme outliers. Robust estimation requires heavy-tailed likelihoods (e.g., Student-$t$, Huber) or Laplace priors ($L_1$/LASSO).
4. **Likelihood Model Selection in Text Classification**: MultinomialNB, BernoulliNB, and ComplementNB significantly outperform GaussianNB both in accuracy and inference latency, because discrete probability distributions naturally model sparse token occurrences, whereas Gaussian bell curves allocate probability mass to invalid negative/continuous domains.
5. **Role of Additive Smoothing**: Laplace smoothing ($\alpha > 0$) is vital in Naive Bayes to prevent zero-probability vetoes from novel tokens in test documents.

---

## References

- Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer (Chapter 3: Linear Models for Regression).
- Murphy, K. P. (2012). *Machine Learning: A Probabilistic Perspective*. MIT Press (Chapter 3: Generative Models for Discrete Data; Chapter 7: Linear Regression).
- Barber, D. (2012). *Bayesian Reasoning and Machine Learning*. Cambridge University Press.
- [Scikit-learn Naive Bayes Documentation](https://scikit-learn.org/stable/modules/naive_bayes.html)
- Course Lecture Slides: `BayesianLearning.pdf`
