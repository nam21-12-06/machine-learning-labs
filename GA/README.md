# Genetic Algorithms with Python

This module explores the design, theoretical foundations, and implementation from scratch of **Genetic Algorithms (GAs)** for combinatorial and rule-based optimization problems in Python, adhering to the canonical mechanisms introduced in evolutionary computation theory.

---

## Module Overview & Status

| Problem | Notebook | Problem Domain / Dataset | Key Techniques & Operators | Status |
| :---: | :--- | :--- | :--- | :---: |
| **1** | [`GA_Structure_Reproduction_Operators.ipynb`](./GA_Structure_Reproduction_Operators.ipynb) | OneMax Problem ($L=20$, $p=50$) | Roulette Wheel Selection, Single/Two-point/Uniform Crossover, Bit Mutation, Monte Carlo Evaluation | `Completed` |
| **2** | [`GA_Rule_Representation_Hyperparameter_Optimization.ipynb`](./GA_Rule_Representation_Hyperparameter_Optimization.ipynb) | Synthetic Rule Classification | Bit-string Rule Encoding, Hyperparameter Tuning ($r, m$), GABIL Variable-Length Crossover | `Completed` |

---

# Problem 1 — Genetic Algorithm Structure and Reproduction Operators

## Objective

Design and implement a complete Genetic Algorithm from scratch in Python to maximize a binary bit string (**OneMax problem**), benchmarking the convergence dynamics, schema disruption, and search efficiency of three fundamental reproduction operators.

$$\max_{h \in \{0, 1\}^L} \text{Fitness}(h) = \sum_{j=1}^L b_j, \quad L = 20, \quad p = 50$$

Global optimum: $h^* = (1, 1, \dots, 1)$ with maximum fitness $\text{Fitness}(h^*) = 20$.

## Reproduction Operator Configurations

1. **Fitness-Proportionate Selection (Roulette Wheel Selection)**:
   Selection probability is proportional to individual fitness: $\Pr(h_i) = \frac{\text{Fitness}(h_i)}{\sum_{k=1}^p \text{Fitness}(h_k)}$.
   - Direct Retention: A fraction $(1 - r) \cdot p = (1 - 0.8) \times 50 = 10$ individuals are directly retained via Roulette Wheel Selection into the next generation.
   - Parent Sampling: $(r / 2) \cdot p = 20$ parent pairs are selected to produce 40 offspring ($10 + 40 = 50$).

2. **Crossover Operators ($r = 0.8$)**:
   - **Single-point Crossover**: Swaps trailing segments after cut point $k \in \{1, \dots, L-1\}$.
   - **Two-point Crossover**: Swaps the internal segment enclosed between two cut loci $1 \le k_1 < k_2 \le L-1$.
   - **Uniform Crossover**: Swaps alleles independently at each locus via a Bernoulli mask $M \sim \text{Bernoulli}(0.5)^L$.

3. **Bit-flip Mutation ($m = 0.05$)**:
   - Each individual admitted to the candidate population has a 5% probability of undergoing mutation by inverting one randomly chosen bit ($b_j \leftarrow 1 - b_j$).

4. **Termination Criteria**:
   - Generational limit: 100 generations.
   - Early stopping: Reaching global maximum fitness ($\text{Fitness}(h) = 20$).

---

## Benchmark Results ($N = 30$ Independent Monte Carlo Trials)

| Crossover Operator | Success Rate (%) | Avg Convergence Gen | Min Conv Gen | Max Conv Gen | Mean Final Max Fitness | Mean Final Avg Fitness | Total Runtime (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Single-point Crossover** | 86.7% | 53.03 +/- 30.40 | 17 | 100 | 19.87 +/- 0.34 | 18.24 +/- 0.62 | 8.237 |
| **Two-point Crossover** | **100.0%** | **34.20 +/- 16.82** | **14** | **82** | **20.00 +/- 0.00** | **18.78 +/- 0.44** | **1.520** |
| **Uniform Crossover** | 96.7% | 37.43 +/- 22.86 | 15 | 100 | 19.97 +/- 0.18 | 18.65 +/- 0.51 | 2.092 |

---

## Key Theoretical Insights (Problem 1)

1. **Schema Theorem in Linear Landscapes**:
   The expected sampling frequency of schema $H$ satisfies: $\mathbb{E}[m(H, t+1)] \ge m(H, t) \cdot \frac{f(H)}{\bar{f}(t)} \left[ 1 - p_c \frac{\delta(H)}{L - 1} - o(H) p_m \right]$.
   On OneMax, each bit is an independent order-1 schema ($o(H) = 1$) with zero defining length ($\delta(H) = 0$). Epistasis is absent, enabling building blocks to assemble additively without inter-locus disruption.

2. **Positional Bias vs. Distributional Bias**:
   - Single-point crossover suffers from high positional bias: genes at opposite ends are readily separated while adjacent genes are tightly linked. It struggles to combine independent 1-bits when separated by unfavorable segments.
   - Two-point crossover eliminates edge effects by mapping the linear string to a continuous ring (torus), achieving the fastest convergence (34.20 generations) and 100% success rate.
   - Uniform crossover possesses zero positional bias, making it well-suited for additive, epistasis-free fitness landscapes.

3. **Exploration vs. Exploitation Balance**:
   - Direct retention of $(1 - r) = 20\%$ acts as a soft elitist memory, preserving superior building blocks from being corrupted.
   - Crossover ($r = 80\%$) drives rapid exploitation of beneficial genetic combinations.
   - Mutation ($m = 5\%$) injects the necessary exploration to re-introduce lost alleles when a locus accidentally fixates to zero.

---

# Problem 2 — Rule Representation and GA Hyperparameter Optimization

## Objective

Implement a propositional rule-based classification system using bit strings based on the **GABIL** framework (De Jong et al., 1993), investigate the sensitivity of reproduction hyperparameters ($r, m$) through full-factorial grid search, and eliminate the expressive bottleneck of single conjunctive rules via **variable-length chromosomes with offset-aligned crossover**.

---

## Hypothesis Representation & Non-linear Fitness (Task a)

### 1. Bit-string Propositional Encoding

Each rule is encoded into an 8-bit chromosome:

$$\text{Chromosome } h = [\underbrace{b_0 \, b_1 \, b_2}_{A_1 \in \{0, 1, 2\}} \mid \underbrace{b_3 \, b_4}_{A_2 \in \{0, 1\}} \mid \underbrace{b_5 \, b_6}_{A_3 \in \{0, 1\}} \mid \underbrace{b_7}_{C \in \{0, 1\}}]$$

- **Internal Disjunction**: Multiple 1-bits within an attribute slot represent logical OR (e.g. $[1, 1, 0]$ means $A_1 = 0 \lor A_1 = 1$; $[1, 1, 1]$ represents "don't care").
- **External Conjunction**: Conditions across distinct attributes are combined via logical AND:

$$\text{match}(h, \mathbf{x}) = \bigwedge_{j=1}^3 \left( h[\text{offset}_j + x_j] == 1 \right)$$

- **Prediction Function**:

$$\hat{y}(\mathbf{x}) = \begin{cases} b_7, & \text{if } \text{match}(h, \mathbf{x}) = 1 \\ 1 - b_7, & \text{if } \text{match}(h, \mathbf{x}) = 0 \end{cases}$$

### 2. Squared Fitness Function

$$\text{Fitness}(h) = \big(\text{correct}(h)\big)^2 = \left( \frac{1}{N} \sum_{i=1}^N \mathbb{I}\big(\hat{y}(\mathbf{x}_i) = y_i\big) \times 100\% \right)^2$$

The quadratic exponent provides **super-linear amplification**, steepening the fitness gradient near high classification performance ($100^2 - 83.33^2 = 3056$ vs $50^2 - 33.33^2 = 1389$), ensuring that superior hypotheses are strongly preferred by roulette wheel selection.

---

## Hyperparameter Grid Search Results (Task b)

Full factorial exploration across Crossover Rates $r \in [0.2, 0.5, 0.8, 1.0]$ and Mutation Rates $m \in [0.01, 0.05, 0.1, 0.2]$ (5 replications per cell, 80 executions total):

| Crossover Rate ($r$) | Mutation Rate ($m$) | Mean Accuracy (%) | Std Accuracy (%) | Mean Conv Gen | Mean Runtime (ms) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 0.2 | 0.01 | 83.17 | 0.33 | 0.6 | 35.12 |
| 0.2 | 0.05 | 83.33 | 0.00 | 2.8 | 32.12 |
| 0.2 | 0.10 | 83.33 | 0.00 | 0.8 | 28.89 |
| 0.2 | 0.20 | 83.33 | 0.00 | 2.4 | 27.93 |
| 0.5 | 0.01 | 83.33 | 0.00 | 0.4 | 31.14 |
| 0.5 | 0.05 | 83.33 | 0.00 | 0.0 | 37.46 |
| 0.5 | 0.10 | 81.00 | 4.67 | 3.2 | 33.87 |
| 0.5 | 0.20 | 83.33 | 0.00 | 0.8 | 36.63 |
| 0.8 | 0.01 | 82.17 | 2.33 | 0.6 | 38.29 |
| 0.8 | 0.05 | **83.33** | **0.00** | **0.4** | **36.42** |
| 0.8 | 0.10 | 83.33 | 0.00 | 1.4 | 38.76 |
| 0.8 | 0.20 | 83.33 | 0.00 | 0.8 | 43.06 |
| 1.0 | 0.01 | 83.33 | 0.00 | 0.6 | 38.68 |
| 1.0 | 0.05 | 83.33 | 0.00 | 0.6 | 39.18 |
| 1.0 | 0.10 | 83.33 | 0.00 | 0.4 | 37.14 |
| 1.0 | 0.20 | 83.33 | 0.00 | 0.8 | 37.25 |

### Findings from Grid Search
- **Single-Rule Ceiling**: All single-rule GA configurations plateau at **83.33% accuracy**. Because the synthetic ground truth is a disjunctive concept ($C = 1 \iff ((A_1 \in \{0, 1\} \land A_2 = 1) \lor (A_1 = 2 \land A_3 = 0))$), a single conjunctive rule can at most capture one disjunct.
- **Robustness**: Parameter pair $r = 0.8, m = 0.05$ demonstrates the most balanced convergence stability with zero standard deviation across trials.

---

## Variable-Length Multi-Rule GA Experiment (Task c)

### 1. GABIL Modulo-8 Offset-Aligned Crossover
To allow variable numbers of rules without corrupting attribute slots:
1. Choose cut point $pt_1 \in [1, 8k_1 - 1]$ in parent 1.
2. Determine offset: $\text{offset} = pt_1 \pmod 8$.
3. Choose cut point $pt_2$ in parent 2 strictly from loci that share this offset: $pt_2 \equiv \text{offset} \pmod 8$.
4. Splice chromosomes: $C_1 = P_1[:pt_1] + P_2[pt_2:]$ and $C_2 = P_2[:pt_2] + P_1[pt_1:]$. Both offspring preserve exact 8-bit rule boundaries.

### 2. Multi-Rule Benchmark Results ($N = 10$ Independent Monte Carlo Trials)

| Metric | Single-Rule GA (Fixed $L=8$) | Variable-Length GA (Dynamic $L=8k$) | Improvement / Gain |
| :--- | :---: | :---: | :---: |
| **Max Classification Accuracy** | 83.33% | **100.00%** | **+16.67% (Perfect DNF Learning)** |
| **Success Rate (100% Accuracy)** | 0.0% (0/10) | **100.0% (10/10)** | **100% Reliability** |
| **Mean Convergence Generation** | 0.4 +/- 0.5 | 28.6 +/- 8.4 | Deeper exploration of DNF space |
| **Mean Execution Runtime** | 36.42 ms | 507.82 ms | Scales gracefully with variable length |
| **Final Rule Set Complexity** | Fixed $k = 1$ rule | $\bar{k} = 2.4 \pm 0.8$ rules | Parsimony penalty prevents bloat |

### Decoded Optimal Solution
```text
Rule 1: IF (A1 == 2 AND A2 in {0, 1} AND A3 == 0) THEN Class = 1
Rule 2: IF (A1 in {0, 1} AND A2 == 1 AND A3 in {0, 1}) THEN Class = 1
Default: ELSE Class = 0
Empirical Validation: 100.00% Accuracy on full dataset.
```

---

## Synthesis & Comparison Across All GA Problems

| Problem | Domain | Chromosome Structure | Crossover Operator | Fitness Objective | Key Takeaway |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Problem 1** | OneMax ($L=20$) | Fixed 20-bit binary string | 1-point, 2-point, Uniform | Linear bit sum: $\sum b_j$ | Two-point crossover eliminates edge bias; achieves 100% convergence. |
| **Problem 2** | Propositional Rule Induction | Variable $8k$-bit binary string | GABIL Modulo-8 Offset-Aligned | Non-linear: $(\text{correct}(h))^2 - \lambda \max(0, k-2)$ | Single rules cap at 83.33%; variable-length DNF rules achieve 100.0% accuracy. |

---

## References

- De Jong, K. A., & Spears, W. M. (1991). *Learning Concept Classification Rules Using Genetic Algorithms*. Proceedings of the 12th International Joint Conference on Artificial Intelligence (IJCAI).
- De Jong, K. A., Spears, W. M., & Gordon, D. F. (1993). *Using Genetic Algorithms for Concept Learning*. Machine Learning, 13(2-3), 161-188.
- Mitchell, T. M. (1997). *Machine Learning*. Chapter 9: Genetic Algorithms. McGraw-Hill.
- Holland, J. H. (1975). *Adaptation in Natural and Artificial Systems*. University of Michigan Press.
- Course Lecture Notes: `GA.pdf` (HCMUT Machine Learning Coursework).
