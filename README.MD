# LoRA Hyperparameter Optimisation with Metaheuristic Algorithms

ℹ️ This project was developed collaboratively. Contributors section is listed below.

For more detailed information, please refer to the full project report:  
[Project Report](Project%20Report.pdf)

## Overview

This project studies how to optimise **Low-Rank Adaptation (LoRA)** hyperparameters when fine-tuning a transformer model for text classification under **limited computational resources**. The work uses the **DistilBERT** model on the **dair-ai/emotion** dataset and compares several optimisation approaches within a common experimental framework.

The main focus of the project is not simply to fine-tune a model once, but to treat LoRA configuration as an **optimisation problem**. Each candidate configuration must be built, trained, and evaluated before its quality can be measured. This makes the search process expensive and motivates the use of **metaheuristic optimisation algorithms**.

The study compares:

* Random Search
* Aquila Optimizer (AO)
* Hybrid DPSO-GA
* Blindfolded Spiderman Optimization (BSO)
* MOEA/D-DE for the multi-objective extension

The results show that **BSO achieved the strongest overall performance** in the single-objective setting, reaching the highest validation accuracy while using fewer evaluations than the population-based methods.

---

## Problem Statement

Fine-tuning transformer models is often computationally expensive because even a single training run can require substantial GPU time and memory. This becomes even more difficult when the goal is not just to train one model, but to search over many hyperparameter configurations in order to find a strong one.

To reduce the cost of full fine-tuning, this project uses **Low-Rank Adaptation (LoRA)**. LoRA is a parameter-efficient fine-tuning method that keeps most of the pretrained transformer frozen and inserts trainable low-rank updates into selected parts of the model. This makes adaptation more efficient than updating all model parameters. However, although LoRA reduces training cost, its success depends heavily on how its hyperparameters are chosen.

In this project, the following LoRA-related and training-related hyperparameters are considered:

* learning rate
* warm-up ratio
* LoRA rank
* LoRA scaling factor
* LoRA dropout
* target modules to adapt

These hyperparameters are difficult to optimise for several reasons.

### 1. The search space is mixed

The search space is not purely continuous.

* Some variables are **continuous**, such as learning rate, warm-up ratio, and dropout.
* Some variables are **discrete**, such as LoRA rank and LoRA alpha.
* Some variables are **categorical**, such as whether LoRA is applied only to attention layers or to both attention and feedforward layers.

Because of this, the optimisation problem cannot be handled easily by methods that assume a smooth continuous search space.

### 2. Hyperparameter interactions are non-linear

These hyperparameters do not act independently. A good value for one parameter may depend on the values chosen for others. For example, the effect of LoRA rank may change depending on the target modules or learning rate. This makes the relationship between hyperparameter choices and final model performance highly irregular.

### 3. The objective is expensive to evaluate

To evaluate one candidate solution, the model must be:

* configured with a new LoRA setup
* trained for three epochs
* evaluated on the validation set

This means each fitness evaluation is costly. The search algorithm cannot afford to test a very large number of candidates.

### 4. The objective behaves like a black-box function

The optimisation target in this project is **validation accuracy** after training. There is no direct analytical formula linking the hyperparameters to this accuracy, and gradients with respect to the hyperparameter choices are not available in a useful form for the search process. In practice, the optimiser only sees the final score after each expensive trial.

### 5. The project is compute-constrained

The coursework setting assumes limited GPU availability, using a Google Colab T4 environment. Because of this, the optimisation strategy must work under a **small evaluation budget**. It is not enough for an algorithm to eventually find good solutions; it must do so efficiently.

### Practical optimisation question

The central question of this project is:

> How can we efficiently search the LoRA hyperparameter space for DistilBERT, under a strict GPU budget, in order to achieve strong validation accuracy and stable performance?

This is why the project focuses on **metaheuristic algorithms**. These methods are well suited to:

* black-box optimisation
* mixed-variable search spaces
* noisy and irregular objective landscapes
* limited evaluation budgets

---

## Project Objectives

The work is designed around four main objectives.

### 1. Search efficiently under a limited budget

The first objective is to explore the LoRA hyperparameter space while keeping the number of expensive model evaluations low.

### 2. Compare different optimisation paradigms

The second objective is to compare multiple optimisation styles on the same task and under the same training conditions. This includes:

* a simple baseline
* a population-based method
* a hybrid method
* a trajectory-based method
* a multi-objective method

### 3. Measure both performance and cost

The third objective is not only to compare final accuracy, but also to examine:

* convergence behaviour
* number of evaluations
* training time
* GPU memory usage
* stability across random seeds

### 4. Explore the accuracy-efficiency trade-off

The final objective is to move beyond a single-objective setting and study whether the problem benefits from a multi-objective formulation that balances:

* validation performance
* number of trainable parameters

---

## Why LoRA Hyperparameter Optimisation Matters

LoRA is attractive because it allows efficient adaptation of pretrained transformer models without fully retraining them. However, using LoRA effectively still requires careful tuning.

A weak LoRA configuration may lead to poor accuracy, unstable learning, or unnecessary parameter growth. A strong configuration can deliver competitive performance while keeping fine-tuning lightweight. This makes hyperparameter selection a key practical issue, especially when computational resources are limited.

In other words, LoRA reduces the cost of model adaptation, but it does not remove the need for intelligent search. This project examines how different optimisation algorithms behave in that setting.

---

## Dataset

The experiments use the **dair-ai/emotion** dataset from Hugging Face.

This is a text classification dataset consisting of short English text snippets labelled with one of six emotion classes:

* anger
* disgust
* fear
* joy
* sadness
* surprise

To keep the experiments within the available GPU budget:

* the original validation and test splits are retained
* the training split is reduced to a random subset of **3000 samples**

This keeps the training runs manageable while preserving the same validation procedure for all optimisation methods.

---

## Model and Fine-Tuning Setup

The base model used in the project is:

* **DistilBERT (`distilbert-base-uncased`)**

LoRA adapters are inserted into selected modules of the transformer. Depending on the candidate configuration, adapters are applied to either:

* attention-only modules: `q_lin`, `v_lin`
* attention + feedforward modules: `q_lin`, `v_lin`, `ffn.lin1`, `ffn.lin2`

The backbone transformer remains frozen, while the following are trainable:

* LoRA adapter parameters
* classification head

Each candidate configuration is evaluated by fine-tuning the model for **3 epochs** and computing **validation accuracy**.

---

## Hyperparameter Search Space

All algorithms search over the same mixed hyperparameter space.

| Hyperparameter | Range / Values                      |
| -------------- | ----------------------------------- |
| Learning rate  | Log-uniform in `[1e-6, 5e-4]`       |
| Warm-up ratio  | Continuous in `[0.0, 0.2]`          |
| LoRA rank `r`  | `{2, 4, 8, 16, 32}`                 |
| LoRA alpha `α` | `{8, 16, 32, 64, 128}`              |
| LoRA dropout   | Continuous in `[0.0, 0.3]`          |
| Target modules | `{attention-only, attention + FFN}` |

To make the algorithms comparable, candidate solutions are represented in a common normalised space, and then decoded into concrete LoRA hyperparameters before training.

---

## Optimisation Algorithms

## 1. Random Search

Random Search is used as the baseline.

It samples candidate configurations directly from the search space and evaluates them independently. It is simple to implement and provides a useful reference point, but it does not use information from previous evaluations to improve future search decisions.

This makes it a good baseline for comparison, especially in a search space where evaluations are expensive.

---

## 2. Aquila Optimizer (AO)

Aquila Optimizer is a population-based metaheuristic inspired by eagle hunting behaviour. It maintains multiple candidate solutions at the same time and updates them using different exploration and exploitation operators.

In this project, AO is used to test a population-based search strategy for LoRA tuning. Its role is to explore whether maintaining a set of candidates in parallel can improve search quality in a mixed and noisy hyperparameter space.

---

## 3. Hybrid DPSO-GA

The Hybrid DPSO-GA combines ideas from:

* Particle Swarm Optimization
* Genetic Algorithms

The method represents each candidate as a vector in a continuous normalised space and then decodes it into the actual LoRA hyperparameters. This allows the algorithm to work on mixed discrete-continuous choices through a common representation.

Its design aims to combine:

* the faster directional search of PSO
* the diversity-preserving behaviour of genetic operators such as crossover and mutation

This makes it a reasonable choice for a search space where both exploitation and diversity are important.

---

## 4. Blindfolded Spiderman Optimization (BSO)

BSO is a trajectory-based, single-solution metaheuristic. Instead of maintaining a population, it focuses on one incumbent solution at a time and explores the search space through stepwise trajectories.

This makes BSO especially relevant in a setting where each evaluation is expensive. Rather than spreading the budget across many candidates per iteration, BSO uses its evaluations to refine one path at a time and restarts when progress stops.

In this project, BSO is particularly important because it represents a different optimisation philosophy from the population-based methods.

---

## 5. MOEA/D-DE

For the multi-objective extension, the project uses **MOEA/D-DE**, a decomposition-based multi-objective evolutionary algorithm with differential evolution operators.

Instead of optimising only validation accuracy, the multi-objective formulation considers two objectives:

* minimise validation error
* minimise the number of trainable parameters

This allows the project to study trade-offs between predictive performance and parameter efficiency, which is especially relevant in the context of LoRA.

---

## Experimental Setup

All methods are evaluated within a unified framework so that comparisons remain fair.

### Common settings

* same dataset and train/validation/test splits
* same training subset size
* same base model
* same LoRA search space
* same number of epochs per evaluation
* same validation accuracy fitness measure

### Seeds

Experiments are repeated using multiple random seeds:

* 1
* 42
* 999
* 1234

This allows the study to observe variability across stochastic runs rather than relying on a single result.

### Resource monitoring

In addition to validation accuracy, the project logs:

* training time
* peak GPU memory usage

This is important because the project is not only about maximising performance, but also about understanding the computational cost of the search process.

---

## Results Summary

## Single-objective results

The single-objective comparison evaluates the four methods in terms of accuracy, convergence, evaluation efficiency, and computational cost.

### Reported performance

| Method                             | Best Accuracy | Mean Accuracy | Total Evaluations |
| ---------------------------------- | ------------: | ------------: | ----------------: |
| Random Search                      |        89.40% |        85.16% |                80 |
| Aquila                             |        91.20% |        91.11% |               400 |
| PSO + GA                           |        91.65% |        90.98% |               420 |
| Blindfolded Spiderman Optimization |        94.00% |        93.91% |               102 |

### Main findings

* **BSO achieved the highest best validation accuracy: 94.00%**
* **BSO also had the highest mean performance**
* BSO required far fewer evaluations than Aquila and PSO+GA
* Random Search produced the weakest overall results
* PSO+GA was competitive, but its computational cost was substantially higher
* Aquila was stable, but plateaued below BSO

These results suggest that, in this setting, trajectory-based search was particularly effective under a restricted evaluation budget.

---

## Convergence behaviour

The convergence analysis in the report shows that:

* BSO reached strong validation accuracy very quickly
* Aquila improved early but plateaued at a lower level
* PSO+GA improved more gradually
* Random Search behaved irregularly and was less reliable

This matters because in expensive optimisation problems, the number of evaluations used to reach a good solution can be just as important as the final best score.

---

## Computational cost

The report also compares computational cost using training time and GPU memory usage.

### Average GPU memory usage per evaluation

* Random Search: 1.77 GB
* Aquila: 1.58 GB
* BSO: 1.11 GB
* PSO+GA: 0.94 GB

### Average training time per seed

* Random Search: approximately 1504 s
* Aquila: approximately 7503.9 s
* BSO: approximately 8129.6 s
* PSO+GA: approximately 8899 s

Random Search was the cheapest method, but also the least effective. BSO was not the cheapest, but it delivered the strongest overall balance between accuracy and evaluation efficiency.

---

## Multi-objective Extension

The project extends the single-objective setting to a bi-objective one.

### Objective 1

Minimise validation error:

[
f_1(x) = 1 - \text{validation accuracy}(x)
]

### Objective 2

Minimise the number of trainable parameters:

[
f_2(x) = P(x)
]

This formulation is meaningful because LoRA is intended to be parameter-efficient. A configuration with better accuracy may also require more trainable parameters, so the two objectives can conflict.

### Multi-objective findings

The report shows that MOEA/D-DE identified Pareto-optimal solutions that represent different trade-offs between:

* higher accuracy
* lower parameter count

The best accuracy found in the multi-objective setting was **90.2%**.

This is lower than the best single-objective result, but that does not mean the multi-objective method failed. Its purpose is different: it is designed to provide a set of trade-off solutions rather than maximise accuracy alone.

---

## Key Takeaways

This project shows several important points.

### 1. LoRA tuning is a genuine optimisation problem

Even though LoRA reduces the cost of full model adaptation, its performance still depends strongly on the chosen hyperparameters.

### 2. The search space is difficult

The problem combines continuous, discrete, and categorical choices, and each evaluation requires model training.

### 3. Algorithm design matters under limited budgets

Methods that use the evaluation budget more carefully can perform better than methods that test many candidates at higher cost.

### 4. BSO performed best in this setting

Among the single-objective methods studied, BSO achieved the strongest combination of:

* best accuracy
* mean performance
* fast convergence
* lower evaluation count

### 5. Multi-objective optimisation adds a different perspective

When parameter efficiency matters, it is useful to study trade-offs rather than focus only on the single best accuracy value.

---

## Repository Contents

A typical structure for this project may look like this:

```text
.
├── Algorithms/
│   ├── Aquila Optimizer.ipynb
│   ├── BSO.ipynb
│   ├── Discrete PSO (DPSO) with GA.ipynb
│   ├── MOEA_D_DE.ipynb
│   └── Random Search.ipynb
│
├── Results/
│   ├── ao_results_all_seeds_combined.json
│   ├── bso_blindfolded_all_seeds_merged.json
│   ├── dpso_ga_results_all_seeds.json
│   ├── MOEAD_DE_results.json
│   └── random_search_results.json
│
├── Comparative Study of Algorithms.ipynb
├── Project Report.pdf
├── README.md
```

You should adjust this section so it matches your actual repo structure.

---

## Reproducibility Notes

To keep comparisons fair, all algorithms should use:

* the same dataset split
* the same preprocessing
* the same training subset
* the same fitness definition
* the same number of epochs per evaluation

Because the evaluation process involves model training, results may vary slightly across runs. This is why the report uses multiple seeds and reports average behaviour in addition to best-case outcomes.

---

## Future Work

Possible directions for extending the project include:

* using a larger training budget
* running more evaluations per method
* training for more epochs per trial
* testing on larger transformer models
* exploring additional multi-objective algorithms
* studying how algorithm rankings change when the computational budget is increased

---

## 👥 Contributors
This project was developed collaboratively by:

- [Upasana Das](https://github.com/UpasanaD-99)
- [Shivam Kochhar](https://github.com/ShivamKochh)
- [Nneamaka Oluonye](https://github.com/NneamakaOluonye)
- [Lohit Ashwa](https://github.com/Lohit20)

## 📌 Note
This repository is a personal copy of a group project, uploaded for portfolio purposes.  
All original contributors are credited above.
