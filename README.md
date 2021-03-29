# mlmc-model-evidence

The multilevel Monte Carlo methods (MLMC) are sophisticated variance reduction technique of Monte Carlo computation. In the context of the Bayesian statistics, it can be applied to efficiently estimate/maximize the model evidence (log marginal likelihood) when we want to fit the model to a large dataset, as discussed in [efficient debiased variational Bayes by MLMC](https://arxiv.org/abs/2001.04676).

This repository provides examples of application of MLMC to Bayesian statistical models. The implementation is in Python 3 and Tensorflow 2.

## Directory Structure

```
mlmc-model-evidence
├── src/
│   ├── experiments/
│   │   ├── bayesian_random_effect_logistic_regression_convergence.py
│   │   ├── bayesian_random_effect_logistic_regression_simple_example.py
│   │   ├── clean_adult_dataset.py
│   │   ├── gaussian_process_classification_convergence.py
│   │   ├── plot_efficiency.py
│   │   ├── plot_learning_curve.py
│   │   ├── random_effect_logistic_regression_convergence.py
│   │   ├── random_effect_logistic_regression_efficiency.py
│   │   ├── random_effect_logistic_regression_learning_log.py
│   │   ├── random_effect_logistic_regression_utils.py
│   │   ├── random_forest.py
│   │   └── table_accuracy.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── convolutional_variational_autoencoder.py
│   │   ├── gaussian_process_classification.py
│   │   └── random_effect_logistic_regression.py
│   └── utils/
│       └── utils.py
├── out/
│   ├── bayesian_random_effect_logistic_regression/
│   ├── gaussian_process_classification/
│   └── random_effect_logistic_regression/
├── data/
│   └── adults_cleaned.csv
├── README.md
├── requirements.txt
└── LICENSE
```



## Usage

To reproduce the figures in the paper, move to the `./src/experiments` and run the following commands. To run all the commands below, move to the `./src/experiments` and run `runall.sh`.

### Plot Convergence of the Coupled Difference Estimators

```bash
# Random Effect Logistic Regression
mlmc-model-evidence/src/experiments$ python random_effect_logistic_regression_convergence.py

# Bayesian Random Effect Logistic Regression
mlmc-model-evidence/src/experiments$ python bayesian_random_effect_logistic_regression_convergence.py

# Sparse Gaussian Process Classification
mlmc-model-evidence/src/experiments$ python gaussian_process_classification_convergence.py
```

### Plot the Learning Curve for Random Effect Logistic Regression

```bash
# Train model and take logs
mlmc-model-evidence/src/experiments$ python random_effect_logistic_regression_learning_log.py --output_file  ../../out/random_effect_logistic_regression/learning_log.csv
# Plot the learning curve
mlmc-model-evidence/src/experiments$ python plot_learning_curve.py --input_files ../../out/random_effect_logistic_regression/learning_log.csv --output_file  ../../out/random_effect_logistic_regression/learning_curve.pdf
```

### Create the Table of Parameter Estimation Accuracy for Random Effect Logistic Regression

```bash
# Train model and take logs (again)
mlmc-model-evidence/src/experiments$ python random_effect_logistic_regression_learning_log.py --output_file  ../../out/random_effect_logistic_regression/learning_log.csv
# Create the table of accuracy
mlmc-model-evidence/src/experiments$ python table_accuracy.py --input_files ../../out/random_effect_logistic_regression/learning_log.csv --output_file  ../../out/random_effect_logistic_regression/accuracy.csv
```

### Plot the Efficiency of Different Estimators for Random Effect Logistic Regression

```bash
# Compute and save the efficiency
mlmc-model-evidence/src/experiments$ python random_effect_logistic_regression_efficiency.py --output_file  ../../out/random_effect_logistic_regression/efficiency.csv
# Plot the estimation accuracy
mlmc-model-evidence/src/experiments$ python random_effect_logistic_regression_efficiency.py --input_files ../../out/random_effect_logistic_regression/efficiency.csv --output_file  ../../out/random_effect_logistic_regression/efficiency.pdf
```

