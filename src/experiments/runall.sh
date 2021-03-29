# Random Effect Logistic Regression
mlmc-model-evidence/src/experiments$ python random_effect_logistic_regression_convergence.py

# Bayesian Random Effect Logistic Regression
mlmc-model-evidence/src/experiments$ python bayesian_random_effect_logistic_regression_convergence.py

# Sparse Gaussian Process Classification
mlmc-model-evidence/src/experiments$ python gaussian_process_classification_convergence.py



# Train model and take logs
mlmc-model-evidence/src/experiments$ python random_effect_logistic_regression_learning_log.py --output_file  ../../out/random_effect_logistic_regression/learning_log.csv

# Plot the learning curve
mlmc-model-evidence/src/experiments$ python plot_learning_curve.py --input_files ../../out/random_effect_logistic_regression/learning_log.csv --output_file  ../../out/random_effect_logistic_regression/learning_curve.pdf

# Create the table of accuracy
mlmc-model-evidence/src/experiments$ python table_accuracy.py --input_files ../../out/random_effect_logistic_regression/learning_log.csv --output_file  ../../out/random_effect_logistic_regression/accuracy.csv



# Compute and save the efficiency
mlmc-model-evidence/src/experiments$ python random_effect_logistic_regression_efficiency.py --output_file  ../../out/random_effect_logistic_regression/efficiency.csv
# Plot the estimation accuracy
mlmc-model-evidence/src/experiments$ python random_effect_logistic_regression_efficiency.py --input_files ../../out/random_effect_logistic_regression/efficiency.csv --output_file  ../../out/random_effect_logistic_regression/efficiency.pdf

