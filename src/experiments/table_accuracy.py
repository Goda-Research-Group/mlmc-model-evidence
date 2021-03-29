import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os

from random_effect_logistic_regression_utils import generate_data, timestamp

# Argparse
parser = argparse.ArgumentParser()
parser.add_argument( "--input_files", default="", type=str,
    help="Input file(s) obtained from 'random_effect_logistic_regression_learning_log.py'.\nAn example usage: `python table_accuracy.py --input_files \"a.csv b.csv\"`")
parser.add_argument("--output_file", default="../../out/random_effect_logistic_regression/accuracy_table_{}.pdf".format(timestamp()), type=str,
    help="Output file name. \nAn example usage: `python table_accuracy.py --output_file accuracy_table.csv`")
parser.add_argument("--last_step", type=int, help="Last step of optimization. The value of squared error at the last step is used to create table.")
args = parser.parse_args()

input_files = args.input_files.split()
output_file = args.output_file
last_step = args.last_step

# Load data for learning curve and estimation accuracy
if len(input_files) == 0: # in case of no input, find appropriate files from `../../out/random_effect_logistic_regression/`.
    match_input_file = lambda f: ("learning_log" == f[:12]) and (".csv"==f[-4:])
    default_dir = "../../out/random_effect_logistic_regression/"
    input_files = [default_dir + f for f in os.listdir(default_dir) if match_input_file(f)]
    print(input_files)
    if len(input_files) == 0:
        raise(FileNotFoundError("No input file found. Consider running `random_effect_logistic_regression_learning_log.py` first."))

data = []
for f in input_files:
    data.append(pd.read_csv(f))
data = pd.concat(data, axis=0)


# Estimation Accuracy

if last_step == None:
    last_steps = dict(data.groupby("objective")["step"].max())
else:
    last_steps = {obj: last_step for obj in data["objective"].unique()}

accuracy_table = {}
for obj, last_step in last_steps.items():
    condition = (data["objective"]==obj) & (data["step"]==last_step)
    last_data = data.loc[condition]
    par_val_dict = {}
    for par in ["alpha", "beta0", "beta1", "beta2", "beta3"]:
        mean = np.mean(last_data[par])
        std = np.std(last_data[par])
        par_val_dict[par]= '{:.5f} ± {:.5f}'.format(mean, std)
    par_val_dict["MSE"] = np.mean(last_data["squared error"])
    accuracy_table[obj] = par_val_dict
accuracy_table = pd.DataFrame(accuracy_table).T
print(accuracy_table)
accuracy_table.to_csv(output_file)
