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
    help="Input file(s) obtained from 'random_effect_logistic_regression_learning_log.py'.\nAn example usage: `python plot_learning_curve.py --input_files \"a.csv b.csv\"`")
parser.add_argument("--output_file", default="../../out/random_effect_logistic_regression/learning_curve_{}.pdf".format(timestamp()), type=str,
    help="Output file name. \nAn example usage: `python plot_learning_curve.py --output_file example.pdf`")
parser.add_argument("--x_axis", default="elapsed time", type=str, choices=["elapsed time", "step"])
parser.add_argument("--time_discretization", default=5, type=int)
args = parser.parse_args()

input_files = args.input_files.split()
output_file = args.output_file
x_axis = args.x_axis
time_discretization = args.time_discretization


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

# Plot Learning Curve
plt.style.use("seaborn-whitegrid")
sns.set_palette("tab10")
data["MSE"] = data["squared error"]
data["elapsed time"] = np.floor(data["elapsed time"] / time_discretization) * time_discretization
sns.lineplot(data=data, x=x_axis, y="MSE", hue="objective")
max_time = float(data[["objective", "elapsed time"]].groupby(by="objective").max().min())
plt.xlim([0, max_time * time_discretization])
plt.yscale("log")
plt.savefig(output_file)
