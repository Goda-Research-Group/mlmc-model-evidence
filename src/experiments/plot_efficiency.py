import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from random_effect_logistic_regression_utils import generate_data, timestamp

# Argparse
parser = argparse.ArgumentParser()
parser.add_argument( "--input_files", default="", type=str,
    help="Input file(s) obtained from 'random_effect_logistic_regression_efficiency.py'.\nAn example usage: `python plot_efficiency.py --input_files \"a.csv b.csv\"`")
parser.add_argument("--output_file", default="../../out/random_effect_logistic_regression/efficiency_{}.pdf".format(timestamp()), type=str,
    help="Output file name. \nAn example usage: `python plot_efficiency.py --output_file example.pdf`")
args = parser.parse_args()

input_files = args.input_files
output_file = args.output_file

if len(input_files) == 0:
    match_input_file = lambda f: ("efficiency" == f[:10]) and (".csv"==f[-4:])
    default_dir = "../../out/random_effect_logistic_regression/"
    input_files = [default_dir + f for f in os.listdir(default_dir) if match_input_file(f)]
    print(input_files)
    if len(input_files) == 0:
        raise(FileNotFoundError("No input file found. Consider running `random_effect_logistic_regression_learning_efficiency.py` first."))

# Load Data
data = []
for f in input_files:
    data.append(pd.read_csv(f))
data = pd.concat(data, axis=0)

# Computational Efficiency
plt.style.use("seaborn-whitegrid")
names = list(data["objective"][data["objective"].duplicated()==False])
for name in names:
    dat = data.loc[data["objective"]==name]
    x = dat["level"]
    y = dat["var_par_reciprocal_runtime"]
    dy = np.exp(dat["stddiv_log_vprr"])
    plt.plot(x, y)
    plt.fill_between(x, y*dy, y/dy, alpha=0.2)
plt.legend([_name for _name in names])
plt.xlabel('Level')
plt.ylabel(r'Variance ( tr(Cov) ) per Reciprocal Runtime')
plt.yscale('log')
plt.savefig(output_file)

