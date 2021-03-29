import numpy as np
import pandas as pd
import tensorflow as tf
import time
import argparse
from matplotlib import pyplot as plt

from random_effect_logistic_regression_utils import generate_data, timestamp
import sys
sys.path.append('../models')
from random_effect_logistic_regression import random_effect_logistic_regression as RELR
from random_effect_logistic_regression import bayesian_random_effect_logistic_regression as BRELR

# Turn GPUs off
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

EPS = 1e-6


def d(f, params):
    # Take the derivative of f
    # returned value is a function df/d[beta0, beta, alpha]
    def df(x,y,level):
        with tf.GradientTape(persistent=True) as g:
            g.watch(params)
            target = f(x,y,level)
        est0 = g.gradient(target, params)
        est = np.concatenate([e.numpy().reshape([-1]) for e in est0], axis=0)
        return est
    return df

def get_mlmc_cost(N, max_level, b, w0):
    # compute the cost of MLMC estimation 
    # when the size of x (and that of y) is N
    if max_level==0:
        levels = np.array([0])
        weights = np.array([1.])
    else:
        weights = 2.**(-(b+1)/2*np.arange(max_level))
        weights /= sum(weights)
        weights = np.concatenate([[w0], (1-w0)*weights])
        levels = np.arange(max_level+1)
    cost = N * weights[0] + N * sum( weights[1:] * (2**levels[1:] + 2**(levels[1:]-1)) )
    return cost


# Argparse
parser = argparse.ArgumentParser()
parser.add_argument("--output_file", default="../../out/random_effect_logistic_regression/learning_curve_{}.csv".format(timestamp()), type=str,
    help="Output file name. \nAn example usage: `python random_effect_logistic_regression_learning_log.py --output_file example.csv`")
args = parser.parse_args()

input_files = args.input_files.split()
output_file = args.output_file
x_axis = args.x_axis
time_discretization = args.time_discretization



### Initializations
N_total = 100000
B,T,D = (1000, 2, 3) if tf.test.is_gpu_available() else (100, 2, 3)#(100, 2, 3)

cost_nmc  = B * 2**9
cost_mlmc = get_mlmc_cost(B, max_level=9, b=1.8, w0=0.9)
cost_sumo = B * 9
B_mlmc = np.math.ceil(B * (cost_nmc / cost_mlmc))
B_sumo = np.math.ceil(B * (cost_nmc / cost_sumo))


alpha = np.float64(1.)
beta0 = np.float64(0.)
beta  = np.array([0.25, 0.50, 0.75]) #np.random.randn(D) / np.sqrt(D)
model = RELR(D=D)
# True model parameters

X,Y,_ = generate_data(N_total, D, T, beta0, beta, alpha)

objectives = {
    "iwelbo1": lambda x, y: model.IWELBO(x, y, n_MC=1),
    "iwelbo8": lambda x, y: model.IWELBO(x, y, n_MC=8),
    "iwelbo64": lambda x, y: model.IWELBO(x, y, n_MC=64),
    "iwelbo512": lambda x, y: model.IWELBO(x, y, n_MC=512),
    "iwelbo512_mlmc": lambda x, y: model.IWELBO_MLMC(x, y, max_level=9, b=1.8, w0=0.9, randomize=False),
    "iwelbo512_randmlmc": lambda x, y: model.IWELBO_MLMC(x, y, max_level=9, b=1.8, w0=0.9, randomize=True),
    "iwelbo512_sumo": lambda x, y: model.IWELBO_SUMO(x, y, K_max=512),
    "jvi8": lambda x, y: model.JVI_IWELBO(x, y, n_MC=8),
    "jvi64": lambda x, y: model.JVI_IWELBO(x, y, n_MC=64),
    "jvi512": lambda x, y: model.JVI_IWELBO(x, y, n_MC=512),
}

# for parallelization
#obj_id = int(input())
#objectives = {k:objectives[k] for i, k in enumerate(objectives.keys()) if i == obj_id}
#print(objectives)

n_train_steps = {
    "iwelbo1": 2000,
    "iwelbo8": 2000,
    "iwelbo64": 2000,
    "iwelbo512": 17000,
    "iwelbo512_mlmc": 3000,
    "iwelbo512_randmlmc": 3000,
    "iwelbo512_sumo": 2000,
    "jvi8": 2000,
    "jvi64": 2000,
    "jvi512": 17000,
}

data = [] 

n_repeat = 100 # TODO: change to 20
params_repeated = {name:[] for name in objectives.keys()}

for name, obj in objectives.items():
for i in range(n_repeat):
    print("training {}.... #iter:{} ".format(name,i))
    # initialize parameters
    model.beta0 = tf.Variable(0.0,  dtype=tf.float64)
    model.beta  = tf.Variable(np.zeros([model.D]),  dtype=tf.float64)
    model.alpha = tf.Variable(0.0,  dtype=tf.float64)
    # pointers to the parameters of trained model
    params_list = [
        model.beta0, 
        model.beta,
        model.alpha
    ]
    optimizer = tf.keras.optimizers.Adam(0.005)

    # Training
    start = time.time()

    for t in range(n_train_steps[name] + 1):

        # Balance the cost of mlmc and nmc when level=9 (n_MC=512)
        # by changing the batch size adoptively
        if 'mlmc' in name:
            batch = np.random.choice(np.arange(N_total), B_mlmc)
        elif 'sumo' in name:
            batch = np.random.choice(np.arange(N_total), B_sumo)
        else:
            batch = np.random.choice(np.arange(N_total), B)
        x = X[batch]
        y = Y[batch]

        # Train step
        with tf.GradientTape() as g:
            g.watch(params_list)
            loss = - obj(x, y)
        dparams = g.gradient(loss, params_list)
        optimizer.apply_gradients(zip(dparams, params_list))

        # Take a log
        if t%5==0:
            data.append({
                "objective": name,
                "#iter": i,
                "step": t,
                "elapsed time": time.time() - start,
                "alpha": model.alpha.numpy(),
                "beta0": model.beta0.numpy(),
                "beta1": model.beta.numpy()[0],
                "beta2": model.beta.numpy()[1],
                "beta3": model.beta.numpy()[2],
                "squared error": sum(
                    np.concatenate([
                        [alpha - model.alpha.numpy()],
                        [beta0 - model.beta0.numpy()],
                        beta - model.beta.numpy()
                    ]) ** 2
                )
            })

        if t%200==0 and i == 0:
            print("#iter: {},\tloss: {}".format(t, loss.numpy()))

print()


print("\n======== Results ========\n")
data = pd.DataFrame(
data=data,
columns = [
    "objective", "#iter", "elapsed time", "step", 
    "alpha", "beta0", "beta1", "beta2", "beta3", "squared error"
]
)
print(data)
data.to_csv(output_file)
print("\nSaved the results to:\n{}".format(output_file))

