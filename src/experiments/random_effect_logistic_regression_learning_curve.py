import numpy as np
import pandas as pd
import tensorflow as tf
import time
from matplotlib import pyplot as plt
import seaborn as sns
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

def main():
    ### Initializations
    N_total = 100000
    B,T,D = (1000, 2, 3) if tf.test.is_gpu_available() else (200, 2, 3)

    cost_nmc  = B * 2**9
    cost_mlmc = get_mlmc_cost(B, max_level=9, b=1.8, w0=0.9)
    B_mlmc = np.math.ceil(B * (cost_nmc / cost_mlmc))

    alpha = np.float64(1.)
    beta0 = np.float64(0.)
    beta  = np.array([0.25, 0.50, 0.75]) #np.random.randn(D) / np.sqrt(D)
    model = RELR(D=D)
    optimizer = tf.keras.optimizers.Adam(0.005)
    # True model parameters

    X,Y,_ = generate_data(N_total, D, T, beta0, beta, alpha)

    objectives = {
        "iwelbo512": lambda x, y: model.IWELBO(x, y, n_MC=512),
        "iwelbo512_mlmc": lambda x, y: model.IWELBO_MLMC(x, y, max_level=9, b=1.8, w0=0.9, randomize=False),
        "iwelbo512_randmlmc": lambda x, y: model.IWELBO_MLMC(x, y, max_level=9, b=1.8, w0=0.9, randomize=True),
        "iwelbo512_sumo": lambda x, y: model.IWELBO_SUMO(x, y, K_max=512)
    }
    n_train_steps = {
        "iwelbo512": 2000,
        "iwelbo512_mlmc": 1000,
        "iwelbo512_randmlmc": 1000,
        "iwelbo512_sumo": 3000
    }

    data = [] 

    n_repeat = 30 # TODO: change to 20
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

            # Training
            start = time.time()

            for t in range(n_train_steps[name] + 1):

                # Balance the cost of mlmc and nmc when level=9 (n_MC=512)
                # by changing the batch size adoptively
                if 'mlmc' in name:
                    batch = np.random.choice(np.arange(N_total), B_mlmc)
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
                if t%10==0:
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
                            ]) * 2
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
    filename = '../../out/random_effect_logistic_regression/learning_curve_{}.csv'.format(timestamp())
    data.to_csv(filename)
    print("\nSaved the results to:\n{}".format(filename))

    # Plot the results
    plt.style.use("seaborn-whitegrid")
    data["MSE"] = data["squared error"]
    data["elapsed time"] = np.ceil(data["elapsed time"]/5)*5
    sns.lineplot(data=data, x="elapsed time", y="MSE", hue="objective")
    max_time = float(data[["objective", "elapsed time"]].groupby(by="objective").max().min())
    plt.xlim([0, max_time])
    filename = '../../out/random_effect_logistic_regression/learning_curve_{}.png'.format(timestamp())
    plt.savefig(filename)
    print("saved the results to:\n{}\n".format(filename))
    
if __name__=="__main__":
    main()
