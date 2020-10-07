import numpy as np
import pandas as pd
import tensorflow as tf
import time
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


def main():
    # Set Paramters
    N = 1000
    D = 3
    T = 2

    alpha = np.float64(1.)
    beta0 = np.float64(0.)
    beta  = np.array([0.25, 0.50, 0.75]) #np.random.randn(D) / np.sqrt(D)
    param0 = {
        'alpha': alpha,
        'beta0': beta0,
        'beta': beta
    }

    # Train random effect logistic regression 
    # by IWELBO estimated by MLMC estimator

    N_total = 10000 # size of the dataset
    B = 1000 # batch size
    model = BRELR(D=D)
    optimizer = tf.keras.optimizers.Adam(0.005)
    params = [
        model.alpha,
        model.beta0, 
        model.beta,
        model.inv_sp_stddev_beta0,
        model.inv_sp_stddev_beta
    ]
    X,Y,_ = generate_data(N_total, D, T, beta0, beta, alpha)

    print("\n======== starting the training ========\n")

    for t in range(2001):  # TODO: change to 2001

        batch = np.random.choice(np.arange(N_total), B)
        x = X[batch]
        y = Y[batch]
        with tf.GradientTape() as g:
            g.watch(params)
            loss = - model.IWELBO_MLMC(x, y, max_level=9, w0=0.9, randomize=False)
        dparams = g.gradient(loss, params)
        optimizer.apply_gradients(zip(dparams, params))
        if t%200==0:
            print("#iter: {},\tloss: {}".format(t, loss.numpy()))

    print("\n======== training finished ========\n")

    L = 13
    objectives = {
        'NMC':      lambda x,y,level: model.IWELBO(x, y, n_MC=2**level),
        'MLMC':     lambda x,y,level: model.IWELBO_MLMC(x, y, max_level=level, randomize=False),
        'RandMLMC': lambda x,y,level: model.IWELBO_MLMC(x, y, max_level=level, randomize=True),
        'SUMO':     lambda x,y,level: model.IWELBO_SUMO(x, y, K_max=2**level)
    }

    results = {'NMC':[], 'MLMC':[], 'RandMLMC':[], 'SUMO':[]}  # TODO: use all objectives
    runtime = {'NMC':[], 'MLMC':[], 'RandMLMC':[], 'SUMO':[]}  # TODO: use all objectives

    for name, obj in objectives.items():

        print("evaluating the variance of {}...".format(name))
        for level in range(L):
            results[name].append([])
            for i in range(100):  # TODO: changed to 100
                x,y,_ = generate_data(N=4000, D=3, T=2, beta0=beta0, beta=beta, alpha=alpha)
                results[name][level].append( d(obj, params)(x,y,level) )

        print("evaluating the runtime of {}...".format(name))
        for level in range(L):
            x,y,_ = generate_data(N=20000, D=3, T=2, beta0=beta0, beta=beta, alpha=alpha)
            runtime[name].append([])
            for i in range(100):  # TODO: change to 100
                # Avoid the memery runout by 
                # manipulating the case of NMC with large n_MC (large level)
                if level>10 and name=='NMC':
                    start = time.time()
                    d(obj, params)(*[vec[:200] for vec in [x,y]], level)
                    end = time.time()
                    runtime[name][level].append((end - start)*100)
                else:
                    start = time.time()
                    d(obj, params)(x,y,level)
                    end = time.time()
                    runtime[name][level].append(end - start)
                
    ### Save the results
    out_csv = pd.DataFrame(
        columns = ["objective", "level", "var", "runtime", "var_par_reciprocal_runtime", "stddiv_log_vprr"]
    )
    for name, ests, rtime in zip(results.keys(), results.values(), runtime.values()):
        ests, rtime = map(np.array, [ests, rtime])
        out_var = ests.var(axis=1).sum(axis=1)
        out_runtime = rtime.mean(axis=1)
        out_vprr = out_var * out_runtime
        bootstrap_log_vprr = []
        for i in range(100):
            b_ests = ests[:, np.random.choice(100, 100), :]  # TODO: change to 100
            b_rtime = rtime[:, np.random.choice(100, 100)]  # TODO: change to 100
            bootstrap_log_vprr.append(
                np.log(b_ests.var(axis=1).sum(axis=1) * b_rtime.mean(axis=1) + EPS)
            )
            out_stddiv_log_vprr = np.std(bootstrap_log_vprr, axis=0)
        out_csv = out_csv.append(pd.DataFrame({
            "objective": [name]*L, 
            "level": np.arange(L, dtype=float), 
            "var": out_var, 
            "runtime": out_runtime, 
            "var_par_reciprocal_runtime": out_vprr, 
            "stddiv_log_vprr": out_stddiv_log_vprr
        }))
    filename = '../../out/random_effect_logistic_regression/time_variance_efficiency_{}.csv'.format(timestamp())
    out_csv.to_csv(filename)
    print("saved the results to:\n{}\n".format(filename))

    # Plot the results
    plt.style.use("seaborn-whitegrid")
    for name in results.keys():
        dat = out_csv.loc[out_csv["objective"]==name]
        x = dat["level"]
        y = dat["var_par_reciprocal_runtime"]
        dy = np.exp(dat["stddiv_log_vprr"])
        plt.plot(x, y)
        plt.fill_between(x, y*dy, y/dy, alpha=0.2)
    plt.legend([_name for _name in results.keys()])
    plt.xlabel('Level')
    plt.ylabel(r'Variance ( tr(Cov) ) per Reciprocal Runtime')
    plt.yscale('log')
    filename = '../../out/random_effect_logistic_regression/time_variance_efficiency_{}.png'.format(timestamp())
    plt.savefig(filename)
    print("saved the results to:\n{}\n".format(filename))
    
if __name__=="__main__":
    main()