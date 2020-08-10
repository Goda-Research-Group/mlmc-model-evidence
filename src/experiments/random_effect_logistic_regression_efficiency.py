import numpy as np
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
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


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

    for t in range(2001):

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

    results = {'NMC':[], 'MLMC':[], 'RandMLMC':[], 'SUMO':[]}
    runtime = {'NMC':[], 'MLMC':[], 'RandMLMC':[], 'SUMO':[]}

    for name, obj in objectives.items():

        print("evaluating the variance of {}...".format(name))
        for i in range(100):
            results[name].append([])
            x,y,_ = generate_data(N=4000, D=3, T=2, beta0=beta0, beta=beta, alpha=alpha)
            for level in range(L):
                results[name][i].append( d(obj, params)(x,y,level) )

        print("evaluating the runtime of {}...".format(name))
        x,y,_ = generate_data(N=20000, D=3, T=2, beta0=beta0, beta=beta, alpha=alpha)
        for level in range(L):
            # Avoid the memery runout by 
            # manipulating the case of NMC with large n_MC (large level)
            if level>10 and name=='NMC':
                start = time.time()
                for j in range(10):
                    d(obj, params)(*[vec[:200] for vec in [x,y]], level)
                end = time.time()
                runtime[name].append((end - start)*100)
            else:
                start = time.time()
                for j in range(10):
                    d(obj, params)(x,y,level)
                end = time.time()
                runtime[name].append(end - start)
                
    ### Plot and Save the Results
    for ests, rtime in zip(results.values(), runtime.values()):
        var_per_recip_runtime = np.array(ests).var(axis=0).sum(axis=1) * np.array(rtime)
        plt.plot(var_per_recip_runtime)
    plt.legend([name for name in results.keys()])
    plt.xlabel('Level')
    plt.ylabel(r'Variance ( tr(Cov) ) per Reciprocal Runtime')
    plt.yscale('log')
    filename = '../../out/random_effect_logistic_regression/time_variance_efficiency_{}.eps'.format(timestamp())
    plt.savefig(filename)
    print("saved the results to:\n{}\n".format(filename))
    
if __name__=="__main__":
    main()