import numpy as np
import tensorflow as tf
import pandas as pd

from random_effect_logistic_regression_utils import generate_data, softplus, timestamp 

import sys
sys.path.append('../models')
from random_effect_logistic_regression import random_effect_logistic_regression as RELR
from random_effect_logistic_regression import bayesian_random_effect_logistic_regression as BRELR

# Turn GPUs off
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


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

# function for formatting the output results
def expand(key, val):
    # expand {"name":array([1,2,3,4,5])}
    # into {"name1":1, "name2":2, ..., "name5":5}
    if type(val)==np.ndarray:
        return {key+str(i+1): x for i,x in enumerate(val)} 
    else:
        return {key:val} 

    
# function for formatting the output results
def expand_param(param):
    expanded_param = {}
    for key, val in param.items():
        expanded_param.update(expand(key,val))
    return expanded_param

def main():
    ### Initializations
    N_total = 100000
    B,T,D = (1000, 2, 3) if tf.test.is_gpu_available() else (200, 2, 3)

    cost_nmc  = B * 2**9
    cost_mlmc = get_mlmc_cost(B, max_level=9, b=1.8, w0=0.9)
    cost_sumo = B * 9
    B_mlmc = np.math.ceil(B * (cost_nmc / cost_mlmc))
    B_sumo = np.math.ceil(B * (cost_nmc / cost_sumo))

    model = RELR(D=D)
    optimizer = tf.keras.optimizers.Adam(0.005)
    # True model parameters
    alpha = np.float64(1.)
    beta0 = np.float64(0.)
    beta  = np.array([0.25, 0.50, 0.75]) #np.random.randn(D) / np.sqrt(D)
    param0 = {
        'alpha': alpha,
        'beta0': beta0,
        'beta': beta
    }

    X,Y,_ = generate_data(N_total, D, T, beta0, beta, alpha)

    objectives = {
        "signorm":   lambda x, y: model.sigmoid_normal_likelihood(x, y),
        "elbo":      lambda x, y: model.IWELBO(x, y, n_MC=1),
        "iwelbo8":   lambda x, y: model.IWELBO(x, y, n_MC=8),
        "iwelbo64":  lambda x, y: model.IWELBO(x, y, n_MC=64),
        "iwelbo512": lambda x, y: model.IWELBO(x, y, n_MC=512),
        "iwelbo512_mlmc": lambda x, y: model.IWELBO_MLMC(x, y, max_level=9, b=1.8, w0=0.9, randomize=False),
        "iwelbo512_randmlmc": lambda x, y: model.IWELBO_MLMC(x, y, max_level=9, b=1.8, w0=0.9, randomize=True),
       "iwelbo512_sumo": lambda x, y: model.IWELBO_SUMO(x, y, K_max=512)
    }


    n_repeat = 10
    params_repeated = {name:[] for name in objectives.keys()}

    for name, obj in objectives.items():
        alpha_s = []
        beta0_s = []
        beta_s = []
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
            for t in range(2001):

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

                if t%200==0 and i == 0:
                    print("#iter: {},\tloss: {}".format(t, loss.numpy()))

            alpha_s.append(model.alpha.numpy())
            beta0_s.append(model.beta0.numpy())
            beta_s.append(model.beta.numpy())
        print()
        params_repeated[name] = {
                'alpha': np.array(alpha_s),
                'beta0': np.array(beta0_s),
                'beta': np.array(beta_s)
        }

    params = {'ground_truth': expand_param(param0)}
    params['ground_truth'].update({'MSE':0})
    for name in objectives.keys():
        param_repeated = params_repeated[name]
        param_mean   = expand_param({name: array.mean(axis=0) for name, array in  param_repeated.items()})
        param_var = expand_param({name: array.std(axis=0) for name, array in  param_repeated.items()})
        param = {name_:'{:.5f} ± {:.5f}'.format(mean,var**(1/2.)) 
                 for name_, mean, var 
                 in zip( param_mean.keys(), param_mean.values(), param_var.values() )}
        error = [var+(mean-true_mean)**2 
                 for var, mean, true_mean 
                 in zip( param_var.values(), param_mean.values(), params['ground_truth'].values() )]
        MSE = sum(error)
        param.update({'MSE':MSE})
        params.update({name :param})

    data = pd.DataFrame(params).T

    print("\n======== Results ========\n")
    print(data)
    filename = '../../out/random_effect_logistic_regression/MLE_estimation_accuracy_{}.csv'.format(timestamp())
    data.to_csv(filename)
    print("\nSaved the results to:\n{}".format(filename))
    
    
if __name__ == "__main__":
    main()