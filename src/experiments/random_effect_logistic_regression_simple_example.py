import numpy as np
import tensorflow as tf
from random_effect_logistic_regression_utils import generate_data, softplus 

import sys
sys.path.append('../models')
from random_effect_logistic_regression import random_effect_logistic_regression as RELR
from random_effect_logistic_regression import bayesian_random_effect_logistic_regression as BRELR

# Turn GPUs off
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


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
    model = BRELR(D=D, N_total=N_total)
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
            # loss = - model.IWELBO(x, y, n_MC=64) # this gives nested MC estimator
        dparams = g.gradient(loss, params)
        optimizer.apply_gradients(zip(dparams, params))
        if t%200==0:
            print("#iter: {},\tloss: {}".format(t, loss.numpy()))

    print("\n======== training finished ========\n")

    # Print out results
    print("Results:")
    print("                \tbeta0\tbeta1\tbeta2\tbeta3")
    print("ground truth    \t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}".format(alpha, beta0, *list(beta)))
    print("posterior mean  \t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}".format( params[1].numpy(), *list(params[2].numpy()) ))
    print("posterior stddev\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}".format(
        softplus(params[3].numpy()), 
        *[softplus(val) for val in params[4].numpy()] 
    ))
    
if __name__=="__main__":
    main()