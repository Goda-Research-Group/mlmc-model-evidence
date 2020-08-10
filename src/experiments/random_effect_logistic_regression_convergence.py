import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.stats import bernoulli, norm
from matplotlib import pyplot as plt

from random_effect_logistic_regression_utils import generate_data, timestamp 

import sys
sys.path.append('../models')
from random_effect_logistic_regression import random_effect_logistic_regression as RELR
from random_effect_logistic_regression import bayesian_random_effect_logistic_regression as BRELR

# Turn GPUs off
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def conv_stats_dIWELBO(level, model, beta0, beta, alpha):
    # Compute dIWELBO (and IWELBO) for each sample and 
    # summarize them into several statistics.
    print("evaluating the dIWELBO at level {}".format(level))
    N0 = 2000000
    N = N0//2**level
    n_MC = 2**level
    x,y,_ = generate_data(N=N, D=3, T=2, beta0=beta0, beta=beta, alpha=alpha)
    q_param = model.laplace_approx(x, y)
    mu, sigma = q_param['mu'], q_param['sigma']
    z = norm(loc=mu, scale=sigma).rvs([n_MC, N])
    
    diwelbos = model.pointwise_dIWELBO(x, y, z, q_param).numpy()
    iwelbos = model.pointwise_IWELBO(x, y, z, q_param).numpy()
    
    return {'mean_dIWELBO':np.mean(diwelbos), 
            'mean_abs_dIWELBO':np.mean(np.abs(diwelbos)), 
            'mean_squared_dIWELBO':np.mean(diwelbos**2),
            'var_dIWELBO':np.var(diwelbos), 
            'var_IWELBO':np.var(iwelbos)}


def conv_stats_grad_dIWELBO(level, model, beta0, beta, alpha, params):
    # Compute the gradient of dIWELBO (and IWELBO) for each sample and 
    # summarize them into several statistics.

    print("evaluating the gradients of dIWELBO at level {}".format(level))

    N0 = 2000000
    N = N0//2**level
    n_MC = 2**level
    x,y,_ = generate_data(N=N, D=3, T=2, beta0=beta0, beta=beta, alpha=alpha)
    
    q_param = model.laplace_approx(x, y)
    mu  = q_param['mu']
    sigma = q_param['sigma']
    z = norm(loc=mu, scale=sigma).rvs([n_MC, N]).T

    # Define a gradient function to be vectorized (vectorization for better performance)
    def get_grad(args):
        # get gradient of dIWELBO (and IWELBO) given one sample
        x_, y_, z_, mu, sigma = args
        q_param = {'mu':mu, 'sigma':sigma}
        z_ = tf.reshape(z_, [-1,1])
        with tf.GradientTape(persistent=True) as g:
            g.watch(params)
            diwelbos = model.pointwise_dIWELBO(x_, y_, z_, q_param)
            iwelbos = model.pointwise_IWELBO(x_, y_, z_, q_param)    
        a = g.gradient(diwelbos, params)
        b = g.gradient(iwelbos, params)
        del g
        return a,b
    
    # Compute the gradient of dIWELBO (and IWELBO) for each sample
    args = [tf.expand_dims(arg, axis=1) for arg in [x, y, z, mu, sigma]]
    grads = tf.vectorized_map(get_grad, args)
        
    grad_diwelbos = tf.concat([
        tf.expand_dims(g,1) if len(g.shape)<2 else g 
        for g in grads[0]
    ], axis=1)#[:D+1]
    grad_iwelbos = tf.concat([
        tf.expand_dims(g,1) if len(g.shape)<2 else g 
        for g in grads[1]
    ], axis=1)#[:D+1]
    
    # return summary statistics of the gradients
    return {'norm_mean_grad_dIWELBO': np.linalg.norm(np.mean(grad_diwelbos, axis=0)), 
            'mean_norm_grad_dIWELBO': np.mean(np.linalg.norm(grad_diwelbos, axis=1)), 
            'mean_squared_norm_grad_dIWELBO': np.mean(np.linalg.norm(grad_diwelbos, axis=1)**2),
            'trace_covariance_grad_dIWELBO': np.sum(np.var(grad_diwelbos, axis=0)), 
            'trace_covariance_grad_IWELBO': np.sum(np.var(grad_iwelbos, axis=0))}
    
    
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
    model = RELR(D=D)
    optimizer = tf.keras.optimizers.Adam(0.005)
    params = [
        model.alpha,
        model.beta0, 
        model.beta,
    ]
    X,Y,_ = generate_data(N_total, D, T, beta0, beta, alpha)

    print("\n======== starting the training ========\n")

    for t in range(1):

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

    print("========= starting the evaluation of convergence ========\n")
    
    ### evaluation of dIWELBO
    L=13
    conv_stats = [conv_stats_dIWELBO(l, model, beta0, beta, alpha) for l in range(L)]
    conv_stats = pd.DataFrame(conv_stats)

    # plot results
    plt.plot(conv_stats[['mean_abs_dIWELBO', 'var_dIWELBO']], linewidth=2)

    # plot O(2^{-l/2}), O(2^{-l}), O(2^{-2l})
    s,t = conv_stats[['mean_abs_dIWELBO', 'var_dIWELBO']].iloc[1]
    plt.plot(s*2.**(1-np.arange(L)), c='grey', linewidth=1)
    plt.plot(t*2.**(1-np.arange(L)*2), c='grey', linewidth=1)

    plt.legend([r'$\mathrm{E} | \Delta \mathrm{IW}$-$\mathrm{ELBO}|$', 
                r'$\mathrm{Var}[\Delta \mathrm{IW}$-$\mathrm{ELBO}]$', 
                r'$O(2^{-\ell}), O(2^{-2\ell})$'])
    plt.xlabel('Level')
    plt.yscale('log')
    # save fig
    filename = '../../out/random_effect_logistic_regression/dIWELBO_convergence_{}.eps'.format(timestamp())
    plt.savefig(filename)
    plt.figure()
    print("saved the results to:\n {}\n".format(filename))
    
    ### evaluation of the gradient of dIWELBO
    L=13
    conv_stats = [conv_stats_grad_dIWELBO(l, model, beta0, beta, alpha, params) for l in range(L)]
    conv_stats = pd.DataFrame(conv_stats)
    
    # plot results
    plt.plot(conv_stats[['norm_mean_grad_dIWELBO', 'trace_covariance_grad_dIWELBO']], linewidth=2)

    # plot O(2^{-l/2}), O(2^{-l}), O(2^{-2l})
    s,t = conv_stats[['norm_mean_grad_dIWELBO', 'trace_covariance_grad_dIWELBO']].iloc[1]
    plt.plot(s*2.**(1-np.arange(L)), c='grey', linewidth=1)
    plt.plot(t*2.**(1-np.arange(L)*2), c='grey', linewidth=1)

    plt.legend([r'$||\mathrm{E} [\nabla (\Delta \mathrm{IW}$-$\mathrm{ELBO})]||_2$', 
                r'$\mathrm{tr}(\mathrm{Cov}[\nabla(\Delta \mathrm{IW}$-$\mathrm{ELBO})])$',
                r'$O(2^{-\ell}), O(2^{-2\ell})$'])
    plt.xlabel('Level')
    plt.yscale('log')
    
    # save fig
    filename = '../../out/random_effect_logistic_regression/dIWELBO_gradients_convergence_{}.eps'.format(timestamp())
    plt.savefig(filename)
    print("saved the results to:\n {}\n".format(filename))
    
    print("========= evaluation of convergence finished ========\n")

    
if __name__=="__main__":
    main()