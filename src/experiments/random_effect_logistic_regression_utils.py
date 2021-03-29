import numpy as np
import pandas as pd
import datetime
from scipy.stats import bernoulli, norm

sigmoid = lambda x: 1/(1+np.exp(-x))
softplus = lambda x: np.log(1+np.exp(x))

def timestamp():
    now = datetime.datetime.now()
    return now.strftime("%Y%m%d%H%M%S") 

def generate_data(N, D, T, beta0, beta, alpha):
    """
    Genarate N samples of data from the true model with parameter [beta0, beta, alpha]. 
    Returns:
    x: 3-d array of size [N, T, D]
    y: 2-d array of size [N, T]
    z: 1-d array of size [n_MC, N]
    """

    z = np.random.randn(N) * softplus(alpha)**(1/2.)
    x = np.random.randn(N*T*D).reshape([N,T,D])
    y = bernoulli(p=sigmoid(beta0+x@beta+z.reshape([N,1]))).rvs()
    return x,y,z