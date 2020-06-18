import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from matplotlib import pyplot as plt

# Utilities
sigmoid = lambda x: 1/(1+np.exp(-x))
softplus = lambda x: np.log(1+np.exp(x))
as_tf_float = lambda x: tf.cast(x, tf.float64)

def tf_logsumexp(ary, axis=1, keepdims=False):
    return tf.math.reduce_logsumexp(ary, axis=axis, keepdims=keepdims)

def tf_logmeanexp(ary, axis=1, keepdims=False):
    return tf.math.reduce_logsumexp(ary, axis=axis, keepdims=keepdims) \
        - tf.math.log(as_tf_float(ary.shape[axis]))


class random_effect_logistic_regression:
    
    def __init__(self, x, alpha=None, beta0=None, beta=None):
        self.N, self.T, self.D  = x.shape
        
        alpha = 0, if alpha is None
        beta0 = 0, if beta0 is None
        beta = np.zeros([self.D], dtype=np.float64) if beta is None
        
        self.alpha = tf.Variable(alpha, dtype=tf.float64)
        self.beta0 = tf.Variable(beta0, dtype=tf.float64)
        self.beta  = tf.variable(beta,  dtype=tf.float64)
    
    def sigmoid_normal_prob(x):
        # Compute p(Y=1|X=x) for N samples of x_n
        kappa = 1 / (1 + np.pi*tf.math.softplus(self.alpha)/8)**(1/2)
        return tf.math.sigmoid( kappa * (self.beta0 + tf.reshape( x@tf.reshape(self.beta, [self.D,1]), [self.N, self.T])) )
    
    
    def sigmoid_normal_likelihood(x, y):
        # Compute log p(Y=y|X=x) for N samples of (x_n, y_n) and sum them up
        pred_prob = sigmoid_normal_prob(x, self.beta0, self.beta, self.alpha)
        score = tf.reduce_mean(tf.reduce_sum(
            tf.math.log(pred_prob)*y + tf.math.log(1-pred_prob)*(1-y), 
            axis=1))
        return score


    def laplace_approx(x, y, beta0, beta, alpha):
        """
        Compute the mean and the varince of the 
        Laplace approximation of p(z|x,y) for each sample point.

        Arguments:
        x: 3-d array of size [N, T, D]
        y: 2-d array of size [N, T]
        beta0: scalar
        beta: 1-d array of size [D]
        alpha: scalar

        Returns:
        mu: 1-d array of size [N]
        sigma: 1-d array of size [N]
        """

        z = np.zeros([self.N, 1])
        alpha = self.alpha.numpy()
        beta0 = self.beta0.numpy()
        beta  = self.beta.numpy()
        
        _sig = lambda z: sigmoid( z + beta0 + x@beta )

        # Newton optimization to calculate the MAP of z|x,y
        for i in range(10):
            sig = _sig(z)
            hessian = 1/softplus(alpha) + np.sum( sig*(1-sig), axis=1, keepdims=True)
            grad    = z/softplus(alpha) + np.sum( sig - y,     axis=1, keepdims=True)
            z -= grad / hessian

        mu = z.reshape([N])
        sigma = (1 / hessian).reshape([N])**(1/2)
        return mu, sigma
    
    
    def pointwise_IWELBO(x, y, z, beta0, beta, alpha, mu, sigma):
        """
        Compute IWELBOs using n_MC inner Monte Carlo samples of Z's at each sample point. 

        Arguments:
        x: 3-d array of size [N, T, D]
        y: 2-d array of size [N, T]
        z: 1-d array of size [n_MC, N]
        beta0: scalar
        beta: 1-d array of size [D]
        alpha: scalar
        mu: 1-d array of size [N]
        sigma: 1-d array of size [N]

        Returns:
        iwelbos: 1-d array of size [N]
        """

        (N, T, D), (n_MC, n) = x.shape, z.shape
        y = as_tf_float( tf.reshape(y, [1,N,T]) )
        mu = tf.reshape(mu, [1,N])
        sigma = tf.reshape(sigma, [1,N])

        y_logits = tf.convert_to_tensor( beta0\
                                        + tf.reshape( x@tf.reshape(beta, [D,1]), [1, N, T])\
                                        + tf.reshape(z, [n_MC, N, 1]) 
                                       )
        p_y = tfp.distributions.Bernoulli(logits=y_logits)
        p_z = tfp.distributions.Normal(loc=np.zeros([1, N]), scale=tf.math.softplus(alpha)**(1/2.))
        q_z = tfp.distributions.Normal(loc=mu, scale=sigma)

        log_prob_ratio = \
            tf.reduce_sum( p_y.log_prob(y), axis=2)\
            + p_z.log_prob(z)\
            - q_z.log_prob(z)

        iwelbos = tf_logmeanexp(log_prob_ratio, axis=0)
        return iwelbos
    
    
    def IWELBO(x, y, beta0, beta, alpha, mu, sigma, n_MC):
        """
        Compute IWELBO

        Arguments:
        x: 3-d array of size [N, T, D]
        y: 2-d array of size [N, T]
        beta0: scalar
        beta: 1-d array of size [D]
        alpha: scalar
        mu: 1-d array of size [N]
        sigma: 1-d array of size [N]

        Returns:
        iwelbo: scalar value of average of iwelbo's at each sample point.
        """

        N, = mu.shape
        z = norm(loc=mu, scale=sigma).rvs([n_MC, N])
        iwelbo = tf.reduce_mean( pointwise_IWELBO(x, y, z, beta0, beta, alpha, mu, sigma) )
        return iwelbo
    
    
    def pointwise_dIWELBO(x, y, z, beta0, beta, alpha, mu, sigma):
        """
        Compute the coupled differences of IWELBO's at each sample point.
        Differences between "IWELBO with n_MC inner Monte Carlo samples" 
        and "IWELBO with n_MC/2 inner Monte Carlo samples" are taken.

        Note that difference is not taken when n_MC = 1. 
        In that case, IWELBO with n_MC = 1 is Evaluated.

        Arguments:
        x: 3-d array of size [N, T, D]
        y: 2-d array of size [N, T]
        z: 1-d array of size [n_MC, N]
        beta0: scalar
        beta: 1-d array of size [D]
        alpha: scalar
        mu: 1-d array of [N]
        sigma: 1-d array of [N]

        Returns:
        scores: 1-d array of size [N]
        """

        (N, T, D), (n_MC, N) = x.shape, z.shape
        assert np.log2(n_MC)%1==0

        if n_MC == 1:
            scores = pointwise_IWELBO(x, y, z, beta0, beta, alpha, mu, sigma)
        else:
            scores = pointwise_IWELBO(x, y, z, beta0, beta, alpha, mu, sigma)
            scores -= (1/2.) * pointwise_IWELBO(x, y, z[:n_MC//2 ], beta0, beta, alpha, mu, sigma)
            scores -= (1/2.) * pointwise_IWELBO(x, y, z[ n_MC//2:], beta0, beta, alpha, mu, sigma)
        return scores
    
    
    def dIWELBO(x, y, beta0, beta, alpha, mu, sigma, level):
        """
        Compute average of the coupled differences of IWELBO's with n_MC.
        Differences between "IWELBO with n_MC inner Monte Carlo samples" 
        and "IWELBO with n_MC/2 inner Monte Carlo samples" are taken.

        Note that difference is not taken when n_MC = 1. 
        In that case, average IWELBO with n_MC = 1 is Evaluated.

        Arguments:
        x: 3-d array of size [N, T, D]
        y: 2-d array of size [N, T]
        beta0: scalar
        beta: 1-d array of size [D]
        alpha: scalar
        mu: 1-d array of size [N]
        sigma: 1-d array of size [N]

        Returns:
        score: scalar value of average of differnece of iwelbo's at each sample point (except when n_MC=1).
        """

        N, = mu.shape
        n_MC = 2**level
        z = norm(loc=mu, scale=sigma).rvs([n_MC, N])

        score = tf.reduce_mean( pointwise_dIWELBO(x, y, z, beta0, beta, alpha, mu, sigma) )
        return score
    
    
    def IWELBO_MLMC(x, y, beta0, beta, alpha, mu, sigma, max_level=8, w0=1-2.**(-3/2), b=2, randomize=False):
        """
        Compute IWELBO by MLMC

        Arguments:
        x: 3-d array of size [N, T, D]
        y: 2-d array of size [N, T]
        beta0: scalar
        beta: 1-d array of size [D]
        alpha: scalar
        mu: 1-d array of size [N]
        sigma: 1-d array of size [N]
        max_level: integer
        w0: the proportion of total samples in (x,y) used at the level 0.
            in other words, 100*(1-w0) % of the total samples are used for estimating the correction term.
        b: scalar. the second moment of the coupled difference estimator (dIWELBO) must decrease at a rate of O(2^(-b*level)).
        randomize: whether to use randomization of MLMC.

        Returns:
        iwelbo: scalar estimate of average iwelbo over sample points.
        """

        N, T, D = x.shape

        # determine proportions of the number of smaples among levels
        if max_level==0:
            levels = np.array([0])
            weights = np.array([1.])
        else:
            weights = 2.**(-(b+1)/2*np.arange(max_level))
            weights /= sum(weights)
            weights = np.concatenate([[w0], (1-w0)*weights])
            levels = np.arange(max_level+1)

        # determine the N_l's
        if randomize==True:
            Ns = np.random.multinomial(n=N, pvals=weights)    
        elif randomize==False:
            Ns = np.array([np.math.ceil(w*N) for w in weights], dtype=np.int)
            Ns[0] = N - sum(Ns[1:])
        else:
            raise(Exception("Invarid argument for 'randomize' of function IWELBO_MLMC. It must be True or False."))

        # compute dIWELBO's using disjoint samples at each level and sum them up
        offset = 0
        iwelbo = 0
        for i, l in enumerate(levels):
            if Ns[i]==0:
                continue
            x_tmp = x[offset:offset+Ns[i]]
            y_tmp = y[offset:offset+Ns[i]]
            mu_tmp = mu[offset:offset+Ns[i]]
            sigma_tmp = sigma[offset:offset+Ns[i]]

            if randomize==True:
                iwelbo += dIWELBO(x_tmp, y_tmp, beta0, beta, alpha, mu_tmp, sigma_tmp, level=l) * Ns[i] / N / weights[i]   
            elif randomize==False:
                iwelbo += dIWELBO(x_tmp, y_tmp, beta0, beta, alpha, mu_tmp, sigma_tmp, level=l)

            offset += Ns[i]

        return iwelbo
    
    
    def conditional_IWELBO_SUMO(x, y, beta0, beta, alpha, mu, sigma, K):
        """
        Compute IWELBO by SUMO for one sample point, given K

        Arguments:
        x: 3-d array of size [N, T, D]
        y: 2-d array of size [N, T]
        beta0: scalar
        beta: 1-d array of size [D]
        alpha: scalar
        mu: 1-d array of size [N]
        sigma: 1-d array of size [N]
        K: integer 

        Returns:
        iwelbo: scalar estimate of iwelbo at the given sample point.
        """
        N,T,D = x.shape
        z = tf.random.normal(mean=mu, stddev=sigma, shape=[K,N], dtype=tf.float64)

        # compute prob ratio of shape [K,N]
        y_logit = beta0 + tf.reshape( x @ tf.reshape(beta,[D,1]), [1,N,T] ) + tf.reshape(z,[K,N,1]) 
        p_y = tfp.distributions.Bernoulli(logits=y_logit)
        p_z = tfp.distributions.Normal(loc=0, scale=tf.math.softplus(alpha)**(1/2.))
        q_z = tfp.distributions.Normal(loc=mu, scale=sigma)

        log_prob_ratio = \
            tf.reduce_sum( p_y.log_prob(y), axis=2)\
            + p_z.log_prob(z)\
            - q_z.log_prob(z)

        # compute SUMO est.
        ks = tf.reshape( tf.cast( tf.range(0,K) + 1, tf.float64), [K,1])
        cum_iwelbo = tf.math.cumulative_logsumexp(log_prob_ratio, axis=0) - tf.math.log(ks)
        inv_weights = ks
        iwelbo = cum_iwelbo[0,:] + tf.reduce_sum(inv_weights[1:] * (cum_iwelbo[1:] - cum_iwelbo[:K-1]), axis=0)

        return iwelbo
    
    
    def IWELBO_SUMO(x, y, beta0, beta, alpha, mu, sigma, K_max=64):
        """
        Compute IWELBO by MLMC

        Arguments:
        x: 3-d array of size [N, T, D]
        y: 2-d array of size [N, T]
        beta0: scalar
        beta: 1-d array of size [D]
        alpha: scalar
        mu: 1-d array of size [N]
        sigma: 1-d array of size [N]
        K_max: integer 

        Returns:
        iwelbo: scalar estimate of average iwelbo over sample points.
        """

        N,T,D = x.shape

        Us = tf.random.uniform(shape=[N], dtype=tf.float64)
        Ks = tf.minimum(1/Us, tf.cast(K_max, tf.float64))
        Ks = tf.cast(tf.math.floor(Ks), tf.int64)
        unique, _, counts =  tf.unique_with_counts(tf.sort(Ks))

        offset = 0
        iwelbo = 0
        for K, cnt in zip(unique, counts):
            x_tmp = x[offset:offset+cnt]
            y_tmp = y[offset:offset+cnt]
            mu_tmp = mu[offset:offset+cnt]
            sigma_tmp = sigma[offset:offset+cnt]
            iwelbo += (1/N) * tf.reduce_sum( conditional_IWELBO_SUMO(x_tmp, y_tmp, beta0, beta, alpha, mu_tmp, sigma_tmp, K) ) 
            offset += cnt

        return iwelbo
    
    
    