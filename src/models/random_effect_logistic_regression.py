import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import bernoulli, norm

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


    def __init__(self, alpha=None, beta0=None, beta=None, D=None):
        
        self.D = beta.shape[0] if D is None else D
        
        if alpha is None: alpha = 0. 
        if beta0 is None: beta0 = 0. 
        if beta is None:  beta = np.zeros([self.D], dtype=np.float64) 
        
        self.alpha = tf.Variable(alpha, dtype=tf.float64)
        self.beta0 = tf.Variable(beta0, dtype=tf.float64)
        self.beta  = tf.Variable(beta,  dtype=tf.float64)
    
    
    def sigmoid_normal_prob(self, x):
        (N, T, D) = x.shape
        # Compute p(Y=1|X=x) for N samples of x_n
        kappa = 1 / (1 + np.pi*tf.math.softplus(self.alpha)/8)**(1/2)
        return tf.math.sigmoid( kappa * (self.beta0 + tf.reshape( x@tf.reshape(self.beta, [D,1]), [N, T])) )
    
    
    def sigmoid_normal_likelihood(self, x, y):
        # Compute log p(Y=y|X=x) for N samples of (x_n, y_n) and sum them up
        pred_prob = self.sigmoid_normal_prob(x)
        score = tf.reduce_mean(tf.reduce_sum(
            tf.math.log(pred_prob)*y + tf.math.log(1-pred_prob)*(1-y), 
            axis=1))
        return score


    def laplace_approx(self, x, y):
        """
        Compute the mean and the varince of the 
        Laplace approximation of p(z|x,y) for each sample point.

        Arguments:
        x: 3-d array of size [N, T, D]
        y: 2-d array of size [N, T]
        
        Returns:
        mu: 1-d array of size [N]
        sigma: 1-d array of size [N]
        """
        (N, T, D) = x.shape
        z = np.zeros([N, 1])
        alpha = self.alpha.numpy()
        beta0 = self.beta0.numpy()
        beta  = self.beta.numpy()
        
        _sig = lambda z: sigmoid( z + beta0 + x@beta )# utility

        # Newton optimization to calculate the MAP of z|x,y
        for i in range(10):
            sig = _sig(z)
            hessian = 1/softplus(alpha) + np.sum( sig*(1-sig), axis=1, keepdims=True)
            grad    = z/softplus(alpha) + np.sum( sig - y,     axis=1, keepdims=True)
            z -= grad / hessian

        mu = z.reshape([N])
        sigma = (1 / hessian).reshape([N])**(1/2)
        q_params = {'mu':mu, 'sigma':sigma}
        return q_params
    
    
    def pointwise_IWELBO(self, x, y, z, q_param):
        """
        Compute IWELBOs using n_MC inner Monte Carlo samples of Z's at each sample point. 

        Arguments:
        x: 3-d array of size [N, T, D]
        y: 2-d array of size [N, T]
        z: 1-d array of size [n_MC, N]
        q_param['mu']: 1-d array of [N]
        q_param['sigma']: 1-d array of [N]

        Returns:
        iwelbos: 1-d array of size [N]
        """

        (N, T, D), (n_MC, _) = x.shape, z.shape
        y = as_tf_float( tf.reshape(y, [1,N,T]) )
        mu = tf.reshape(q_param['mu'], [1,N])
        sigma = tf.reshape(q_param['sigma'], [1,N])

        y_logits = tf.convert_to_tensor( self.beta0\
                                        + tf.reshape( x@tf.reshape(self.beta, [D,1]), [1, N, T])\
                                        + tf.reshape(z, [n_MC, N, 1]) 
                                       )
        p_y = tfp.distributions.Bernoulli(logits=y_logits)
        p_z = tfp.distributions.Normal(loc=np.zeros([1, N]), scale=tf.math.softplus(self.alpha)**(1/2.))
        q_z = tfp.distributions.Normal(loc=mu, scale=sigma)

        log_prob_ratios = \
            tf.reduce_sum( p_y.log_prob(y), axis=2)\
            + p_z.log_prob(z)\
            - q_z.log_prob(z)

        iwelbos = tf_logmeanexp(log_prob_ratios, axis=0)
        return iwelbos
    
    
    def IWELBO(self, x, y, n_MC):
        """
        Compute IWELBO

        Arguments:
        x: 3-d array of size [N, T, D]
        y: 2-d array of size [N, T]
        q_param['mu']: 1-d array of [N]
        q_param['sigma']: 1-d array of [N]

        Returns:
        iwelbo: scalar value of average of iwelbo's at each sample point.
        """

        q_param = self.laplace_approx(x, y)
        mu = q_param['mu']
        sigma = q_param['sigma']
        N, = mu.shape
        z = norm(loc=mu, scale=sigma).rvs([n_MC, N])
        iwelbo = tf.reduce_mean( self.pointwise_IWELBO(x, y, z, q_param) )
        return iwelbo
    

    def JVI_IWELBO(self, x, y, n_MC):
        """
        Compute first order Jackknife estimator of IWELBO

        Arguments:
        x: 3-d array of size [N, T, D]
        y: 2-d array of size [N, T]
        q_param['mu']: 1-d array of [N]
        q_param['sigma']: 1-d array of [N]

        Returns:
        iwelbo: scalar value of average of iwelbo's at each sample point.
        """

        if n_MC==1:
            return self.IWELBO(x, y, n_MC)

        N,T,D = x.shape
        K = n_MC 

        q_param = self.laplace_approx(x, y)
        mu = tf.reshape(q_param['mu'], [1,N])
        sigma = tf.reshape(q_param['sigma'], [1,N])

        z = norm(loc=mu, scale=sigma).rvs([n_MC, N])
        y = as_tf_float( tf.reshape(y, [1,N,T]) )

        y_logits = tf.convert_to_tensor( self.beta0\
                                        + tf.reshape( x@tf.reshape(self.beta, [D,1]), [1, N, T])\
                                        + tf.reshape(z, [n_MC, N, 1]) 
                                       )
        p_y = tfp.distributions.Bernoulli(logits=y_logits)
        p_z = tfp.distributions.Normal(loc=np.zeros([1, N]), scale=tf.math.softplus(self.alpha)**(1/2.))
        q_z = tfp.distributions.Normal(loc=mu, scale=sigma)

        log_prob_ratios = \
            tf.reduce_sum( p_y.log_prob(y), axis=2)\
            + p_z.log_prob(z)\
            - q_z.log_prob(z)

        iwelbos = tf_logmeanexp(log_prob_ratios, axis=0)
        jvi_iwelbos = K * iwelbos
        '''
        for k in range(K):
            log_probs_tmp = tf.concat([log_prob[:, :k], log_prob[:, k+1:])
            jvi_iwelbos -= (K-1) / K * tf_logmeanexp( log_prob_tmp )
        '''
        jvi_iwelbos -= (K-1) * np.log(K/(K-1))
        jvi_iwelbos -= (K-1) * tf.reduce_mean(
            tf.math.log(tf.expand_dims(tf.math.exp(iwelbos), 0) - (1/K) * tf.math.exp(log_prob_ratios)),
            axis=0
        )
        return tf.reduce_mean(jvi_iwelbos)
    
    
    def pointwise_dIWELBO(self, x, y, z, q_param):
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
        q_param['mu']: 1-d array of [N]
        q_param['sigma']: 1-d array of [N]

        Returns:
        scores: 1-d array of size [N]
        """

        (N, T, D), (n_MC, N) = x.shape, z.shape
        assert np.log2(n_MC)%1==0
        
        if n_MC == 1:
            scores = self.pointwise_IWELBO(x, y, z, q_param)
        else:
            scores = self.pointwise_IWELBO(x, y, z, q_param)
            scores -= (1/2.) * self.pointwise_IWELBO(x, y, z[:n_MC//2 ], q_param)
            scores -= (1/2.) * self.pointwise_IWELBO(x, y, z[ n_MC//2:], q_param)
        return scores
    
    
    def dIWELBO(self, x, y, level):
        """
        Compute average of the coupled differences of IWELBO's with n_MC.
        Differences between "IWELBO with n_MC inner Monte Carlo samples" 
        and "IWELBO with n_MC/2 inner Monte Carlo samples" are taken.

        Note that difference is not taken when n_MC = 1. 
        In that case, average IWELBO with n_MC = 1 is Evaluated.

        Arguments:
        x: 3-d array of size [N, T, D]
        y: 2-d array of size [N, T]
        mu: 1-d array of size [N]
        sigma: 1-d array of size [N]

        Returns:
        score: scalar value of average of differnece of iwelbo's at each sample point (except when n_MC=1).
        """

        n_MC = 2**level
        N,_,_ = x.shape
        q_param = self.laplace_approx(x, y)
        mu = q_param['mu']
        sigma = q_param['sigma']
        z = norm(loc=mu, scale=sigma).rvs([n_MC, N])

        score = tf.reduce_mean( self.pointwise_dIWELBO(x, y, z, q_param) )
        return score
    
    
    def IWELBO_MLMC(self, x, y, max_level=8, w0=1-2.**(-3/2), b=2, randomize=False):
        """
        Compute IWELBO by MLMC

        Arguments:
        x: 3-d array of size [N, T, D]
        y: 2-d array of size [N, T]
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
            if randomize==True:
                iwelbo += self.dIWELBO(x_tmp, y_tmp, level=l) * Ns[i] / N / weights[i]   
            elif randomize==False:
                iwelbo += self.dIWELBO(x_tmp, y_tmp, level=l)

            offset += Ns[i]

        return iwelbo
    
    
    def conditional_IWELBO_SUMO(self, x, y, K):
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
        q_param = self.laplace_approx(x, y)
        mu = q_param['mu']
        sigma = q_param['sigma']
        z = tf.random.normal(mean=mu, stddev=sigma, shape=[K,N], dtype=tf.float64)

        # compute prob ratio of shape [K,N]
        y_logits = self.beta0 + tf.reshape( x @ tf.reshape(self.beta,[D,1]), [1,N,T] ) + tf.reshape(z,[K,N,1]) 
        p_y = tfp.distributions.Bernoulli(logits=y_logits)
        p_z = tfp.distributions.Normal(loc=0, scale=tf.math.softplus(self.alpha)**(1/2.))
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
    
    
    def IWELBO_SUMO(self, x, y, K_max=64):
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
            iwelbo += (1/N) * tf.reduce_sum( self.conditional_IWELBO_SUMO(x_tmp, y_tmp, K) ) 
            offset += cnt

        return iwelbo


class bayesian_random_effect_logistic_regression(random_effect_logistic_regression):
    
    
    def __init__(self, alpha=None, beta0=None, beta=None, D=None, N_total=None):
        
        super(bayesian_random_effect_logistic_regression, self).__init__(alpha, beta0, beta, D)
        
        self.N_total = N_total
        
        # additional parameters which express the variance of the posterior of betas
        self.inv_sp_stddev_beta0 = tf.Variable(0., dtype=tf.float64)
        self.inv_sp_stddev_beta = tf.Variable(np.zeros([self.D]),  dtype=tf.float64)
        
        # parameters for prior of betas
        self.prior_stddev_beta0 = tf.Variable(1.,  dtype=tf.float64)
        self.prior_stddev_beta = tf.Variable(np.ones([self.D]),  dtype=tf.float64)
        
        
    def sample_betas(self, N):
        """
        Sample beta0 and beta from posterior distribution (variational approximation).
        
        Arguments:
        N: the number of samples, usually set to be the batch size.
        
        """
        beta0_stddev = tf.math.softplus( self.inv_sp_stddev_beta0 )
        beta_stddev  = tf.math.softplus( self.inv_sp_stddev_beta )
        beta0 = tf.random.normal(mean=self.beta0, stddev=beta0_stddev, shape=[N,1], dtype=tf.float64)
        beta  = tf.random.normal(mean=self.beta, stddev=beta_stddev,  shape=[N,self.D], dtype=tf.float64)
        return beta0, beta
    
    
    def sigmoid_normal_prob(self, x):
        """
        Compute p(Y=1|X=x, beta) for N samples of x_n's, 
        for betas sampled from posterior
        
        Output:
        prob: of shape [N, T]
        """
        (N, T, D) = x.shape
        beta0, beta = self.sample_betas(N)
        kappa = 1 / (1 + np.pi*tf.math.softplus(self.alpha)/8)**(1/2)
        return tf.math.sigmoid( kappa * ( beta0 + tf.reshape( x@tf.expand_dims(beta, 2), [N, T])) )
    
    
    def sigmoid_normal_likelihood(self, x, y):
        # Compute log p(Y=y|X=x) for N samples of (x_n, y_n) and sum them up
        pred_prob = self.sigmoid_normal_prob(x)
        score = tf.reduce_mean(tf.reduce_sum(
            tf.math.log(pred_prob)*y + tf.math.log(1-pred_prob)*(1-y), 
            axis=1))
        return score


    def laplace_approx(self, x, y):
        """
        Compute the mean and the varince of the 
        Laplace approximation of p(z|x,y) for each sample point.

        Arguments:
        x: 3-d array of size [N, T, D]
        y: 2-d array of size [N, T]
        
        Returns:
        mu: 1-d array of size [N]
        sigma: 1-d array of size [N]
        beta0: 2-d array of [N, 1]
        beta: 2-d array of [N, D]
        """
        (N, T, D) = x.shape
        z = np.zeros([N, 1])
        alpha = self.alpha.numpy()
        beta0, beta = self.sample_betas(N)
        np_beta0 = beta0.numpy()
        np_beta  = beta.numpy()
        
        _sig = lambda z: sigmoid( z + np_beta0 + tf.einsum("NTD, ND->NT",x,np_beta) )# utility

        # Newton optimization to calculate the MAP of z|x,y
        for i in range(10):
            sig = _sig(z)
            hessian = 1/softplus(alpha) + np.sum( sig*(1-sig), axis=1, keepdims=True)
            grad    = z/softplus(alpha) + np.sum( sig - y,     axis=1, keepdims=True)
            z -= grad / hessian

        mu = z.reshape([N])
        sigma = (1 / hessian).reshape([N])**(1/2)   * 2  # When the coupling is not stable, wider proposal distribution makes the coupling more stable.
        q_params = {'mu':mu, 'sigma':sigma, 'beta0':beta0, 'beta':beta}
        return q_params
    
    
    def kl_div_betas(self):
        beta0_stddev = tf.math.softplus( self.inv_sp_stddev_beta0 )
        beta_stddev  = tf.math.softplus( self.inv_sp_stddev_beta )
        kl0 = tf.math.log(self.prior_stddev_beta0 / beta0_stddev) \
              + (beta0_stddev**2 + self.beta0**2) / (2. * self.prior_stddev_beta0**2)\
              - 1./2.
        kl = tf.reduce_sum(
             tf.math.log(self.prior_stddev_beta / beta_stddev) \
              + (beta_stddev**2 + self.beta**2) / (2. * self.prior_stddev_beta**2)\
              - 1./2.
        )
        return kl0 + kl
    
    
    def pointwise_IWELBO(self, x, y, z, q_param):
        """
        Compute IWELBOs using n_MC inner Monte Carlo samples of Z's at each sample point. 

        Arguments:
        x: 3-d array of size [N, T, D]
        y: 2-d array of size [N, T]
        z: 1-d array of size [n_MC, N]
        q_param['mu']: 1-d array of [N]
        q_param['sigma']: 1-d array of [N]
        q_param['beta0']: 2-d array of [N, 1]
        q_param['beta']: 2-d array of [N, D]

        Returns:
        iwelbos: 1-d array of size [N]
        """

        (N, T, D), (n_MC, n) = x.shape, z.shape
        y = as_tf_float( tf.reshape(y, [1,N,T]) )
        mu = tf.reshape(q_param['mu'], [1,N])
        sigma = tf.reshape(q_param['sigma'], [1,N])
        beta0 = q_param['beta0']
        beta  = q_param['beta']

        y_logits = tf.convert_to_tensor( tf.reshape(beta0, [1, N, 1])\
                                        + tf.reshape( x@tf.reshape(beta, [N, D, 1]), [1, N, T])\
                                        + tf.reshape(z, [n_MC, N, 1]) 
                                       )
        p_y = tfp.distributions.Bernoulli(logits=y_logits)
        p_z = tfp.distributions.Normal(loc=np.zeros([1, N]), scale=tf.math.softplus(self.alpha)**(1/2.))
        q_z = tfp.distributions.Normal(loc=mu, scale=sigma)

        log_prob_ratios = \
            tf.reduce_sum( p_y.log_prob(y), axis=2)\
            + p_z.log_prob(z)\
            - q_z.log_prob(z)

        iwelbos = tf_logmeanexp(log_prob_ratios, axis=0)
        return iwelbos
    
        
    def IWELBO_MLMC(self, x, y, max_level=8, w0=1-2.**(-3/2), b=2, randomize=False):
        """
        Compute IWELBO by MLMC

        Arguments:
        x: 3-d array of size [N, T, D]
        y: 2-d array of size [N, T]
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
            if randomize==True:
                iwelbo += self.dIWELBO(x_tmp, y_tmp, level=l) * Ns[i] / N / weights[i]   
            elif randomize==False:
                iwelbo += self.dIWELBO(x_tmp, y_tmp, level=l)

            offset += Ns[i]

        return iwelbo
    
    
    def conditional_IWELBO_SUMO(self, x, y, K):
        """
        Compute IWELBO by SUMO for one sample point, given K

        Arguments:
        x: 3-d array of size [N, T, D]
        y: 2-d array of size [N, T]
        q_param['mu']: 1-d array of [N]
        q_param['sigma']: 1-d array of [N]
        q_param['beta0']: 2-d array of [N, 1]
        q_param['beta']: 2-d array of [N, D]
        K: integer 

        Returns:
        iwelbo: scalar estimate of iwelbo at the given sample point.
        """
        N,T,D = x.shape
        q_param = self.laplace_approx(x, y)
        mu = q_param['mu']
        sigma = q_param['sigma']
        beta0 = q_param['beta0']
        beta  = q_param['beta']
        z = tf.random.normal(mean=mu, stddev=sigma, shape=[K,N], dtype=tf.float64)

        # compute prob ratio of shape [K,N]
        y_logits = tf.reshape( beta0, [1, N, 1]) \
                    + tf.reshape( x @ tf.reshape(beta,[N,D,1]), [1,N,T] ) \
                    + tf.reshape(z,[K,N,1]) 
        p_y = tfp.distributions.Bernoulli(logits=y_logits)
        p_z = tfp.distributions.Normal(loc=0, scale=tf.math.softplus(self.alpha)**(1/2.))
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
    
    
    def IWELBO_SUMO(self, x, y, K_max=64):
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
            iwelbo += (1/N) * tf.reduce_sum( self.conditional_IWELBO_SUMO(x_tmp, y_tmp, K) ) 
            offset += cnt

        return iwelbo

    
    def LMELBO(self, x, y, n_MC):
        return self.IWELBO(x, y, n_MC) - self.kl_div_betas() / self.N_total
    
    
    def LMELBO_MLMC(self, x, y, max_level=8, w0=1-2.**(-3/2), b=2, randomize=False):
        return self.IWELBO_MLMC(x, y, max_level, w0, b, randomize) - self.kl_div_betas() / self.N_total
    
    
    def LMELBO_SUMO(self, x, y, K_max=64):
        return self.IWELBO_SUMO(x, y, K_max) - self.kl_div_betas() / self.N_total
