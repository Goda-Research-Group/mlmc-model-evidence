import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from matplotlib import pyplot as plt

# utility functions for numerical stability
as_tf_float = lambda x: tf.cast(x, tf.float64)

def tf_logsumexp(ary, axis=1, keepdims=False):
    return tf.math.reduce_logsumexp(ary, axis=axis, keepdims=keepdims)

def tf_logmeanexp(ary, axis=1, keepdims=False):
    return tf.math.reduce_logsumexp(ary, axis=axis, keepdims=keepdims) \
        - tf.math.log(as_tf_float(ary.shape[axis]))

# define kernel 
def get_K(alpha, beta):
    D = alpha.shape[0]
    sp_alpha = tf.reshape( tf.math.softplus( alpha ), [1,1,D])
    # get kernel function for given hyper-parameters
    def K(x1,x2):
        n1 = x1.shape[0]
        n2 = x2.shape[0]
        x1 = tf.reshape(x1, [n1, 1, D])
        x2 = tf.reshape(x2, [1 ,n2, D])
        return tf.exp(beta - tf.reduce_sum( sp_alpha*(x1-x2)**2, axis=2))
    return K


class gaussian_process_classification:
    '''
    Binary classification by Gaussian Process Classification
    with ARD (autonomous relevance detection) kernel

    Available Methods:
    - fit: fit data to the model 
    - score: compute various scores (e.g. predictive log prob., ELBO, LM-ELBO)
    - plot_convergence: plot the convergence behavior of 2nd moment of difference estimator used in MLMC
    (Other methods are not supposed to be called from outside.)
    '''

    def __init__(self, M=30, N_total=10000):
        self.M = M
        self.N_total = N_total
        self.not_initialized = True
    
    
    def __init_param(self, x):
        N, D = x.shape
        M = self.M
        # z represents inducing points, and alpha,beta are hyper-parameters of ARD kernel 
        self.theta = {
            'z': tf.Variable(x[np.random.choice(np.arange(N), size=M, replace=False)]),
            'alpha': tf.Variable(np.ones([D]), dtype=tf.float64),
            'beta': tf.Variable(1., dtype=tf.float64)
        }
        # m and S (=CholS CholS^T) are mean and covariance of GP at inducing points
        self.phi = {
            'm': tf.Variable(np.zeros([M]), dtype=tf.float64),
            'CholS': tf.Variable(0.1*np.eye(M), dtype=tf.float64)
        }
    
    
    def __init_param_if_needed(self, x):
        if self.not_initialized:
            self.__init_param(x)
            self.not_initialized=False
    
    
    def __ELBO(self, x, y, theta, phi):
        '''
        Inputs:
        x: 2-d array of shape [N,D]
        y: 1-d array of shape [N]
        theta: disctionary of model parameters
        phi: disctionary of variational parameters

        Returns:
        elbo: scalar
        '''

        z = theta['z']
        alpha = theta['alpha']
        beta = theta['beta']
        K = get_K(alpha, beta)

        N = y.shape[0]
        M = z.shape[0]

        m = phi['m']
        CholS = phi['CholS']

        # sample u = f_0(z_1,...,z_M) from q
        K_mm = K(z, z) + 1e-6 * tf.eye(M, dtype=tf.float64)
        CholK_mm = tf.linalg.cholesky(K_mm)

        p_u = tfp.distributions.MultivariateNormalTriL(loc=0., scale_tril=CholK_mm)
        q_u = tfp.distributions.MultivariateNormalTriL(loc=m, scale_tril=CholS)
        u = q_u.sample(N)


        # sample f conditionally given u = f_0(z_1,...,z_M)
        inv_CholK_mm = tf.linalg.inv(CholK_mm)
        inv_K_mm = tf.transpose(inv_CholK_mm)@inv_CholK_mm
        K_nm = K(x, z)
        K_mn = tf.transpose(K_nm)

        mean_f = tf.linalg.einsum('ni,ij,nj->n', K_nm, inv_K_mm, u)
        var_f = tf.vectorized_map(lambda x:K(x,x), tf.expand_dims(x, axis=1))
        var_f = tf.reshape(var_f, [N])
        var_f = var_f - tf.linalg.einsum('ni,ij,jn->n', K_nm, inv_K_mm, K_mn)

        q_f = tfp.distributions.Normal(loc=mean_f, scale=var_f)
        f = q_f.sample()

        # compute ELBO estimate
        p_y = tfp.distributions.Bernoulli(logits=f)
        kl_qu_pu = tfp.distributions.kl_divergence(q_u, p_u)
        elbo = tf.reduce_mean(p_y.log_prob(y)) - kl_qu_pu / self.N_total
        return elbo

    
    def __predictive_log_prob(self, x, y, theta, phi, n_MC):

        z = theta['z']
        alpha = theta['alpha']
        beta = theta['beta']
        K = get_K(alpha, beta)

        N = y.shape[0]
        M = z.shape[0]

        m = phi['m']
        CholS = phi['CholS']

        # sample u = f_0(z_1,...,z_M) from q
        K_mm = K(z, z) + 1e-6 * tf.eye(M, dtype=tf.float64)
        CholK_mm = tf.linalg.cholesky(K_mm)

        p_u = tfp.distributions.MultivariateNormalTriL(loc=0., scale_tril=CholK_mm)
        q_u = tfp.distributions.MultivariateNormalTriL(loc=m, scale_tril=CholS)
        u = q_u.sample(N)


        # sample f conditionally given u = f_0(z_1,...,z_M)
        inv_CholK_mm = tf.linalg.inv(CholK_mm)
        inv_K_mm = tf.transpose(inv_CholK_mm)@inv_CholK_mm
        K_nm = K(x, z)
        K_mn = tf.transpose(K_nm)

        mean_f = tf.linalg.einsum('ni,ij,nj->n', K_nm, inv_K_mm, u)
        var_f = tf.vectorized_map(lambda x:K(x,x), tf.expand_dims(x, axis=1))
        var_f = tf.reshape(var_f, [N])
        var_f = var_f - tf.linalg.einsum('ni,ij,jn->n', K_nm, inv_K_mm, K_mn)

        q_f = tfp.distributions.Normal(loc=mean_f, scale=var_f)
        f = q_f.sample(n_MC)

        # compute LMELBO estimate
        p_y = tfp.distributions.Bernoulli(logits=f)
        log_prob_y = tf.reduce_mean( tf_logmeanexp( p_y.log_prob(y) , axis=0) ) 
        kl_qu_pu = tfp.distributions.kl_divergence(q_u, p_u)
        lmelbo = log_prob_y
        return lmelbo
    
    
    def __LMELBO(self, x, y, theta, phi, n_MC):
        """
        Compute (averaged) LMELBO by Nested MC

        Arguments:
        x: 3-d array of size [N, T, D]
        y: 2-d array of size [N, T]
        theta: 
        phi: 
        n_MC: integer, the number of inner MC samples
        
        Returns:
        lmelbo: scalar estimate of averaged lmelbo over sample points.
        """
        z = theta['z']
        alpha = theta['alpha']
        beta = theta['beta']
        K = get_K(alpha, beta)

        N = y.shape[0]
        M = z.shape[0]

        m = phi['m']
        CholS = phi['CholS']

        # sample u = f_0(z_1,...,z_M) from q
        K_mm = K(z, z) + 1e-6 * tf.eye(M, dtype=tf.float64)
        CholK_mm = tf.linalg.cholesky(K_mm)

        p_u = tfp.distributions.MultivariateNormalTriL(loc=0., scale_tril=CholK_mm)
        q_u = tfp.distributions.MultivariateNormalTriL(loc=m, scale_tril=CholS)
        u = q_u.sample(N)


        # sample f conditionally given u = f_0(z_1,...,z_M)
        inv_CholK_mm = tf.linalg.inv(CholK_mm)
        inv_K_mm = tf.transpose(inv_CholK_mm)@inv_CholK_mm
        K_nm = K(x, z)
        K_mn = tf.transpose(K_nm)

        mean_f = tf.linalg.einsum('ni,ij,nj->n', K_nm, inv_K_mm, u)
        var_f = tf.vectorized_map(lambda x:K(x,x), tf.expand_dims(x, axis=1))
        var_f = tf.reshape(var_f, [N])
        var_f = var_f - tf.linalg.einsum('ni,ij,jn->n', K_nm, inv_K_mm, K_mn)

        q_f = tfp.distributions.Normal(loc=mean_f, scale=var_f)
        f = q_f.sample(n_MC)

        # compute LMELBO estimate
        p_y = tfp.distributions.Bernoulli(logits=f)
        log_prob_y = tf.reduce_mean( tf_logmeanexp( p_y.log_prob(y) , axis=0) ) 
        kl_qu_pu = tfp.distributions.kl_divergence(q_u, p_u)
        lmelbo = log_prob_y - kl_qu_pu / self.N_total
        return lmelbo
    
    
    def __dconditional_likelihood(self, x, y, mean_f, var_f, level):

        N = y.shape[0]
        # sample f_n's
        q_f = tfp.distributions.Normal(loc=mean_f, scale=var_f)
        n_MC = 2**level
        f = q_f.sample(n_MC)

        # sample conditional likelihoods
        p_y = tfp.distributions.Bernoulli(logits=f)
        w = p_y.log_prob(y)
        w = tf.reshape(w, [n_MC,N])

        if level==0:
            dL = tf.reshape(w, [N]) 
        else:
            dL = tf_logmeanexp(w, axis=0)\
                    - (1/2.) * tf_logmeanexp(w[:n_MC//2 ], axis=0)\
                    - (1/2.) * tf_logmeanexp(w[ n_MC//2:], axis=0)
        return tf.reduce_mean( dL )
    
    
    def __LMELBO_MLMC(self, x, y, theta, phi, max_level=8, w0=1-2.**(-3/2), b=2, randomize=False):
        """
        Compute (averaged) LMELBO by MLMC

        Arguments:
        x: 3-d array of size [N, T, D]
        y: 2-d array of size [N, T]
        theta: 
        phi: 
        max_level: integer
        w0: the proportion of total samples in (x,y) used at the level 0.
            in other words, 100*(1-w0) % of the total samples are used for estimating the correction term.
        b: scalar. the second moment of the coupled difference estimator (dIWELBO) must decrease at a rate of O(2^(-b*level)).
        randomize: whether to use randomization of MLMC.

        Returns:
        lmelbo: scalar estimate of averaged lmelbo over sample points.
        """
        # unpack parameters
        idx = tf.random.shuffle(tf.range(x.shape[0]))
        x = x[idx]
        y = y[idx]
        z = theta['z']
        alpha = theta['alpha']
        beta = theta['beta']
        K = get_K(alpha, beta)

        N = y.shape[0]
        M = z.shape[0]

        m = phi['m']
        CholS = phi['CholS']

        # calculate KL divergence of p(u) and q(u) of u = f_0(z_1,...,z_M)
        K_mm = K(z, z) + 1e-6 * tf.eye(M, dtype=tf.float64)
        CholK_mm = tf.linalg.cholesky(K_mm)

        p_u = tfp.distributions.MultivariateNormalTriL(loc=0., scale_tril=CholK_mm)
        q_u = tfp.distributions.MultivariateNormalTriL(loc=m, scale_tril=CholS)
        kl_qu_pu = tfp.distributions.kl_divergence(q_u, p_u)

        # calculate distribution of f conditionally on u = f_0(z_1,...,z_M)
        u = q_u.sample(N)
        inv_CholK_mm = tf.linalg.inv(CholK_mm)
        inv_K_mm = tf.transpose(inv_CholK_mm)@inv_CholK_mm
        K_nm = K(x, z)
        K_mn = tf.transpose(K_nm)

        mean_f = tf.linalg.einsum('ni,ij,nj->n', K_nm, inv_K_mm, u)
        var_f = tf.vectorized_map(lambda x:K(x,x), tf.expand_dims(x, axis=1))
        var_f = tf.reshape(var_f, [N])
        var_f = var_f - tf.linalg.einsum('ni,ij,jn->n', K_nm, inv_K_mm, K_mn)

        # determine proportions of the number of samples among levels
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
        lmelbo = - kl_qu_pu / self.N_total
        for l in levels:
            if Ns[l]==0:
                continue
            x_tmp = x[offset:offset+Ns[l]]
            y_tmp = y[offset:offset+Ns[l]]
            mean_f_tmp = mean_f[offset:offset+Ns[l]]
            var_f_tmp = var_f[offset:offset+Ns[l]]

            if randomize==True:
                lmelbo += self.__dconditional_likelihood(x_tmp, y_tmp, mean_f_tmp, var_f_tmp, level=l) * Ns[l] / N / weights[l]   
            elif randomize==False:
                lmelbo += self.__dconditional_likelihood(x_tmp, y_tmp, mean_f_tmp, var_f_tmp, level=l)

            offset += Ns[l]

        return lmelbo

    
    def __pointwise_dconditional_likelihood(self, x, y, theta, phi, level):

        z = theta['z']
        alpha = theta['alpha']
        beta = theta['beta']
        K = get_K(alpha, beta)

        N = y.shape[0]
        M = z.shape[0]

        m = phi['m']
        CholS = phi['CholS']

        # sample u = f_0(z_1,...,z_M) from q
        K_mm = K(z, z) + 1e-6 * tf.eye(M, dtype=tf.float64)
        CholK_mm = tf.linalg.cholesky(K_mm)

        p_u = tfp.distributions.MultivariateNormalTriL(loc=0., scale_tril=CholK_mm)
        q_u = tfp.distributions.MultivariateNormalTriL(loc=m, scale_tril=CholS)
        u = q_u.sample(N)

        # sample f conditionally given u = f_0(z_1,...,z_M)
        inv_CholK_mm = tf.linalg.inv(CholK_mm)
        inv_K_mm = tf.transpose(inv_CholK_mm)@inv_CholK_mm
        K_nm = K(x, z)
        K_mn = tf.transpose(K_nm)

        mean_f = tf.linalg.einsum('ni,ij,nj->n', K_nm, inv_K_mm, u)
        var_f = tf.vectorized_map(lambda x:K(x,x), tf.expand_dims(x, axis=1))
        var_f = tf.reshape(var_f, [N])
        var_f = var_f - tf.linalg.einsum('ni,ij,jn->n', K_nm, inv_K_mm, K_mn)

        q_f = tfp.distributions.Normal(loc=mean_f, scale=var_f)
        n_MC = 2**level
        f = q_f.sample(n_MC)

        # compute ELBO estimate
        p_y = tfp.distributions.Bernoulli(logits=f)
        w = p_y.log_prob(y)
        w = tf.reshape(w, [n_MC,N])
        if level==0:
            return tf_logmeanexp(w, axis=0) 
        else:
            return tf_logmeanexp(w, axis=0)\
                    - (1/2.) * tf_logmeanexp(w[:n_MC//2 ], axis=0)\
                    - (1/2.) * tf_logmeanexp(w[ n_MC//2:], axis=0)
    
    
    def plot_convergence(self, x,y,max_level):
        dcond_L = lambda l: self.__pointwise_dconditional_likelihood(x, y, self.theta, self.phi, level=l).numpy()
        plt.plot([np.mean(dcond_L(l)**2) for l in range(10)])
        plt.yscale('log')
        plt.xlabel('level')
        plt.ylabel(r'$\mathrm{E}||\nabla\ (\Delta \mathrm{LM}$-${ELBO})\ ||_2^2$')
        
    
    def fit(self, x, y, learning_rate=0.01, n_iter=401, objective='ELBO', obj_param={}, verbose=True):
        
        self.__init_param_if_needed(x)
        
        if objective=='ELBO':
            obj_func = lambda x,y,theta,phi: self.__ELBO(x, y, theta, phi)
        elif objective=='LMELBO':
            obj_func = lambda x,y,theta,phi: self.__LMELBO(x, y, theta, phi, **obj_param)
        elif objective=='LMELBO_MLMC':
            obj_func = lambda x,y,theta,phi: self.__LMELBO_MLMC(x, y, theta, phi, **obj_param)
        
        optimizer = tf.keras.optimizers.Adam(learning_rate)
        losses = []
        print_interval = n_iter//20
        for t in range(n_iter):

            with tf.GradientTape() as g:
                g.watch([self.theta, self.phi])
                loss = - obj_func(x, y, self.theta, self.phi)
            dtheta, dphi = g.gradient(loss, [self.theta, self.phi])

            gradients = list(dtheta.values()) + list(dphi.values())
            variables = list(self.theta.values()) + list(self.phi.values())
            optimizer.apply_gradients(zip(gradients, variables))

            losses.append(loss.numpy())
            if t%print_interval+1==print_interval and verbose==True:
                print('#iter: {}-{}\t{}'.format(t-print_interval+1, t, np.mean(losses)))
                losses = []
    
    
    def score(self, x, y, objective='predictive_log_prob', obj_param={'n_MC':64}, verbose=True):
        
        self.__init_param_if_needed(x)

        if objective=='predictive_log_prob':
            return self.__predictive_log_prob(x, y, self.theta, self.phi, **obj_param)
        elif objective=='ELBO':
            return self.__ELBO(x, y, self.theta, self.phi)
        elif objective=='LMELBO':
            return self.__LMELBO(x, y, self.theta, self.phi, **obj_param)
        elif objective=='LMELBO_MLMC':
            return self.__LMELBO_MLMC(x, y, self.theta, self.phi, **obj_param)
        