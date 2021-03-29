import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import pandas as pd
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
    - predict_prob: predict p(y=1) given x
    - score: compute various scores (e.g. predictive log prob., ELBO, LM-ELBO)
    - plot_convergence: plot the convergence behavior of 2nd moment of difference estimator used in MLMC
    (Other methods are not supposed to be called from outside.)
    '''

    def __init__(self, M=30, N_total=10000):
        self.M = M
        self.N_total = N_total
        self.not_initialized = True
    
    
    def _init_param(self, x, learning_rate):
        N, D = x.shape
        M = self.M
        # z represents inducing points
        self.z = x[np.random.choice(np.arange(N), size=M, replace=False)]

        # alpha,beta are hyper-parameters of ARD kernel 
        self.theta = {
            'alpha': tf.Variable(np.ones([D]), dtype=tf.float64),
            'beta': tf.Variable(1., dtype=tf.float64)
        }
        # m and S (=CholS CholS^T) are mean and covariance of GP at inducing points
        self.phi = {
            'm': tf.Variable(np.zeros([M]), dtype=tf.float64),
            'CholS': tf.Variable(0.1*np.eye(M), dtype=tf.float64)
        }
        self.optimizer = tf.keras.optimizers.Adam(learning_rate)

    
    
    def _init_param_if_needed(self, x, learning_rate=0.005):
        if self.not_initialized:
            self._init_param(x, learning_rate)
            self.not_initialized=False
    
    
    def _ELBO(self, x, y, theta, phi):
        '''
        Inputs:
        x: 2-d array of shape [N,D]
        y: 1-d array of shape [N]
        theta: disctionary of model parameters
        phi: disctionary of variational parameters

        Returns:
        elbo: scalar
        '''

        alpha = theta['alpha']
        beta = theta['beta']
        K = get_K(alpha, beta)

        N = y.shape[0]
        M = self.M

        m = phi['m']
        CholS = phi['CholS']

        # sample u = f_0(z_1,...,z_M) from q
        K_mm = K(self.z, self.z) + 1e-6 * tf.eye(M, dtype=tf.float64)
        CholK_mm = tf.linalg.cholesky(K_mm)

        p_u = tfp.distributions.MultivariateNormalTriL(loc=0., scale_tril=CholK_mm)
        q_u = tfp.distributions.MultivariateNormalTriL(loc=m, scale_tril=CholS)
        u = q_u.sample(N)


        # sample f conditionally given u = f_0(z_1,...,z_M)
        inv_CholK_mm = tf.linalg.inv(CholK_mm)
        inv_K_mm = tf.transpose(inv_CholK_mm)@inv_CholK_mm
        K_nm = K(x, self.z)
        K_mn = tf.transpose(K_nm)

        mean_f = tf.linalg.einsum('ni,ij,nj->n', K_nm, inv_K_mm, u)
        var_f = tf.vectorized_map(lambda x:K(x,x), tf.expand_dims(x, axis=1))
        var_f = tf.reshape(var_f, [N])
        var_f = var_f - tf.linalg.einsum('ni,ij,jn->n', K_nm, inv_K_mm, K_mn)

        q_f = tfp.distributions.Normal(loc=mean_f, scale=var_f)
        f = q_f.sample()
        f = tf.clip_by_value(f, -10, 10)  # For numerical stability

        # compute ELBO estimate
        p_y = tfp.distributions.Bernoulli(logits=f)
        kl_qu_pu = tfp.distributions.kl_divergence(q_u, p_u)
        log_p_y = p_y.log_prob(y)
        elbo = tf.reduce_mean( log_p_y ) - kl_qu_pu / self.N_total
        return elbo

    
    def _predictive_log_likelihood(self, x, y, theta, phi, n_MC=64):

        alpha = theta['alpha']
        beta = theta['beta']
        K = get_K(alpha, beta)

        N = y.shape[0]
        M = self.M

        m = phi['m']
        CholS = phi['CholS']

        # sample u = f_0(z_1,...,z_M) from q
        K_mm = K(self.z, self.z) + 1e-6 * tf.eye(M, dtype=tf.float64)
        CholK_mm = tf.linalg.cholesky(K_mm)

        p_u = tfp.distributions.MultivariateNormalTriL(loc=0., scale_tril=CholK_mm)
        q_u = tfp.distributions.MultivariateNormalTriL(loc=m, scale_tril=CholS)
        u = q_u.sample(N)


        # sample f conditionally given u = f_0(z_1,...,z_M)
        inv_CholK_mm = tf.linalg.inv(CholK_mm)
        inv_K_mm = tf.transpose(inv_CholK_mm)@inv_CholK_mm
        K_nm = K(x, self.z)
        K_mn = tf.transpose(K_nm)

        mean_f = tf.linalg.einsum('ni,ij,nj->n', K_nm, inv_K_mm, u)
        var_f = tf.vectorized_map(lambda x:K(x,x), tf.expand_dims(x, axis=1))
        var_f = tf.reshape(var_f, [N])
        var_f = var_f - tf.linalg.einsum('ni,ij,jn->n', K_nm, inv_K_mm, K_mn)

        q_f = tfp.distributions.Normal(loc=mean_f, scale=var_f)
        f = q_f.sample(n_MC)
        f = tf.clip_by_value(f, -10, 10)  # For numerical stability

        # compute p(y=1)
        p_y = tfp.distributions.Bernoulli(logits=f)
        log_p_y = p_y.log_prob(y)
        log_prob = tf_logmeanexp( log_p_y , axis=0) 
        return log_prob
    
    
    def _LMELBO(self, x, y, theta, phi, n_MC=64):
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
        alpha = theta['alpha']
        beta = theta['beta']
        K = get_K(alpha, beta)

        N = y.shape[0]
        M = self.M

        m = phi['m']
        CholS = phi['CholS']

        # sample u = f_0(z_1,...,z_M) from q
        K_mm = K(self.z, self.z) + 1e-6 * tf.eye(M, dtype=tf.float64)
        CholK_mm = tf.linalg.cholesky(K_mm)

        p_u = tfp.distributions.MultivariateNormalTriL(loc=0., scale_tril=CholK_mm)
        q_u = tfp.distributions.MultivariateNormalTriL(loc=m, scale_tril=CholS)
        u = q_u.sample(N)


        # sample f conditionally given u = f_0(z_1,...,z_M)
        inv_CholK_mm = tf.linalg.inv(CholK_mm)
        inv_K_mm = tf.transpose(inv_CholK_mm)@inv_CholK_mm
        K_nm = K(x, self.z)
        K_mn = tf.transpose(K_nm)

        mean_f = tf.linalg.einsum('ni,ij,nj->n', K_nm, inv_K_mm, u)
        var_f = tf.vectorized_map(lambda x:K(x,x), tf.expand_dims(x, axis=1))
        var_f = tf.reshape(var_f, [N])
        var_f = var_f - tf.linalg.einsum('ni,ij,jn->n', K_nm, inv_K_mm, K_mn)

        q_f = tfp.distributions.Normal(loc=mean_f, scale=var_f)
        # q_f = tfp.distributions.Normal(loc=mean_f, scale=4*var_f)
        # p_f = tfp.distributions.Normal(loc=mean_f, scale=var_f)
        f = q_f.sample(n_MC)
        f = tf.clip_by_value(f, -10, 10)  # For numerical stability

        # compute LMELBO estimate
        p_y = tfp.distributions.Bernoulli(logits=f)
        log_p_y = p_y.log_prob(y)
        # log_p_y = p_y.log_prob(y) + p_f.log_prob(f) - q_f.log_prob(f)
        log_prob_y = tf.reduce_mean( tf_logmeanexp( log_p_y , axis=0) ) 
        kl_qu_pu = tfp.distributions.kl_divergence(q_u, p_u)
        lmelbo = log_prob_y - kl_qu_pu / self.N_total
        return lmelbo
        
    
    def _LMELBO_MLMC(self, x, y, theta, phi, max_level=6, w0=1-2.**(-3/2), b=2, N0=1, randomize=False):
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
        b: scalar. the second moment of the coupled difference estimator (dLMELBO) must decrease at a rate of O(2^(-b*level)).
        randomize: whether to use randomization of MLMC.

        Returns:
        lmelbo: scalar estimate of averaged lmelbo over sample points.
        """
        N = y.shape[0]
        M = self.M
        
        # unpack parameters
        idx = tf.random.shuffle(tf.range(N))
        x = x[idx]
        y = y[idx]
        alpha = theta['alpha']
        beta = theta['beta']
        K = get_K(alpha, beta)

        m = phi['m']
        CholS = phi['CholS']

        # calculate KL divergence of p(u) and q(u) of u = f_0(z_1,...,z_M)
        K_mm = K(self.z, self.z) + 1e-6 * tf.eye(M, dtype=tf.float64)
        CholK_mm = tf.linalg.cholesky(K_mm)

        p_u = tfp.distributions.MultivariateNormalTriL(loc=0., scale_tril=CholK_mm)
        q_u = tfp.distributions.MultivariateNormalTriL(loc=m, scale_tril=CholS)
        kl_qu_pu = tfp.distributions.kl_divergence(q_u, p_u)

        # calculate distribution of f conditionally on u = f_0(z_1,...,z_M)
        u = q_u.sample(N)
        inv_CholK_mm = tf.linalg.inv(CholK_mm)
        inv_K_mm = tf.transpose(inv_CholK_mm)@inv_CholK_mm
        K_nm = K(x, self.z)
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
            raise(Exception("Invarid argument for 'randomize' of function LMELBO_MLMC. It must be True or False."))

        # compute dLMELBO's using disjoint samples at each level and sum them up
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
                lmelbo += self._dconditional_likelihood(x_tmp, y_tmp, mean_f_tmp, var_f_tmp, l, N0) * Ns[l] / N / weights[l]   
            elif randomize==False:
                lmelbo += self._dconditional_likelihood(x_tmp, y_tmp, mean_f_tmp, var_f_tmp, l, N0)

            offset += Ns[l]

        return lmelbo

    
    def _dconditional_likelihood(self, x, y, mean_f, var_f, level, N0=16):

        N = y.shape[0]
        # sample f_n's
        q_f = tfp.distributions.Normal(loc=mean_f, scale=var_f)  # This ensures the coupling of the correction terms
        # q_f = tfp.distributions.Normal(loc=mean_f, scale=4*var_f) # this causes nan in the training
        # p_f = tfp.distributions.Normal(loc=mean_f, scale=var_f)
        n_MC = 2**level * N0
        f = q_f.sample(n_MC)
        f = tf.clip_by_value(f, -10, 10)  # For numerical stability

        # sample conditional likelihoods
        p_y = tfp.distributions.Bernoulli(logits=f)
        log_p_y = p_y.log_prob(y)
        # log_p_y = p_y.log_prob(y) + p_f.log_prob(f) - q_f.log_prob(f)
        log_p_y = tf.reshape(log_p_y, [n_MC, N])

        if level==0:
            dL = tf_logmeanexp(log_p_y, axis=0)
        else:
            dL = tf_logmeanexp(log_p_y, axis=0)\
                    - (1/2.) * tf_logmeanexp(log_p_y[:n_MC//2 , :], axis=0)\
                    - (1/2.) * tf_logmeanexp(log_p_y[ n_MC//2:, :], axis=0)
        return tf.reduce_mean( dL )

    
    def _pointwise_dconditional_likelihood(self, x, y, theta, phi, level, N0=16):

        alpha = theta['alpha']
        beta = theta['beta']
        K = get_K(alpha, beta)

        N = y.shape[0]
        M = self.M

        m = phi['m']
        CholS = phi['CholS']

        # sample u = f_0(z_1,...,z_M) from q
        K_mm = K(self.z, self.z) + 1e-6 * tf.eye(M, dtype=tf.float64)
        CholK_mm = tf.linalg.cholesky(K_mm)

        p_u = tfp.distributions.MultivariateNormalTriL(loc=0., scale_tril=CholK_mm)
        q_u = tfp.distributions.MultivariateNormalTriL(loc=m, scale_tril=CholS)
        u = q_u.sample(N)

        # sample f conditionally given u = f_0(z_1,...,z_M)
        inv_CholK_mm = tf.linalg.inv(CholK_mm)
        inv_K_mm = tf.transpose(inv_CholK_mm)@inv_CholK_mm
        K_nm = K(x, self.z)
        K_mn = tf.transpose(K_nm)

        mean_f = tf.linalg.einsum('ni,ij,nj->n', K_nm, inv_K_mm, u)
        var_f = tf.vectorized_map(lambda x:K(x,x), tf.expand_dims(x, axis=1))
        var_f = tf.reshape(var_f, [N])
        var_f = var_f - tf.linalg.einsum('ni,ij,jn->n', K_nm, inv_K_mm, K_mn)

        q_f = tfp.distributions.Normal(loc=mean_f, scale=var_f)  # This ensures the coupling of the correction terms
        # q_f = tfp.distributions.Normal(loc=mean_f, scale=4*var_f)
        # p_f = tfp.distributions.Normal(loc=mean_f, scale=var_f)
        n_MC = 2**level * N0
        f = q_f.sample(n_MC)
        f = tf.clip_by_value(f, -10, 10)  # For numerical stability

        # compute ELBO estimate
        p_y = tfp.distributions.Bernoulli(logits=f)
        log_p_y = p_y.log_prob(y)
        # log_p_y = p_y.log_prob(y) + p_f.log_prob(f) - q_f.log_prob(f)
        log_p_y = tf.reshape(log_p_y, [n_MC,N])
        if level==0:
            return tf_logmeanexp(log_p_y, axis=0) 
        else:
            return tf_logmeanexp(log_p_y, axis=0)\
                    - (1/2.) * tf_logmeanexp(log_p_y[:n_MC//2 ], axis=0)\
                    - (1/2.) * tf_logmeanexp(log_p_y[ n_MC//2:], axis=0)
    
    
    def plot_convergence(self, x, y, L, N0=1):
        
        def get_conv_stats(level):
            print("evaluating the gradients of dIWELBO at level {}".format(level))
            dlmelbos = self._pointwise_dconditional_likelihood(x, y, self.theta, self.phi, level, N0).numpy()

            return {'mean_dLMELBO':np.mean(dlmelbos), 
                    'mean_abs_dLMELBO':np.mean(np.abs(dlmelbos)), 
                    'mean_squared_dLMELBO':np.mean(dlmelbos**2),
                    'var_dLMELBO':np.var(dlmelbos)}
        
        conv_stats = [get_conv_stats(l) for l in range(L)]
        conv_stats = pd.DataFrame(conv_stats)

        # plot results
        plt.style.use("seaborn-whitegrid")
        plt.plot(conv_stats[['mean_abs_dLMELBO', 'var_dLMELBO']], linewidth=2)

        # plot O(2^{-l}), O(2^{-2l})
        s,t = conv_stats[['mean_abs_dLMELBO', 'var_dLMELBO']].iloc[1]
        plt.plot(s*2.**(1-np.arange(L)), c='grey', linewidth=1)
        plt.plot(t*2.**(1-np.arange(L)*2), c='grey', linewidth=1)

        plt.legend([r'$\mathrm{E} | \Delta \hat{\mathcal{L}}|$', 
                    r'$\mathrm{Var}[\Delta \hat{\mathcal{L}}]$', 
                    r'$O(2^{-\ell}), O(2^{-2\ell})$'])
        plt.xlabel('Level')
        plt.yscale('log')
        
    def plot_convergence_grad(self, x, y, L, N0=1):
        
        def get_grad_conv_stat(level):
            print("evaluating the gradients of dIWELBO at level {}".format(level))
            batch_size = 1
            grad_dlmelbos = []
            for i in range(1000):
                index = np.random.choice(x.shape[0], batch_size)
                x_batch = x[index]
                y_batch = y[index]
                with tf.GradientTape() as g:
                    g.watch([self.theta, self.phi])
                    loss = - self._pointwise_dconditional_likelihood(x_batch, y_batch, self.theta, self.phi, level, N0)
                dtheta, dphi = g.gradient(loss, [self.theta, self.phi])
                gradients = list(dtheta.values())# + list(dphi.values())
                grad_dlmelbos.append(np.concatenate([g.numpy().reshape([-1]) for g in gradients]))
            grad_dlmelbos = np.array(grad_dlmelbos)
            return {'norm_mean_grad_dLMELBO': np.linalg.norm(np.mean(grad_dlmelbos, axis=0)), 
                    'mean_norm_grad_dLMELBO': np.mean(np.linalg.norm(grad_dlmelbos, axis=1)), 
                    'mean_squared_norm_grad_dLMELBO': np.mean(np.linalg.norm(grad_dlmelbos, axis=1)**2),
                    'trace_covariance_grad_dLMELBO': np.sum(np.var(grad_dlmelbos, axis=0))}
        
        conv_stats = [get_grad_conv_stat(l) for l in range(L)]
        conv_stats = pd.DataFrame(conv_stats)

        # plot results
        plt.style.use("seaborn-whitegrid")
        plt.plot(conv_stats[['norm_mean_grad_dLMELBO', 'trace_covariance_grad_dLMELBO']], linewidth=2)

        # plot O(2^{-l}), O(2^{-2l})
        s,t = conv_stats[['norm_mean_grad_dLMELBO', 'trace_covariance_grad_dLMELBO']].iloc[1]
        plt.plot(s*2.**(1-np.arange(L)), c='grey', linewidth=1)
        plt.plot(t*2.**(1-np.arange(L)*2), c='grey', linewidth=1)

        plt.legend([r'$||\mathrm{E} [\nabla (\Delta \hat{\mathcal{L}})]||_2$', 
                    r'$\mathrm{tr}(\mathrm{Cov}[\nabla(\Delta \hat{\mathcal{L}})])$',
                    r'$O(2^{-\ell}), O(2^{-2\ell})$'])
        plt.xlabel('Level')
        plt.yscale('log')
    
    def fit(self, x, y, learning_rate=0.005, n_iter=401, batch_size=1000, objective='ELBO', obj_param={}, verbose=True):
        
        self._init_param_if_needed(x, learning_rate=learning_rate)
        
        if objective=='ELBO':
            obj_func = lambda x,y,theta,phi: self._ELBO(x, y, theta, phi)
        elif objective=='LMELBO':
            obj_func = lambda x,y,theta,phi: self._LMELBO(x, y, theta, phi, **obj_param)
        elif objective=='LMELBO_MLMC':
            obj_func = lambda x,y,theta,phi: self._LMELBO_MLMC(x, y, theta, phi, **obj_param)
        
        print_interval = n_iter//20
        for t in range(n_iter):
            
            index = np.random.choice(x.shape[0], batch_size)
            x_batch = x[index]
            y_batch = y[index]

            with tf.GradientTape() as g:
                g.watch([self.theta, self.phi])
                loss = - obj_func(x_batch, y_batch, self.theta, self.phi)
            dtheta, dphi = g.gradient(loss, [self.theta, self.phi])

            gradients = list(dtheta.values()) + list(dphi.values())
            variables = list(self.theta.values()) + list(self.phi.values())
            self.optimizer.apply_gradients(zip(gradients, variables))
        
            if t%print_interval+1==print_interval and verbose==True:
                print('#iter: {}-{}\t{}'.format(t-print_interval+1, t, self.score(x[:1000], y[:1000]).numpy())) 
    
    def predict_prob(self, x):
        self._init_param_if_needed(x)
        return tf.exp(self._predictive_log_likelihood(x, tf.ones([x.shape[0]]), self.theta, self.phi))
    
    def score(self, x, y, objective='predictive_likelihood', obj_param={}, verbose=True):
        
        self._init_param_if_needed(x)

        if objective=='predictive_likelihood':
            logL = self._predictive_log_likelihood(x, y, self.theta, self.phi, **obj_param)
            return tf.reduce_mean(logL)
        elif objective=='ELBO':
            return self._ELBO(x, y, self.theta, self.phi)
        elif objective=='LMELBO':
            return self._LMELBO(x, y, self.theta, self.phi, **obj_param)
        elif objective=='LMELBO_MLMC':
            return self._LMELBO_MLMC(x, y, self.theta, self.phi, **obj_param)
        else:
            raise NotImplementedError
        
