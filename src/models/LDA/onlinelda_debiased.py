import tensorflow as tf
import numpy as np
from scipy.stats import loggamma
from scipy.special import gammaln, psi, polygamma

from . import onlinelda as olda


digamma = psi
trigamma = lambda x:polygamma(n=1, x=x)

tf_float_type = tf.float64
np_float_type = np.float64

as_tf_float = lambda x: tf.cast(x, tf_float_type)

tf_gammaln = tf.math.lgamma
tf_digamma = tf.math.digamma

softplus = lambda x: x + np.log(1 + np.exp(-x))
softplus_inv = lambda x: x + np.log(1 - np.exp(-x))
sigmoid = lambda x: 1 / (1 + np.exp(-x))


# for numerical stability
# apply log(sum(exp(x))) or log(mean(exp(x))) along the selected axis 
def logsumexp(ary, axis=1, keepdims=False):
    # for numerical stability
    _max = ary.max(axis=axis, keepdims=True)
    out = np.log( np.sum( np.exp(ary - _max) ,axis=axis, keepdims=keepdims) ) + _max
    if keepdims==False:
        return out.squeeze(axis=axis)
    elif keepdims==True:
        return out
    else:
        print("keepdims must be bool. Returning None")
        return None

def tf_logsumexp(ary, axis=1, keepdims=False):
    return tf.math.reduce_logsumexp(ary, axis=axis, keepdims=keepdims)

def tf_logmeanexp(ary, axis=1, keepdims=False):
    return tf.math.reduce_logsumexp(ary, axis=axis, keepdims=keepdims) \
        - tf.math.log(as_tf_float(ary.shape[axis]))

def dirichlet_natgrad(alpha, dy_dalpha, K=None):
    '''
    Compute natural gradient of alpha for dirichlet distribution
    
    Argument:
    alpha: Matrix (stacked vectors) representing parameters of n_param dirichlet distributions, 
    whose shape is [n_param, n_dim] (or [1] when the dirichlet parameters are the same for each component).
    dy_dalpha: Matrix (stacked vectors) representing the gradient with respect to the parameters, 
    whose shape is [n_param, n_dim] (or [1] when the dirichlet parameters are the same for each component).

    Returns:
    natgrad: Natural gradient of the parameters of the dirichlet distributions.
    '''
    if type(alpha)==np.ndarray:
        trigamma_alpha = trigamma(alpha) # shape: [n_param, n_dim]
        trigamma_alpha0 = trigamma(np.sum(alpha, axis=1, keepdims=True)) # shape: [n_param, 1]
        natgrad = 0 # of shape [n_param, n_dim]
        natgrad += trigamma_alpha**(-1) * dy_dalpha
        natgrad += trigamma_alpha**(-1)\
                * np.sum(trigamma_alpha**(-1) * dy_dalpha, axis=1, keepdims=True)\
                / (trigamma_alpha0**(-1) - np.sum(trigamma_alpha**(-1), axis=1, keepdims=True))
        return natgrad

    else:
        # when the dirichlet parameters are the same for each component
        assert type(alpha)==np_float_type
        cov = K*trigamma(alpha) - K**2 * trigamma(K*alpha)
        return dy_dalpha / cov
    
def dirichlet_KL(alpha, beta):
    '''
    Compute the KL divergence between Dir(alpha_i) and Dir(beta_i) for i = 1,..., n_param.
    
    Arguments:
    alpha: array of shape [n_param, n_dim]
    beta: array of shape [n_param, n_dim]
    
    Returns:
    KL: sum of KL divergences
    '''
    KL = 0 # of shape [n_param]
    KL += gammaln(alpha.sum(axis=1)) - gammaln(alpha).sum(axis=1)
    KL += -gammaln(beta.sum(axis=1)) + gammaln(beta).sum(axis=1) 
    KL += ((alpha - beta) * (digamma(alpha) - digamma(alpha.sum(axis=1, keepdims=True)))).sum(axis=1)
    return sum(KL)

def tf_dirichlet_expectation(alpha):
    """
    For a vector theta ~ Dir(alpha), compute E[log(theta)] given alpha.
    
    Arguments:
    alpha: Tensor of parameters of n_params different dirichlet distributions whose shape is:[n_params, n_dim] or [n_dim]. 


    Returns:
    : Expectation of log of samples from the dirichlet distributions
    """
    assert len(alpha.shape)<=2
    if (len(alpha.shape) == 1):
        return tf_digamma(alpha) - tf_digamma(tf.reduce_sum(alpha, keepdims=True))
    return tf_digamma(alpha) - tf_digamma(tf.reduce_sum(alpha, axis=1, keepdims=True))

def dirichlet_sample(alpha, n_MC):
    """
    Sample log of dirichlet distribution, whose dimetion is n_dim.
    We sample from n_params diferrent parameter settings, for n_MC times each.
    Here, we sample the logarithm value for the sake of numerical stability.
    
    Arguments:
    alpha: Tensor of parameters of n_params different dirichlet distributions whose shape is:[n_params, n_dim] or [n_dim]. 
    n_MC: the # of Monte Carlo sample
    
    Returns: 
    log_dir: Tensor of log of MC-samples log(theta) for theta ~ Dir(alpha). The shape of this tensor is [n_MC, n_params, n_dim].
    """
    
    # make sure that dim(ary)<=2
    assert len(alpha.shape)<=2
    # handle cases where n_params = 1
    flag = (len(alpha.shape)==1)
    if flag==True:
        alpha = np.expand_dims(alpha, axis=0)
        
    (n_params, n_dim) = alpha.shape
    # for the sampling scheme of log of gamma distributions below, 
    # see https://stats.stackexchange.com/questions/7969/how-to-quickly-sample-x-if-expx-gamma  
    log_u = np.log(np.random.uniform(size=[n_MC, n_params, n_dim]))
    log_gammas = loggamma.rvs(c=alpha+1, size=[n_MC, n_params, n_dim]) + log_u / np.expand_dims(alpha, axis=0)
    log_dir = log_gammas - logsumexp(log_gammas, axis=2, keepdims=True)
    
    # handle cases where n_params = 1
    if flag==True:
        log_dir = log_dir.reshape([n_MC, n_dim]) 
    
    return log_dir

def tf_dirichlet_sample(alpha, n_MC):
    """
    Sample log of dirichlet distribution, whose dimetion is n_dim.
    We sample from n_params diferrent parameter settings, for n_MC times each.
    Here, we sample the logarithm value for the sake of numerical stability.
    
    Arguments:
    alpha: Tensor of parameters of n_params different dirichlet distributions whose shape is:[n_params, n_dim] or [n_dim]. 
    n_MC: the # of Monte Carlo sample
    
    Returns: 
    log_dir: Tensor of log of MC-samples log(theta) for theta ~ Dir(alpha). The shape of this tensor is [n_MC, n_params, n_dim].
    """
    
    # make sure that dim(ary)<=2
    assert len(alpha.shape)<=2
    # handle cases where n_params = 1
    flag = (len(alpha.shape)==1)
    if flag==True:
        alpha = tf.expand_dims(alpha, axis=0)
        
    (n_params, n_dim) = alpha.shape
    # for the sampling scheme of log of gamma distributions below, 
    # see https://stats.stackexchange.com/questions/7969/how-to-quickly-sample-x-if-expx-gamma  
    log_u = tf.math.log(tf.random.uniform(shape=[n_MC, n_params, n_dim], dtype=tf_float_type))
    log_gammas = tf.math.log(tf.random.gamma(alpha=alpha+1, shape=[n_MC], dtype=tf_float_type)) + log_u / tf.expand_dims(alpha, axis=0)
    log_dir = log_gammas - tf_logsumexp(log_gammas, axis=2, keepdims=True)
    
    # handle cases where n_params = 1
    if flag==True:
        log_dir = tf.reshape(log_dir, [n_MC, n_dim]) 
    
    return log_dir


class OnlineLDA_deviased(olda.OnlineLDA):
    
    def update_param(self, wordids, wordcts, objective="ELBO"):
        """
        Update model parameters together with variational parameters. 
        """ 
        assert len(wordids)>0
    
        gamma = lda.update_lambda(wordids, wordcts)
        
        with tf.GradientTape() as g:
            alpha = tf.Variable(self._alpha)
            eta = tf.Variable(self._eta)
            
            if objective=="ELBO":
                score = self._approx_ELBO(wordids, wordcts, gamma, alpha, eta, self._lambda)
            elif objective[:6]=="LMELBO": # ex. "LMELBO16" 
                n_MC = int(objective[6:])
                score = self._approx_LMELBO(wordids, wordcts, gamma, alpha, eta, self._lambda, n_MC)
            else:
                print("Unknown objective givein to grad_update. Returning None.")
                return None

        dy_dalpha, dy_deta = [dy.numpy() for dy in g.gradient(score, [alpha, eta])]

        alpha_tilde = softplus_inv(self._alpha)
        eta_tilde = softplus_inv(self._eta)

        alpha_tilde += self._rhot*0.00005 * (1 / sigmoid(alpha_tilde)) * dirichlet_natgrad(self._alpha, dy_dalpha, self._K)
        eta_tilde += self._rhot*0.1 * (1 / sigmoid(eta_tilde)) * dirichlet_natgrad(self._eta, dy_deta, self._V)

        self._alpha = softplus(alpha_tilde)
        self._eta = softplus(eta_tilde)

        return gamma, score.numpy()
    
    def _approx_ELBO(self, wordids, wordcts, gamma, alpha, eta, lambd):
        """
        Estimates the lower bound of log p(w) over *all documents* using only
        the documents passed in as "docs." gamma is the set of parameters
        to the variational distribution q(theta) corresponding to the
        set of documents passed in.

        The output of this function is going to be noisy, but can be
        useful for assessing convergence.
        """
        batchD = len(wordids)

        score = 0
        Elogtheta = olda.dirichlet_expectation(gamma)
        Elogbeta = tf_dirichlet_expectation(lambd)

        # E_q[log p(x, z| theta, beta) - log q(z)]
        for d in range(0, batchD):
            gammad = gamma[d, :]
            ids = wordids[d]
            cts = np.array(wordcts[d])
            temp = Elogtheta[d, :].reshape([-1,1]) + tf.gather(Elogbeta, ids, axis=1)
            phinorm = tf_logsumexp(temp, 0) 
            score += tf.reduce_sum(cts * phinorm)
            
        # E[log p(theta | alpha) - log q(theta | gamma)]
        score += tf.reduce_sum((alpha - gamma)*Elogtheta)
        score += tf.reduce_sum(gammaln(gamma) - tf_gammaln(alpha))
        score += tf.reduce_sum(tf_gammaln(alpha*self._K) - gammaln(np.sum(gamma, 1)))

        # Compensate for the subsampling of the population of documents
        score = score * self._D / batchD
    
        # E[log p(beta | eta) - log q (beta | lambda)]
        score += tf.reduce_sum((eta-lambd)*Elogbeta)
        score += tf.reduce_sum(tf_gammaln(lambd) - tf_gammaln(eta)) \
                 + tf.reduce_sum(tf_gammaln(eta*self._V) - tf_gammaln(tf.reduce_sum(lambd, 1, keepdims=True)))
        
        return(score)
    
    def approx_ELBO(self, wordids, wordcts, gamma):
        return self._approx_ELBO(wordids, wordcts, gamma, self._alpha, self._eta, self._lambda).numpy()

    def _approx_local_logL(self, ids, cts, gammad, alpha, log_thetas, log_beta):
        """
        Estimates E_q[log p(x_d|beta)] with nested MC for a document x_d. 
        
        Arguments:
        ids: List of word tokens that appearing in the document, which corresponds to wordids[d].
        cts: List of word counts for each token in the document, which corresponds to wordcts[d].
        gammad: Tensor of shape [_K] representing the variational parameters of topic ratio for document, which corresponds to gamma[d,:].
        log_thetas: n_MC MC-samples of topic ratio (log) theta of the document, whose shape is [n_MC, _K].
        log_beta: a MC-sample of vocabulary ratio (log) beta, whose shape is shape [_K, _W].
         
        Returns:
        score: MC estimates of the likelihood of the document.
        """
        log_thetas = tf.expand_dims(log_thetas, axis=2) #shape[n_MC, _K,  1]
        log_beta = tf.expand_dims(log_beta, axis=0)     #shape[   1, _K, _W]
        score = 0
        score += tf.reduce_sum(tf_gammaln(gammad) - tf_gammaln(alpha))
        score += tf_gammaln(alpha*self._K) - tf_gammaln(tf.reduce_sum(gammad))

        temp = 0
        temp += tf.linalg.matvec( tf_logsumexp( log_thetas + tf.gather(log_beta, ids, axis=2), axis=1), cts ) #shape: [n_MC]
        temp += tf.linalg.matvec( log_thetas[:,:,0], alpha - gammad ) #shape: [n_MC]
        score += tf_logmeanexp(temp, axis=0)
        
        return(score)

    def _approx_LMELBO(self, wordids, wordcts, gamma, alpha, eta, lambd, n_MC):
        """
        Estimates the lower bound of log p(w) over *all documents* using only
        the documents passed in as "docs." gamma is the set of parameters
        to the variational distribution q(theta) corresponding to the
        set of documents passed in.

        The output of this function is going to be noisy, but can be
        useful for assessing convergence.
        """
        # lower bound of E[log p(x| beta) ]
        batchD = len(wordids)
        score = 0
        
        #shape of betas: [batchD, _K, _W]
        Elogbeta = tf_dirichlet_expectation(lambd)
        log_betas = tf_dirichlet_sample(lambd, n_MC=batchD)
            
        for d in range(0, batchD):
            ids = wordids[d]
            cts = np.array(wordcts[d], dtype=np.float64)
            gammad = gamma[d, :]
            log_thetas = dirichlet_sample(gammad, n_MC=n_MC)
            log_beta = log_betas[d,:,:]
            score += self._approx_local_logL(ids, cts, gammad, alpha, log_thetas, log_beta)
            
        # Compensate for the subsampling of the population of documents
        score = score * self._D / batchD

        # E[log p(beta | eta) - log q (beta | lambda)]
        score = score + tf.reduce_sum((eta - lambd)*Elogbeta)
        score = score + tf.reduce_sum(tf_gammaln(lambd) - tf_gammaln(eta)) 
        score = score + tf.reduce_sum(tf_gammaln(eta*self._V) - tf_gammaln(tf.reduce_sum(lambd, 1, keepdims=True)))
        
        return(score)

    def approx_LMELBO(self, wordids, wordcts, gamma, n_MC):
        return self._approx_LMELBO(wordids, wordcts, gamma, self._alpha, self._eta, self._lambda, n_MC).numpy()
    
    def _approx_dLMELBO(self, wordids, wordcts, gamma, alpha, eta, lambd, level, verbose=True):
        """
        Estimates the lower bound of log p(w) over *all documents* using only
        the documents passed in as "docs." gamma is the set of parameters
        to the variational distribution q(theta) corresponding to the
        set of documents passed in.

        The output of this function is going to be noisy, but can be
        useful for assessing convergence.
        """
        if(level==0):
            if verbose==True:
                print("Using this function at level=0 is not recommended for clarity.")
                print("You are advised to call OnlineLDA_debiased._approx_LMELBO instead.")
            return self._approx_LMELBO(wordids, wordcts, gamma, alpha, eta, lambd, n_MC=1)
        
        # lower bound of E[log p(x| beta) ]
        batchD = len(wordids)
        score = 0
        
        #shape of betas: [batchD, _K, _W]
        log_betas = tf_dirichlet_sample(lambd, n_MC=batchD)
            
        for d in range(batchD):
            ids = wordids[d]
            cts = np.array(wordcts[d], dtype=np_float_type)
            gammad = gamma[d,:]
            log_thetas = dirichlet_sample(gammad, n_MC=2**level)
            log_beta = log_betas[d,:,:]
            score += (
                self._approx_local_logL(ids, cts, gammad, alpha, log_thetas, log_beta)
                - (1/2.) * self._approx_local_logL(ids, cts, gammad, alpha, log_thetas[:2**(level-1) ,:], log_beta)
                - (1/2.) * self._approx_local_logL(ids, cts, gammad, alpha, log_thetas[ 2**(level-1):,:], log_beta)
                     )  
            # The code fails if (1/2) is used instead of (1/2.). 
            # This is because (1/2)==0 in python2. (In python3, (1/2)==0.5)

        # Compensate for the subsampling of the population of documents
        score = score * self._D / batchD
        return(score)
    
    def approx_dLMELBO(self, wordids, wordcts, gamma, level, verbose=True):
        return self._approx_dLMELBO(wordids, wordcts, gamma, self._alpha, self._eta, self._lambda, level, verbose).numpy()
    
    def _approx_mlmc_LMELBO(self, wordids, wordcts, gamma, alpha, eta, lambd, start_level, max_level, batchD_start_level):
        # compute locally marginalized elbo using MLMC
        batchD = len(wordids)
        
        # determine costs per level
        levels = np.arange(start_level, max_level+1)
        ave_n_words = float(sum(map(sum, wordcts))) / batchD
        weights = 2**(-3/2.*levels[1:])
        weights /= sum(weights)
        Ns = np.zeros_like(levels)
        Ns[0] = batchD_start_level
        Ns[1:] = np.array([np.math.ceil(w*(batchD - batchD_start_level)) for w in weights], dtype=np.int)
        Ns[1] = batchD - Ns[0] - sum(Ns[2:])
        cumNs = Ns.cumsum()
            
        score = 0
        for i,l in enumerate(levels):
            temp_wordids = wordids[(cumNs[i] - Ns[i]) : (cumNs[i])]
            temp_wordcts = wordcts[(cumNs[i] - Ns[i]) : (cumNs[i])]
            temp_gamma = gamma[(cumNs[i] - Ns[i]) : (cumNs[i]), :]
            if l==start_level:
                score += self._approx_LMELBO(temp_wordids, temp_wordcts, temp_gamma, alpha, eta, lambd, n_MC=2**l)
            else:
                score += self._approx_dLMELBO(temp_wordids, temp_wordcts, temp_gamma, alpha, eta, lambd, level=l)
        
        return score