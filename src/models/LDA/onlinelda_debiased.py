import tensorflow as tf
import tensorflow_probability as tfp

from . import onlinelda as olda

def tf_logmeanexp(ary, axis=1):
    # for numerical stability
    return(logsumexp(ary, axis=axis) - np.log(ary.shape[axis]))


class OnlineLDA_deviased(olda.OnlineLDA):
    def approx_local_logL(self, ids, cts, gammad, log_thetas, log_beta):
        """
        Estimates the E_q[log p(x|beta)] with nested Monte Carlo 
        
        inputs:
        ids: wordids[d]
        cts: wordcts[d]
        gammad: gamma[d,:], shape[_K]
        log_thetas: shape[n_MC, _K]
        log_beta: shape[_K, _W]
        """
        log_thetas = np.expand_dims(log_thetas, axis=2) #shape[n_MC, _K,  1]
        log_beta = np.expand_dims(log_beta, axis=0)     #shape[   1, _K, _W]
        score = 0
        score += np.sum(gammaln(gammad) - gammaln(self._alpha))
        score += gammaln(self._alpha*self._K) - gammaln(np.sum(gammad))

        temp = 0
        temp += np.dot( logsumexp( log_thetas + log_beta[:,:,ids], axis=1), cts ) #shape: [n_MC]
        temp += np.dot( log_thetas.squeeze(axis=2), self._alpha - gammad ) #shape: [n_MC]
        score += logmeanexp(temp, axis=0)  #* self._lendoc / sum(cts)
        
        return(score)
