import numpy as np
import tensorflow as tf

#utility funcs
float_type = np.float32

tf_int32 = lambda x:tf.cast(x, tf.int32)
tf_float = lambda x:tf.cast(x, float_type)

def tf_reduce_logmeanexp(ary, axis=1):
    n_MC = tf.shape(ary)[axis]
    return tf.reduce_logsumexp(ary, axis=axis) - tf.math.log(tf_float(n_MC))

def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = tf.math.log(2. * np.pi)
    return tf.reduce_sum(
        -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
        axis=raxis)


class IWAE(tf.keras.Model):

    def __init__(self, _latent_dim):
        super(IWAE, self).__init__()
        self._latent_dim = _latent_dim
        self._encoder = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(28, 28, 1)),
            tf.keras.layers.Conv2D(
                filters=32, kernel_size=3, strides=(2, 2), activation='relu', kernel_regularizer='l2'),
            tf.keras.layers.Conv2D(
                filters=64, kernel_size=3, strides=(2, 2), activation='relu'),
            tf.keras.layers.Flatten(),
            # No activation
            tf.keras.layers.Dense(_latent_dim + _latent_dim)
        ])

        self._decoder = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(_latent_dim,)),
            tf.keras.layers.Dense(units=7*7*32, activation=tf.nn.relu),
            tf.keras.layers.Reshape(target_shape=(7, 7, 32)),
            tf.keras.layers.Conv2DTranspose(
                filters=64, kernel_size=3, strides=2, padding='same',
                activation='relu'),
            tf.keras.layers.Conv2DTranspose(
                filters=32, kernel_size=3, strides=2, padding='same',
                activation='relu'),
            # No activation
            tf.keras.layers.Conv2DTranspose(
                filters=1, kernel_size=3, strides=1, padding='same'),
        ])
        
        self._optimizer = tf.keras.optimizers.Adam(2e-4)

    def _sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(100, self._latent_dim))
        return self._decode(eps, apply_sigmoid=True)

    def _encode(self, x):
        mean, logvar = tf.split(self._encoder(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    def _reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean

    def _decode(self, z, apply_sigmoid=False):
        logits = self._decoder(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits

    def compute_elbo(self, x):
        # original implementation from CVAE tutorial
        # this can be used for sanity check of the code 
        # as E[comp_elbo(x)] = E[comp_iwelbo(x,1)]
        mean, logvar = self._encode(x)
        z = self._reparameterize(mean, logvar)
        x_logit = self._decode(z)
        cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
        logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
        logpz = log_normal_pdf(z, 0., 0.)
        logqz_x = log_normal_pdf(z, mean, logvar)
        return tf.reduce_mean(logpx_z + logpz - logqz_x)
    
    def train_step(self, x):
        """Executes one training step and returns the loss.

        This function computes the loss and gradients, and uses the latter to
        update the model's parameters.
        """
        with tf.GradientTape() as tape:
            loss =  - self.compute_elbo(x)
            gradients = tape.gradient(loss, self.trainable_variables)
            self._optimizer.apply_gradients(zip(gradients, self.trainable_variables))
            

class IWAE_MLMC(IWAE):
    
    def __init__(self, latent_dim):
        super(IWAE_MLMC, self).__init__(latent_dim)
        self.__init_optimizers()
        
    def __init_optimizers(self):
        self._optimizers = {
            'dec': tf.keras.optimizers.Adam(2e-4),
            'enc': tf.keras.optimizers.Adam(2e-4)
        }

    def _compute_prob_ratios(self, x, K):
        mean, logvar = self._encode(x)
        # repeat K Monte Carlo samples
        x_rep = tf.repeat(x, K, axis=0)# [x1,x2,...] -> [x1, x1, ...,x1, x2, x2, ...]
        mean_rep = tf.repeat(mean, repeats=K, axis=0)
        logvar_rep = tf.repeat(logvar, repeats=K, axis=0)
        z = self._reparameterize(mean_rep, logvar_rep)
        x_logit = self._decode(z)
        cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x_rep)
        logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
        logpz = log_normal_pdf(z, 0., 0.)
        logqz_x = log_normal_pdf(z, mean_rep, logvar_rep)
        prob_ratios = tf.reshape(logpx_z + logpz - logqz_x, shape=[-1,K])
        return prob_ratios
    
    def _compute_objectives(self, prob_ratios, obj):
        if obj=='iwelbo': 
            return tf_reduce_logmeanexp(prob_ratios, axis=1)
        # log p + D[q||p]
        elif obj=='pearson_ubo':
            return 0.5*tf_reduce_logmeanexp(2.*prob_ratios, axis=1)
        elif obj=='hellinger_lbo':
            return 2.*tf_reduce_logmeanexp(0.5*prob_ratios, axis=1)
        elif obj=='neyman_lbo':
            return -1.*tf_reduce_logmeanexp(-prob_ratios, axis=1)
        # log p - D[p||q]
        elif obj=='pearson_lbo':
            return  -0.5*tf_reduce_logmeanexp(-prob_ratios, axis=1)\
                    +0.5*tf_reduce_logmeanexp(prob_ratios, axis=1)
        elif obj=='hellinger_ubo':
            return -2*tf_reduce_logmeanexp(0.5*prob_ratios, axis=1)\
                   +2*tf_reduce_logmeanexp(prob_ratios, axis=1)
        elif obj=='neyman_ubo':
            return  tf_reduce_logmeanexp(2.*prob_ratios, axis=1)\
                    -tf_reduce_logmeanexp(prob_ratios, axis=1)
        else: 
            print(obj)
            raise ValueError("given input 'obj' is invalid.")

    def compute_objective(self, x, K, obj='iwelbo'):
        prob_ratios = self._compute_prob_ratios(x, K)
        objectives = self._compute_objectives(prob_ratios, obj)
        return tf.reduce_mean(objectives)
    
    def _compute_dobjective(self, x, K, obj):
        prob_ratios = self._compute_prob_ratios(x, K)
        if K==1:
                diff = self._compute_objectives(prob_ratios, obj)
        elif K>1:
            assert(K%2==0)
            diff = self._compute_objectives(prob_ratios, obj)
            diff -= (1/2.)*self._compute_objectives(prob_ratios[:,:K//2 ], obj)
            diff -= (1/2.)*self._compute_objectives(prob_ratios[:, K//2:], obj)
        else:
            raise ValueError("Level K must be evenly-divisible positive integer.")
        return tf.reduce_mean(diff)
    
    def compute_objective_mlmc(self, x, max_level=6, w0=1-2.**(-3/2), b=2, randomize=False, obj='iwelbo'):
        N = x.shape[0]

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
            raise(ValueError("Invarid argument for 'randomize' of function LMELBO_MLMC. It must be True or False."))

        # compute dLMELBO's using disjoint samples at each level and sum them up
        offset = 0
        out = 0
        for l in levels:
            if Ns[l]==0:
                continue
            x_tmp = x[offset:offset+Ns[l]]
            if randomize==True:
                out += self._compute_dobjective(x_tmp, 2**l, obj) * Ns[l] / N / weights[l]   
            elif randomize==False:
                out += self._compute_dobjective(x_tmp, 2**l, obj)
            offset += Ns[l]
        return out

    def train_step(self, x, K=8, max_level=3, w0=1-2.**(-3/2), b=2, obj='elbo'):
        
        if obj=='elbo':
            super(IWAE_MLMC, self).train_step(x)
            return 
        
        # train encoder
        with tf.GradientTape() as tape:
            loss = - self.compute_elbo(x)
            gradients = tape.gradient(loss, self._encoder.trainable_variables)
            self._optimizers['enc'].apply_gradients(zip(gradients, self._encoder.trainable_variables))
        
        # train decoder
        with tf.GradientTape() as tape:
            if obj=='iwelbo_mlmc':
                loss = - self.compute_objective_mlmc(x, max_level, w0, b, randomize=False, obj='iwelbo')
            elif obj=='iwelbo_rmlmc':
                loss = - self.compute_objective_mlmc(x, max_level, w0, b, randomize=True, obj='iwelbo')
            elif obj == 'iwelbo':
                loss = - self.compute_objective(x, K, obj)
            else:
                raise ValueError("Argument 'loss' must be one of elbo, iwelbo, iwelbo_mlmc or iwelbo_rmlmc.")
            gradients = tape.gradient(loss, self._decoder.trainable_variables)
            self._optimizers['dec'].apply_gradients(zip(gradients, self._decoder.trainable_variables))
