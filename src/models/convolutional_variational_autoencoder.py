import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


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

    # original implementation from CVAE tutorial
    # this can be used for sanity check of the code 
    # as E[comp_elbo(x)] = E[comp_iwelbo(x,1)]
    def compute_elbo(self, x):
        mean, logvar = self._encode(x)
        z = self._reparameterize(mean, logvar)
        x_logit = self._decode(z)
        cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
        logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
        logpz = log_normal_pdf(z, 0., 0.)
        logqz_x = log_normal_pdf(z, mean, logvar)
        return tf.reduce_mean(logpx_z + logpz - logqz_x)
    
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

    def compute_iwelbos(self, prob_ratios):
        return tf_reduce_logmeanexp(prob_ratios, axis=1)
    
    def compute_iwelbo(self, x, K):
        prob_ratios = self._compute_prob_ratios(x, K)
        return tf.reduce_mean(self.compute_iwelbos(prob_ratios))
    
    def train_step(self, x, K):
        """Executes one training step and returns the loss.

        This function computes the loss and gradients, and uses the latter to
        update the model's parameters.
        """
        with tf.GradientTape() as tape:
            loss =  - self.compute_iwelbo(x, K)
            gradients = tape.gradient(loss, self.trainable_variables)
            self._optimizer.apply_gradients(zip(gradients, self.trainable_variables))
            
    def generate_images(self, epoch, test_sample):
        mean, logvar = self._encode(test_sample)
        z = self._reparameterize(mean, logvar)
        predictions = self._sample(z)
        fig = plt.figure(figsize=(4, 4))

        for i in range(predictions.shape[0]):
            plt.subplot(4, 4, i + 1)
            plt.imshow(predictions[i, :, :, 0], cmap='gray')
            plt.axis('off')

            
class IWAE_MLMC(IWAE):
    
    def __init__(self, latent_dim):
        super(IWAE_MLMC, self).__init__(latent_dim)
        self._optimizers = {
            'dec': tf.keras.optimizers.Adam(2e-4),
            'enc': tf.keras.optimizers.Adam(2e-4)
        }
        
    def _compute_diwelbos(self, x, L):
        prob_ratios = self._compute_prob_ratios(x, 2**L)
        if L==0:
            return self.compute_iwelbos(prob_ratios)
        elif L>0:
            diwelbos = self.compute_iwelbos(prob_ratios)
            diwelbos -= (1/2.)*self.compute_iwelbos(prob_ratios[:,:2**(L-1) ])
            diwelbos -= (1/2.)*self.compute_iwelbos(prob_ratios[:, 2**(L-1):])
            return diwelbos
        else:
            raise ValueError("Level L must be a non-negative integer.")
            
    def _compute_diwelbo(self, x, L):
        return tf.reduce_mean(self._compute_diwelbos(x, L))
    
    def _compute_iwelbo_mlmc(self, x, max_level=6, w0=1-2.**(-3/2), b=2, randomize=False):
        
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
        iwelbo = 0
        for l in levels:
            if Ns[l]==0:
                continue
            x_tmp = x[offset:offset+Ns[l]]
            
            if randomize==True:
                iwelbo += self._compute_diwelbo(x_tmp, l) * Ns[l] / N / weights[l]   
            elif randomize==False:
                iwelbo += self._compute_diwelbo(x_tmp, l)

            offset += Ns[l]

        return iwelbo

    def compute_iwelbo_mlmc(self, x, max_level, w0=1-2.**(-3/2), b=2):
        return self._compute_iwelbo_mlmc(x, max_level, w0, b, randomize=False)
    
    def compute_iwelbo_rmlmc(self, x, max_level, w0=1-2.**(-3/2), b=2):
        return self._compute_iwelbo_mlmc(x, max_level, w0, b, randomize=True)

    def train_step(self, x, K=8, loss='iwelbo', max_level=3, w0=1-2.**(-3/2), b=2):
        
        if loss=='elbo':
            with tf.GradientTape() as tape:
                loss =  - self.compute_iwelbo(x, 1)
                gradients = tape.gradient(loss, self.trainable_variables)
                self._optimizer.apply_gradients(zip(gradients, self.trainable_variables))
            return 
        
        # train encoder
        with tf.GradientTape() as tape:
            _loss = - self.compute_elbo(x)
            gradients = tape.gradient(_loss, self._encoder.trainable_variables)
            self._optimizers['enc'].apply_gradients(zip(gradients, self._encoder.trainable_variables))
        
        # train decoder
        with tf.GradientTape() as tape:
            if loss=='iwelbo_mlmc':
                _loss = - self.compute_iwelbo_mlmc(x, max_level, w0, b)
            elif loss=='iwelbo_rmlmc':
                _loss = - self.compute_iwelbo_rmlmc(x, max_level, w0, b)
            elif loss == 'iwelbo':
                _loss = - self.compute_iwelbo(x, K)
            else:
                raise ValueError("Argument 'loss' must be one of elbo, iwelbo, iwelbo_mlmc or iwelbo_rmlmc.")
            gradients = tape.gradient(_loss, self._decoder.trainable_variables)
            self._optimizers['dec'].apply_gradients(zip(gradients, self._decoder.trainable_variables))
