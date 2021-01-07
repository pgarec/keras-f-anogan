from keras.layers import Conv2D, Conv2DTranspose, Flatten, Dense, LeakyReLU, BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from keras.models import Sequential
from keras.metrics import mean_squared_error
from keras.optimizers import Adam
from keras.initializers import RandomNormal
import keras.backend as K
import numpy as np

class DCGAN:
    def __init__(self, image_shape=(32, 32, 1), n_filters=64, z_size=(1, 1, 100),
                 alpha=0.2, lr=5e-5, extra_layers=0, clamp_lower=-0.01,
                 clamp_upper=0.01, disc_iters_per_gen_iters=5):

        assert image_shape[0] % 8 == 0, "Image shape must be divisible by 8."

        self.image_shape = image_shape
        self.n_filters = n_filters
        self.z_size = z_size
        self.alpha = alpha
        self.lr = lr
        self.extra_layers = extra_layers
        self.clamp_lower = clamp_lower
        self.clamp_upper = clamp_upper
        self.disc_iters_per_gen_iters = disc_iters_per_gen_iters
        self.weight_init = RandomNormal(mean=0., stddev=0.02)

    def discriminator(self):
        model = Sequential()
        model.add(Conv2D(filters=self.n_filters,
                    kernel_size=(4, 4),
                    strides=2,
                    padding='same',
                    use_bias=False,
                    input_shape=self.image_shape,
                    kernel_initializer=self.weight_init))
        model.add(LeakyReLU(self.alpha))
        model.add(BatchNormalization())

        for n in range(self.extra_layers):
            model.add(Conv2D(filters=self.n_filters,
                    kernel_size=(3, 3),
                    strides=1,
                    padding='same',
                    use_bias=False,
                    kernel_initializer=self.weight_init))
            model.add(LeakyReLU(self.alpha))
            model.add(BatchNormalization())

        model.add(Conv2D(filters=2*self.n_filters,
                    kernel_size=(4, 4),
                    name="feature_extractor",
                    strides=2,
                    padding='same',
                    use_bias=False,
                    kernel_initializer=self.weight_init))
        model.add(LeakyReLU(self.alpha))
        model.add(BatchNormalization())

        model.add(Conv2D(filters=4*self.n_filters,
                    kernel_size=(4, 4),
                    strides=2,
                    padding='same',
                    use_bias=False,
                    kernel_initializer=self.weight_init))
        model.add(LeakyReLU(self.alpha))
        model.add(BatchNormalization())

        model.add(Flatten())
        print(model.summary())
        model.add(Dense(1, use_bias=False))

        return model

    def generator(self):
        model = Sequential()
        model.add(Conv2DTranspose(filters=2*self.n_filters,
                    kernel_size=(4, 4),
                    strides=1,
                    padding='same',
                    activation='relu',
                    use_bias=False,
                    kernel_initializer=self.weight_init))
        model.add(BatchNormalization())

        pixels = 1
        while pixels < self.image_shape[0]//2:

            model.add(Conv2DTranspose(filters=self.n_filters,
                        kernel_size=(4, 4),
                        strides=2,
                        padding='same',
                        activation='relu',
                        use_bias=False,
                        kernel_initializer=self.weight_init))
            model.add(BatchNormalization())

            pixels *= 2

        model.add(Conv2DTranspose(filters=1,
                    kernel_size=(4, 4),
                    strides=2,
                    padding='same',
                    activation='tanh',
                    use_bias=False,
                    kernel_initializer=self.weight_init))

        return model

    def adversarial(self, generator, discriminator):
        model = Sequential()
        model.add(generator)
        model.add(discriminator)

        return model

    def gradient_penalty_loss(y_true, y_pred, averaged_samples, gradient_penalty_weight):
        """Calculates the gradient penalty loss for a batch of "averaged" samples.
        In Improved WGANs, the 1-Lipschitz constraint is enforced by adding a term to the
        loss function that penalizes the network if the gradient norm moves away from 1.
        However, it is impossible to evaluate this function at all points in the input
        space. The compromise used in the paper is to choose random points on the lines
        between real and generated samples, and check the gradients at these points. Note
        that it is the gradient w.r.t. the input averaged samples, not the weights of the
        discriminator, that we're penalizing!
        In order to evaluate the gradients, we must first run samples through the generator
        and evaluate the loss. Then we get the gradients of the discriminator w.r.t. the
        input averaged samples. The l2 norm and penalty can then be calculated for this
        gradient.
        Note that this loss function requires the original averaged samples as input, but
        Keras only supports passing y_true and y_pred to loss functions. To get around this,
        we make a partial() of the function with the averaged_samples argument, and use that
        for model training."""
        # first get the gradients:
        #   assuming: - that y_pred has dimensions (batch_size, 1)
        #             - averaged_samples has dimensions (batch_size, nbr_features)
        # gradients afterwards has dimension (batch_size, nbr_features), basically
        # a list of nbr_features-dimensional gradient vectors
        gradients = K.gradients(y_pred, averaged_samples)[0]
        # compute the euclidean norm by squaring ...
        gradients_sqr = K.square(gradients)
        #   ... summing over the rows ...
        gradients_sqr_sum = K.sum(gradients_sqr,
                                  axis=np.arange(1, len(gradients_sqr.shape)))
        #   ... and sqrt
        gradient_l2_norm = K.sqrt(gradients_sqr_sum)
        # compute lambda * (1 - ||grad||)^2 still for each single sample
        gradient_penalty = gradient_penalty_weight * K.square(1 - gradient_l2_norm)
        # return the mean as loss over all the batch samples
        return K.mean(gradient_penalty)

    def wasserstein_loss(self, y_true, y_pred):
        return -K.mean(y_true * y_pred)