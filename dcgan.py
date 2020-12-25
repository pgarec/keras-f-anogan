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
                    input_shape=self.z_size,
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

    def wasserstein_loss(self, y_true, y_pred):
        return -K.mean(y_true * y_pred)
