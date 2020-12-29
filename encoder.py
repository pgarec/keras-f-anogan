from keras.layers import Conv2D, Conv2DTranspose, Flatten, Dense, LeakyReLU, BatchNormalization, Reshape
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from keras.models import Sequential
from keras.metrics import mean_squared_error
from keras.optimizers import Adam
import keras
from keras.initializers import RandomNormal
import keras.backend as K
from keras.models import load_model
import numpy as np


def wasserstein_loss(y_true, y_pred):
    return -K.mean(y_true * y_pred)


class Encoder:
    def __init__(self, image_shape=(32, 32, 1), n_filters=64, z_size=(1, 1, 100),
                 alpha=0.2, lr=5e-5, extra_layers=0, clamp_lower=-0.01,
                 clamp_upper=0.01, disc_iters_per_gen_iters=5):

        assert image_shape[0] % 8 == 0, "Image shape must be divisible by 8."

        self.discriminator = load_model('disc.h5', custom_objects={'wasserstein_loss': wasserstein_loss})
        self.generator = load_model('gen.h5', custom_objects={'wasserstein_loss': wasserstein_loss})
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

    def encoder(self):
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
                    name="feature_extractor",
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
        model.add(Dense(100, use_bias=False))
        model.add(Reshape((1,1,100)))

        return model

    def encoder_loss(self):
        intermediate_layer_model = keras.Model(inputs=self.discriminator.input,
                                               outputs=self.discriminator.get_layer("feature_extractor").output)
        print(intermediate_layer_model.summary())
        print("---")
        print(self.generator.summary())

        def loss(y_true, y_pred):
            l1 = K.mean(K.square(y_pred - y_true))
            print("True")
            print(y_true)
            print("Predicted")
            print(y_pred)
            x = intermediate_layer_model(y_true)
            #l2 = K.mean(K.square(intermediate_layer_model(y_pred) - intermediate_layer_model(y_true)))
            return l1
        return loss