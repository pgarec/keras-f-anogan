import keras
import keras.backend as K
from keras.models import load_model
from keras.datasets import mnist
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from keras.layers.merge import _Merge
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import math



class RandomWeightedAverage(_Merge):
    """Takes a randomly-weighted average of two tensors. In geometric terms, this
    outputs a random point on the line between each pair of input points.
    Inheriting from _Merge is a little messy but it was the quickest solution I could
    think of. Improvements appreciated."""

    def _merge_function(self, inputs):
        alpha = K.random_uniform((32, 1, 1, 1))
        return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])


def encoder_loss():
    intermediate_layer_model = keras.Model(inputs=discriminator.input,
                                           outputs=discriminator.get_layer("feature_extractor").output)

    def loss(y_true, y_pred):
        l1 = K.mean(K.square(y_pred - y_true))
        l2 = K.mean(K.square(intermediate_layer_model(y_pred) - intermediate_layer_model(y_true)))
        return l1 + l2

    return loss

def encoder_loss2(y_true,y_pred):
    intermediate_layer_model = keras.Model(inputs=discriminator.input,
                                           outputs=discriminator.get_layer("feature_extractor").output)

    l1 = np.mean(np.square(y_pred - y_true))
    #l2 = np.mean(K.square(intermediate_layer_model.predict(y_pred) - intermediate_layer_model.predict(y_true)))
    return l1


def custom_activation(x):
    return K.tanh(x)/2

def wasserstein_loss2(y_true, y_pred):
    return K.mean(y_true * y_pred)

def wasserstein_loss(y_true, y_pred):
    return -K.mean(y_true * y_pred)

z_size=(1, 1, 100)
z_size2=100

discriminator = load_model('disc.h5', custom_objects={'wasserstein_loss': wasserstein_loss})
encoder = load_model('encoder.h5', custom_objects={'loss':encoder_loss(), 'custom_activation':custom_activation})
generator = load_model('gen.h5', custom_objects={'wasserstein_loss': wasserstein_loss})
encodergen = load_model('encodergen.h5', custom_objects={'loss':encoder_loss(), 'custom_activation':custom_activation})


def dataset_0():
    """
    Load dataset, convert to 32x32, constrain input to [-1, 1].
    """

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    train_filter = np.where((y_train == 0))
    test_filter = np.where((y_test == 0))
    (x_train, y_train) = x_train[train_filter], y_train[train_filter]
    (x_test, y_test) = x_test[test_filter], y_test[test_filter]

    x_train = np.reshape(x_train, (-1, 28, 28, 1))
    train_padded = np.zeros((np.shape(x_train)[0], 32, 32, 1))
    train_padded[:, 2:30, 2:30, :] = x_train
    train_padded /= np.max(train_padded)
    train_padded *= 2
    train_padded -= 1

    x_test = np.reshape(x_test, (-1, 28, 28, 1))
    test_padded = np.zeros((np.shape(x_test)[0], 32, 32, 1))
    test_padded[:, 2:30, 2:30, :] = x_test
    test_padded /= np.max(test_padded)
    test_padded *= 2
    test_padded -= 1

    return train_padded, test_padded

def dataset():

    """
    Load dataset, convert to 32x32, constrain input to [-1, 1].
    """

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    train_filter = np.where((y_train != 0))
    test_filter = np.where((y_test != 0))
    (x_train, y_train) = x_train[train_filter], y_train[train_filter]
    (x_test, y_test) = x_test[test_filter], y_test[test_filter]

    x_train = np.reshape(x_train, (-1, 28, 28, 1))
    train_padded = np.zeros((np.shape(x_train)[0], 32, 32, 1))
    train_padded[:, 2:30, 2:30, :] = x_train
    train_padded /= np.max(train_padded)
    train_padded *= 2
    train_padded -= 1

    x_test = np.reshape(x_test, (-1, 28, 28, 1))
    test_padded = np.zeros((np.shape(x_test)[0], 32, 32, 1))
    test_padded[:, 2:30, 2:30, :] = x_test
    test_padded /= np.max(test_padded)
    test_padded *= 2
    test_padded -= 1

    return train_padded, test_padded


x_tr, x_te = dataset()
x0_tr, x0_te = dataset_0()

def get_batch(batch_size):
    idx = np.random.choice(np.shape(x_te)[0], batch_size, replace=False)
    return x_te[idx]

def get_batch_0(batch_size):
    idx = np.random.choice(np.shape(x0_te)[0], batch_size, replace=False)
    return x0_te[idx]

def make_noise(batch_size):
    noise = np.random.normal(scale=0.5, size=(tuple([batch_size]) + z_size))
    return noise


if __name__ == '__main__':

    fake = get_batch(200)
    real = get_batch_0(200)

    fake_regen = encodergen.predict_on_batch(fake)
    real_regen = encodergen.predict_on_batch(real)

    loss_real = []
    loss_fake = []

    for i in range(200):
        loss_fake.append(encoder_loss2(fake[i],fake_regen[i]))
        loss_real.append(encoder_loss2(real[i], real_regen[i]))

    print(loss_fake)
    print(np.mean(loss_fake))
    print(np.mean(loss_real))









