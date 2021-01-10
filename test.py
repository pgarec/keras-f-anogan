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
import statistics as st


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
    l2 = np.mean(np.square(intermediate_layer_model.predict(np.expand_dims(y_pred, axis=0))
                           - intermediate_layer_model.predict(np.expand_dims(y_true, axis=0))))
    return l2


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
def dataset_1():
    """
    Load dataset, convert to 32x32, constrain input to [-1, 1].
    """

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    train_filter = np.where((y_train == 1))
    test_filter = np.where((y_test == 1))
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
def dataset_2():
    """
    Load dataset, convert to 32x32, constrain input to [-1, 1].
    """

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    train_filter = np.where((y_train == 2))
    test_filter = np.where((y_test == 2))
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
def dataset_3():
    """
    Load dataset, convert to 32x32, constrain input to [-1, 1].
    """

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    train_filter = np.where((y_train == 3))
    test_filter = np.where((y_test == 3))
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
def dataset_4():
    """
    Load dataset, convert to 32x32, constrain input to [-1, 1].
    """

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    train_filter = np.where((y_train == 4))
    test_filter = np.where((y_test == 4))
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
def dataset_5():
    """
    Load dataset, convert to 32x32, constrain input to [-1, 1].
    """

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    train_filter = np.where((y_train == 5))
    test_filter = np.where((y_test == 5))
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
def dataset_6():
    """
    Load dataset, convert to 32x32, constrain input to [-1, 1].
    """

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    train_filter = np.where((y_train == 6))
    test_filter = np.where((y_test == 6))
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
def dataset_7():
    """
    Load dataset, convert to 32x32, constrain input to [-1, 1].
    """

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    train_filter = np.where((y_train == 7))
    test_filter = np.where((y_test == 7))
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
def dataset_8():
    """
    Load dataset, convert to 32x32, constrain input to [-1, 1].
    """

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    train_filter = np.where((y_train == 8))
    test_filter = np.where((y_test == 8))
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
def dataset_9():
    """
    Load dataset, convert to 32x32, constrain input to [-1, 1].
    """

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    train_filter = np.where((y_train == 9))
    test_filter = np.where((y_test == 9))
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
x1_tr, x1_te = dataset_1()
x2_tr, x2_te = dataset_2()
x3_tr, x3_te = dataset_3()
x4_tr, x4_te = dataset_4()
x5_tr, x5_te = dataset_5()
x6_tr, x6_te = dataset_6()
x7_tr, x7_te = dataset_7()
x8_tr, x8_te = dataset_8()
x9_tr, x9_te = dataset_9()

def get_batch(batch_size):
    idx = np.random.choice(np.shape(x_te)[0], batch_size, replace=False)
    return x_te[idx]

def get_batch_0(batch_size):
    idx = np.random.choice(np.shape(x0_te)[0], batch_size, replace=False)
    return x0_te[idx]

def get_batch_1(batch_size):
    idx = np.random.choice(np.shape(x1_te)[0], batch_size, replace=False)
    return x1_te[idx]
def get_batch_2(batch_size):
    idx = np.random.choice(np.shape(x2_te)[0], batch_size, replace=False)
    return x2_te[idx]
def get_batch_3(batch_size):
    idx = np.random.choice(np.shape(x3_te)[0], batch_size, replace=False)
    return x3_te[idx]
def get_batch_4(batch_size):
    idx = np.random.choice(np.shape(x4_te)[0], batch_size, replace=False)
    return x4_te[idx]
def get_batch_5(batch_size):
    idx = np.random.choice(np.shape(x5_te)[0], batch_size, replace=False)
    return x5_te[idx]
def get_batch_6(batch_size):
    idx = np.random.choice(np.shape(x6_te)[0], batch_size, replace=False)
    return x6_te[idx]
def get_batch_7(batch_size):
    idx = np.random.choice(np.shape(x7_te)[0], batch_size, replace=False)
    return x7_te[idx]
def get_batch_8(batch_size):
    idx = np.random.choice(np.shape(x8_te)[0], batch_size, replace=False)
    return x8_te[idx]
def get_batch_9(batch_size):
    idx = np.random.choice(np.shape(x9_te)[0], batch_size, replace=False)
    return x9_te[idx]

def make_noise(batch_size):
    noise = np.random.normal(scale=0.5, size=(tuple([batch_size]) + z_size))
    return noise


def random_noise(batch_size):
    noise = np.random.normal(scale=0.5, size=(tuple([batch_size]) + (32,32,1)))
    return noise


if __name__ == '__main__':

    real0 = get_batch_0(1)
    fake1 = get_batch_1(1)
    fake2 = get_batch_2(1)
    fake3 = get_batch_3(1)
    fake4 = get_batch_4(1)
    fake5 = get_batch_5(1)
    fake6 = get_batch_6(1)
    fake7 = get_batch_7(1)
    fake8 = get_batch_8(1)
    fake9 = get_batch_9(1)

    real_regen0 = encodergen.predict_on_batch(real0)
    fake_regen1 = encodergen.predict_on_batch(fake1)
    fake_regen2 = encodergen.predict_on_batch(fake2)
    fake_regen3 = encodergen.predict_on_batch(fake3)
    fake_regen4 = encodergen.predict_on_batch(fake4)
    fake_regen5 = encodergen.predict_on_batch(fake5)
    fake_regen6 = encodergen.predict_on_batch(fake6)
    fake_regen7 = encodergen.predict_on_batch(fake7)
    fake_regen8 = encodergen.predict_on_batch(fake8)
    fake_regen9 = encodergen.predict_on_batch(fake9)

    width = 5
    height = 5
    rows = 10
    cols = 3
    axes = []
    fig = plt.figure()


    axes.append(fig.add_subplot(rows, cols, 0*rows + 1))
    plt.imshow(real0)
    axes.append(fig.add_subplot(rows, cols, 0 * rows + 2))
    plt.imshow(real_regen0)
    axes.append(fig.add_subplot(rows, cols, 0 * rows + 3))
    plt.imshow(real0-real_regen0)

    axes.append(fig.add_subplot(rows, cols, 1 * rows + 1))
    plt.imshow(fake1)
    axes.append(fig.add_subplot(rows, cols, 1 * rows + 2))
    plt.imshow(fake_regen1)
    axes.append(fig.add_subplot(rows, cols, 1 * rows + 3))
    plt.imshow(fake1 - fake_regen1)

    axes.append(fig.add_subplot(rows, cols, 2 * rows + 1))
    plt.imshow(fake2)
    axes.append(fig.add_subplot(rows, cols, 2 * rows + 2))
    plt.imshow(fake_regen2)
    axes.append(fig.add_subplot(rows, cols, 2 * rows + 3))
    plt.imshow(fake2 - fake_regen2)

    axes.append(fig.add_subplot(rows, cols, 3 * rows + 1))
    plt.imshow(fake3)
    axes.append(fig.add_subplot(rows, cols, 3 * rows + 2))
    plt.imshow(fake_regen3)
    axes.append(fig.add_subplot(rows, cols, 3 * rows + 3))
    plt.imshow(fake3 - fake_regen3)

    axes.append(fig.add_subplot(rows, cols, 4 * rows + 1))
    plt.imshow(fake4)
    axes.append(fig.add_subplot(rows, cols, 4 * rows + 2))
    plt.imshow(fake_regen4)
    axes.append(fig.add_subplot(rows, cols, 4 * rows + 3))
    plt.imshow(fake4 - fake_regen4)

    axes.append(fig.add_subplot(rows, cols, 5 * rows + 1))
    plt.imshow(fake5)
    axes.append(fig.add_subplot(rows, cols, 5 * rows + 2))
    plt.imshow(fake_regen5)
    axes.append(fig.add_subplot(rows, cols, 5 * rows + 3))
    plt.imshow(fake5 - fake_regen5)

    axes.append(fig.add_subplot(rows, cols, 6 * rows + 1))
    plt.imshow(fake6)
    axes.append(fig.add_subplot(rows, cols, 6 * rows + 2))
    plt.imshow(fake_regen6)
    axes.append(fig.add_subplot(rows, cols, 6 * rows + 3))
    plt.imshow(fake6 - fake_regen6)

    axes.append(fig.add_subplot(rows, cols, 7 * rows + 1))
    plt.imshow(fake7)
    axes.append(fig.add_subplot(rows, cols, 7 * rows + 2))
    plt.imshow(fake_regen7)
    axes.append(fig.add_subplot(rows, cols, 7 * rows + 3))
    plt.imshow(fake7 - fake_regen7)

    axes.append(fig.add_subplot(rows, cols, 8 * rows + 1))
    plt.imshow(fake8)
    axes.append(fig.add_subplot(rows, cols, 8 * rows + 2))
    plt.imshow(fake_regen8)
    axes.append(fig.add_subplot(rows, cols, 8 * rows + 3))
    plt.imshow(fake8 - fake_regen8)

    axes.append(fig.add_subplot(rows, cols, 9 * rows + 1))
    subplot_title = ("Digit")
    axes[-1].set_title(subplot_title)
    plt.imshow(fake9)
    axes.append(fig.add_subplot(rows, cols, 9 * rows + 2))
    subplot_title = ("Reconstruction")
    axes[-1].set_title(subplot_title)
    plt.imshow(fake_regen9)
    axes.append(fig.add_subplot(rows, cols, 9 * rows + 3))
    subplot_title = ("Difference")
    axes[-1].set_title(subplot_title)
    plt.imshow(fake9 - fake_regen9)

    fig.savefig('recons.png')












































    fig.tight_layout()
    plt.save('reconstruction.png')








