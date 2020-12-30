import keras
import keras.backend as K
from keras.models import load_model
from keras.datasets import mnist
import numpy as np
from PIL import Image


def encoder_loss(self):
    intermediate_layer_model = keras.Model(inputs=self.discriminator.input,
                                           outputs=self.discriminator.get_layer("feature_extractor").output)

    def loss(y_true, y_pred):
        l1 = K.mean(K.square(y_pred - y_true))
        l2 = K.mean(K.square(intermediate_layer_model(self.generator(y_pred)) - intermediate_layer_model(y_true)))
        return l1 + l2

    return loss


def wasserstein_loss(y_true, y_pred):
    return -K.mean(y_true * y_pred)


def dataset():

    """
    Load dataset, convert to 32x32, constrain input to [-1, 1].
    """

    (x_train, _), (x_test, _) = mnist.load_data()

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


def get_batch(batch_size):
    idx = np.random.choice(np.shape(x_te)[0], batch_size, replace=False)
    return x_te[idx]


if __name__ == '__main__':

    encoder = load_model('encoder.h5', custom_objects={'encoder_loss': encoder_loss})
    generator = load_model('gen.h5', custom_objects={'wasserstein_loss': wasserstein_loss})
    im = get_batch(1)
    im2 = generator.predict(encoder.predict(im))
    im.save("image_real.jpg")
    im2.save("image_regen.jpg")



