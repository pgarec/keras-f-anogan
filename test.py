import keras
import keras.backend as K
from keras.models import load_model
from keras.datasets import mnist
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def encoder_loss():
    intermediate_layer_model = keras.Model(inputs=discriminator.input,
                                           outputs=discriminator.get_layer("feature_extractor").output)

    def loss(y_true, y_pred):
        l1 = K.mean(K.square(y_pred - y_true))
        l2 = K.mean(K.square(intermediate_layer_model(y_pred) - intermediate_layer_model(y_true)))
        return l1 + l2

    return loss


def custom_activation(x):
    return K.tanh(x)/2


def wasserstein_loss(y_true, y_pred):
    return -K.mean(y_true * y_pred)

z_size=(1, 1, 100)
discriminator = load_model('disc.h5', custom_objects={'wasserstein_loss': wasserstein_loss})
disc_gp = load_model('disc-gp.h5')
gen_gp = load_model('gen-gp.h5')
encoder = load_model('encoder.h5', custom_objects={'loss':encoder_loss(), 'custom_activation':custom_activation})
generator = load_model('gen.h5', custom_objects={'wasserstein_loss': wasserstein_loss})
encodergen = load_model('encodergen.h5', custom_objects={'loss':encoder_loss(), 'custom_activation':custom_activation})

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


def make_noise(batch_size):
    noise = np.random.normal(scale=0.5, size=(tuple([batch_size]) + z_size))
    return noise


if __name__ == '__main__':


    for i in range(15):
        print(i)
        n = make_noise(1)
        #im = get_batch(1)
        #im2 = encodergen.predict(im)
        im3 = gen_gp.predict(n)

        #plt.imshow(im.squeeze(), cmap='gray')
        #plt.savefig('resultats_encoding/image_real'+str(i)+'.png')

        #plt.imshow(im2.squeeze(), cmap='gray')
        #plt.savefig('resultats_encoding/image_reconstruced' + str(i) + '.png')

        plt.imshow(im3.squeeze(), cmap='gray')
        plt.savefig('proves-wgangp/generated' + str(i) + '.png')




