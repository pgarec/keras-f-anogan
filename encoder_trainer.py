import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from encoder import Encoder
from dcgan import DCGAN
from keras.datasets import mnist
from keras.models import load_model
from keras.optimizers import Adam, RMSprop
import keras.backend as K
import keras
plt.switch_backend('agg')


def wasserstein_loss(y_true, y_pred):
    return -K.mean(y_true * y_pred)


def dataset():

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


class Trainer:

    def __init__(self, encod, optimizer='adam', plot_path='plots'):
        assert optimizer.lower() in ['adam', 'rmsprop'], "Optimizer unrecognized or unavailable."

        if not os.path.isdir(plot_path):
            os.mkdir(plot_path)

        self.generator = load_model('gen.h5',  custom_objects={'wasserstein_loss': wasserstein_loss})
        self.discriminator = load_model('disc.h5', custom_objects={'wasserstein_loss': wasserstein_loss})

        self.encoder_class = encod
        self.encod = encod.encoder()
        self.z_size = encod.z_size
        self.lr = encod.lr
        self.x_train, self.x_test = dataset()
        self.model_compiler(optimizer)
        self.plot_path = plot_path

    def model_compiler(self, optimizer):
        if optimizer.lower() == 'adam':
            opt = Adam(lr=self.lr, beta_1=0.5, beta_2=0.9)
        elif optimizer.lower() == 'rmsprop':
            opt = RMSprop(lr=self.lr)

        self.generator.trainable = False
        self.discriminator.trainable = False

        self.encod.compile(optimizer=opt, loss=self.encoder_class.encoder_loss())

    def gen_batch(self, batch_size):
        latent_vector_batch = self.make_noise(batch_size)
        gen_output = self.generator.predict_on_batch(latent_vector_batch)
        return gen_output

    def get_batch(self, batch_size, train=True):
        if train:
            idx = np.random.choice(np.shape(self.x_train)[0], batch_size, replace=False)
            return self.x_train[idx]
        else:
            idx = np.random.choice(np.shape(self.x_test)[0], batch_size, replace=False)
            return self.x_test[idx]

    def regen_batch(self, batch):

        enc_output = self.encod.predict_on_batch(batch)
        gen_output = self.generator.predict_on_batch(enc_output)
        return gen_output

    def make_noise(self, batch_size):
        noise = np.random.normal(scale=0.5, size=(tuple([batch_size]) + self.z_size))
        return noise

    def plot_dict(self, dictionary):
        for key, item in dictionary.items():
            plt.close()
            plt.plot(range(len(item)), item)
            plt.title(str(key))
            plt.savefig(os.path.join(self.plot_path, '{}.png'.format(key)), bbox_inches='tight')

    def train(self, num_epochs=25, batch_size=32):
        batches_per_epoch = np.shape(self.x_train)[0] // batch_size
        stats = {'encoder_loss': []}

        for epoch in range(num_epochs):
            print('Epoch: {}. Training {}% complete.'.format(
                epoch, np.around(100 * epoch / num_epochs, decimals=1)))

            for i in range(batches_per_epoch):

                data_batch = self.get_batch(batch_size)
                regen_batch = self.regen_batch(data_batch)
                enc_loss = self.encod.train_on_batch(regen_batch, data_batch)
                stats['encoder_loss'].append(enc_loss)

        print(stats)
        self.plot_dict(stats)


if __name__ == '__main__':
    encod = Encoder()
    trainer = Trainer(encod)
    trainer.train()