import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from encoder import Encoder
from keras.datasets import mnist
from keras.models import load_model
from keras.optimizers import Adam, RMSprop
plt.switch_backend('agg')


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


class Trainer:

    def __init__(self, encoder, optimizer='adam', plot_path='plots'):
        assert optimizer.lower() in ['adam', 'rmsprop'], "Optimizer unrecognized or unavailable."

        if not os.path.isdir(plot_path):
            os.mkdir(plot_path)

        self.adversarial = load_model('wgan.h5')
        self.discriminator = self.adversarial.discriminator
        self.generator = self.adversarial.generator
        self.encoder = encoder
        self.z_size = encoder.z_size
        self.lr = encoder.lr
        self.x_train, self.x_test = dataset()
        self.model_compiler(optimizer)
        self.plot_path = plot_path

    def model_compiler(self, optimizer):
        if optimizer.lower() == 'adam':
            opt = Adam(lr=self.lr, beta_1=0.5, beta_2=0.9)
        elif optimizer.lower() == 'rmsprop':
            opt = RMSprop(lr=self.lr)

        self.adversarial.trainable = False
        self.generator.trainable = False
        self.discriminator.trainable = False

        self.encoder.compile(optimizer=opt, loss=self.encoder.encoder_loss)

    def gen_batch(self, batch_size):
        latent_vector_batch = self.make_noise(batch_size)
        gen_output = self.generator.predict_on_batch(latent_vector_batch)
        return gen_output

    def make_noise(self, batch_size):
        noise = np.random.normal(scale=0.5, size=(tuple([batch_size]) + self.z_size))
        return noise

    def train(self, num_epochs=25, batch_size=32):
        batches_per_epoch = np.shape(self.x_train)[0] // batch_size
        stats = {'encoder_loss': []}

        gen_iterations = 0

        for epoch in range(num_epochs):
            print('Epoch: {}. Training {}% complete.'.format(
                epoch, np.around(100 * epoch / num_epochs, decimals=1)))

            if (epoch + 1) % 5 == 0:
                self.make_images(epoch + 1, num_images=3)

            for i in range(batches_per_epoch):

                # Train with a batch of generator (fake) data.
                gen_batch = self.gen_batch(batch_size)
                disc_loss_fake = self.discriminator.train_on_batch(gen_batch, -np.ones(batch_size))

                # Train generator.

                noise = self.make_noise(batch_size)

                self.discriminator.trainable = False
                gen_loss = self.adversarial.train_on_batch(noise, np.ones(batch_size))
                self.discriminator.trainable = True

                stats['generator_loss'].append(gen_loss)
                stats['wasserstein_distance'].append(-(disc_loss_real + disc_loss_fake))

                gen_iterations += 1

        self.plot_dict(stats)


if __name__ == '__main__':
    trainer = Trainer(Encoder)