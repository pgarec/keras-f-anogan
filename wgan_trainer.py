
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dcgan import DCGAN
from keras.datasets import mnist
from keras.optimizers import Adam, RMSprop
plt.switch_backend('agg')

class Trainer:
    def __init__(self, dcgan, optimizer='adam', plot_path='plots'):
        assert optimizer.lower() in ['adam', 'rmsprop'], "Optimizer unrecognized or unavailable."

        if not os.path.isdir(plot_path):
            os.mkdir(plot_path)

        self.dcgan = dcgan
        self.z_size = dcgan.z_size
        self.lr = dcgan.lr
        self.discriminator = dcgan.discriminator()
        self.generator = dcgan.generator()
        self.adversarial = dcgan.adversarial(self.generator, self.discriminator)
        self.x_train, self.x_test = self.dataset()
        self.model_compiler(optimizer)
        self.plot_path = plot_path

    def model_compiler(self, optimizer):
        if optimizer.lower() == 'adam':
            opt = Adam(lr=self.lr, beta_1=0.5, beta_2=0.9)
        elif optimizer.lower() == 'rmsprop':
            opt = RMSprop(lr=self.lr)

        self.discriminator.compile(optimizer=opt, loss=self.dcgan.wasserstein_loss)
        self.generator.compile(optimizer=opt, loss=self.dcgan.wasserstein_loss)

        self.discriminator.trainable = False
        self.adversarial.compile(optimizer=opt, loss=self.dcgan.wasserstein_loss)
        self.discriminator.trainable = True

    def dataset(self):

        """
        Load dataset, convert to 32x32, constrain input to [-1, 1].
        """

        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        train_filter = np.where((y_train == 0))
        test_filter = np.where((y_test == 0))
        (x_train, y_train) = x_train[train_filter], y_train[train_filter]
        (x_test, y_test) = y_test[test_filter], y_test[test_filter]

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

    def get_batch(self, batch_size, train=True):
        if train:
            idx = np.random.choice(np.shape(self.x_train)[0], batch_size, replace=False)
            return self.x_train[idx]
        else:
            idx = np.random.choice(np.shape(self.x_test)[0], batch_size, replace=False)
            return self.x_test[idx]

    def make_noise(self, batch_size):
        noise = np.random.normal(scale=0.5, size=(tuple([batch_size]) + self.z_size))
        return noise

    def gen_batch(self, batch_size):
        latent_vector_batch = self.make_noise(batch_size)
        gen_output = self.generator.predict_on_batch(latent_vector_batch)
        return gen_output

    def plot_dict(self, dictionary):
        for key, item in dictionary.items():
            plt.close()
            plt.plot(range(len(item)), item)
            plt.title(str(key))
            plt.savefig(os.path.join(self.plot_path, '{}.png'.format(key)), bbox_inches='tight')

    def make_images(self, epoch, num_images=10):
        noise = self.make_noise(num_images)
        digits = self.generator.predict(noise).reshape(-1, 32, 32)

        m = 0
        for digit in digits:
            plt.close()
            image = sns.heatmap(digit, cbar=False, square=True)
            plt.xticks([])
            plt.yticks([])
            plt.tight_layout(pad=0)
            plt.savefig(os.path.join(self.plot_path, 'epoch_{}-image_{}.png'.format(epoch, m)), bbox_inches='tight')
            m += 1

    def train(self, num_epochs=25, batch_size=32):
        batches_per_epoch = np.shape(self.x_train)[0]//batch_size
        stats = {'wasserstein_distance': [], 'generator_loss': []}

        gen_iterations = 0

        for epoch in range(num_epochs):
            print('Epoch: {}. Training {}% complete.'.format(
                    epoch, np.around(100*epoch/num_epochs, decimals=1)))

            if (epoch + 1) % 5 == 0:
                self.make_images(epoch + 1, num_images=3)

            for i in range(batches_per_epoch):

                # Train discriminator.

                if gen_iterations < 25 or gen_iterations % 500 == 0:
                    disc_iterations = 100
                else:
                    disc_iterations = self.dcgan.disc_iters_per_gen_iters

                for j in range(disc_iterations):

                    for l in self.discriminator.layers:
                        weights = l.get_weights()
                        weights = [np.clip(w, self.dcgan.clamp_lower, self.dcgan.clamp_upper) for w in weights]
                        l.set_weights(weights)

                    # Train with a batch of real data.
                    data_batch = self.get_batch(batch_size)
                    disc_loss_real = self.discriminator.train_on_batch(data_batch, np.ones(batch_size))

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
    dcgan = DCGAN()
    trainer = Trainer(dcgan)
    trainer.train()
    trainer.generator.save('gen.h5')
    trainer.discriminator.save('disc.h5')