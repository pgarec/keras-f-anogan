import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dcgan import DCGAN
from keras.datasets import mnist
from keras.optimizers import Adam, RMSprop
plt.switch_backend('agg')

class Trainer:

    def __init__(self, encoder, optimizer='adam', plot_path='plots'):
        assert optimizer.lower() in ['adam', 'rmsprop'], "Optimizer unrecognized or unavailable."

        if not os.path.isdir(plot_path):
            os.mkdir(plot_path)

        self.dcgan = encoder
        self.z_size = encoder.z_size
        self.lr = dcgan.lr
        self.x_train, self.x_test = self.dataset()
        self.model_compiler(optimizer)
        self.plot_path = plot_path
