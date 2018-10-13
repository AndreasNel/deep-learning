from torchvision.datasets import Omniglot
import torchvision.transforms as transforms
import numpy as np


class DataManager:
    def __init__(self):
        self.training_data = Omniglot(
            root='data/training',
            background=True,
            download=True,
            transform=transforms.ToTensor())

        self.testing_data = Omniglot(
            root='data/testing_data',
            background=False,
            download=True,
            transform=transforms.ToTensor())

        self.x_train = np.array([t.numpy()[0] for t, l in self.training_data])
        self.x_test = np.array([t.numpy()[0] for t, l in self.testing_data])
        self.y_train = np.array([l for t, l in self.training_data])
        self.y_test = np.array([l for t, l in self.testing_data])
