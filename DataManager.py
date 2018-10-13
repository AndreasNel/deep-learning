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

        self.X_train = np.array([t.numpy() for t, l in self.training_data])
        self.X_test = np.array([t.numpy() for t, l in self.testing_data])
        self.Y_train = np.array([l for t, l in self.training_data])
        self.Y_test = np.array([l for t, l in self.testing_data])

    def loadTestingData(self):
        return (self.X_test, self.Y_test)

    def loadTrainingData(self):
        return (self.X_train, self.Y_train)
