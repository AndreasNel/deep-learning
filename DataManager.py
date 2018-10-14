from torchvision.datasets import Omniglot
import torchvision.transforms as transforms
import numpy as np
from sklearn.model_selection import train_test_split


class DataManager:
    def __init__(self, test_size=0.3, random_state=0):
        self.dataset = Omniglot(
            root='data/',
            background=True,
            download=True,
            transform=transforms.ToTensor())

        self.data = np.array([t.numpy()[0] for t, _ in self.dataset])
        self.labels = np.array([l for _, l in self.dataset])
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.data, self.labels, test_size=test_size, random_state=random_state)

    def loadTestingData(self):
        return self.x_test, self.y_test

    def loadTrainingData(self):
        return self.x_train, self.y_train
