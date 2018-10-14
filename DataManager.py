from torchvision.datasets import Omniglot
import torchvision.transforms as transforms
import numpy as np
from sklearn.model_selection import train_test_split


class DataManager:
    def __init__(self, test_size=0.2, eval_size=0.2, random_state=0):
        self.background = Omniglot(
            root='data/',
            background=True,
            download=True,
            transform=transforms.ToTensor())

        self.evaluation = Omniglot(
            root='data/',
            background=False,
            download=True,
            transform=transforms.ToTensor())

        self.background_data = np.array(
            [t.numpy()[0] for t, _ in self.background])
        self.evaulation_data = np.array(
            [t.numpy()[0] for t, _ in self.evaluation])

        self.background_labels = np.array([l for _, l in self.background])
        self.evaluation_labels = np.array(
            [l for _, l in self.evaluation_labels])

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            np.concatenate((self.background_data, self.evaluation_data)),
            np.concatenate((self.background_labels, self.evaluation_labels)),
            test_size=test_size,
            random_state=random_state)

        self.x_train, self.x_valid, self.y_train, self.y_valid = train_test_split(
            self.x_train,
            self.y_train,
            test_size=test_size,
            random_state=random_state)

    def loadTestingData(self):
        return self.x_test, self.y_test

    def loadTrainingData(self):
        return self.x_train, self.y_train

    def loadValidationData(self):
        return self.x_valid, self.y_valid
