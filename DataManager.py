from torchvision.datasets import Omniglot
import torchvision.transforms as transforms
import numpy as np
from sklearn.model_selection import train_test_split
from keras import utils
from keras import backend as K

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

        self.num_classes, self.img_rows, self.img_cols = 1623, 105, 105

        self.background_data = np.array([t.numpy()[0] for t, _ in self.background])
        self.evaluation_data = np.array([t.numpy()[0] for t, _ in self.evaluation])

        self.background_labels = np.array([l for _, l in self.background])
        # Due to labels in the evaluation data also starting from 0, they are offset in order
        # to follow immediately after the background labels.
        self.evaluation_labels = np.array([l for _, l in self.evaluation]) + np.max(self.background_labels) + 1

        # Split the entire data set into a training and testing set.
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            np.concatenate((self.background_data, self.evaluation_data)),
            np.concatenate((self.background_labels, self.evaluation_labels)),
            test_size=test_size,
            random_state=random_state)

        # Split the training further into a final training set and a validation set.
        self.x_train, self.x_valid, self.y_train, self.y_valid = train_test_split(
            self.x_train,
            self.y_train,
            test_size=eval_size,
            random_state=random_state)

        self.y_test = utils.to_categorical(self.y_test, self.num_classes)
        self.y_train = utils.to_categorical(self.y_train, self.num_classes)
        self.y_valid = utils.to_categorical(self.y_valid, self.num_classes)

        if K.image_data_format() == 'channels_first':
            self.x_train = self.x_train.reshape(self.x_train.shape[0], 1, self.img_rows, self.img_cols)
            self.x_test = self.x_test.reshape(self.x_test.shape[0], 1, self.img_rows, self.img_cols)
            self.x_valid = self.x_valid.reshape(self.x_valid.shape[0], 1, self.img_rows, self.img_cols)
            self.input_shape = (1, self.img_rows, self.img_cols)
        else:
            self.x_train = self.x_train.reshape(self.x_train.shape[0], self.img_rows, self.img_cols, 1)
            self.x_test = self.x_test.reshape(self.x_test.shape[0], self.img_rows, self.img_cols, 1)
            self.x_valid = self.x_valid.reshape(self.x_valid.shape[0], self.img_rows, self.img_cols, 1)
            self.input_shape = (self.img_rows, self.img_cols, 1)

        self.x_train = self.x_train.astype('float32')
        self.x_test = self.x_test.astype('float32')
        self.x_valid = self.x_valid.astype('float32')

    def load_testing_data(self):
        return self.x_test, self.y_test

    def load_training_data(self):
        return self.x_train, self.y_train

    def load_validation_data(self):
        return self.x_valid, self.y_valid
