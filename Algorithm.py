from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from DataManager import DataManager
from IO import IO


class Algorithm:
    def __init__(self, model, x_train, x_test, x_valid, y_train, y_test, y_valid):
        self.model = model
        self.x_train = x_train
        self.x_test = x_test
        self.x_valid = x_valid
        self.y_train = y_train
        self.y_test = y_test
        self.y_valid = y_valid

    def print_summary(self):
        print(self.model.summary())

    def run(self, batch_size, epochs, name):
        history = self.model.fit(self.x_train, self.y_train,
                                 batch_size=batch_size,
                                 epochs=epochs,
                                 verbose=1, validation_data=(self.x_valid, self.y_valid))

        self.save(history, batch_size, name)

    def run_with_flow(self, flow, batch_size, epochs, name):
        history = self.model.fit_generator(flow,
                                           steps_per_epoch=10,
                                           epochs=epochs,
                                           verbose=1, validation_data=(self.x_valid, self.y_valid))
        self.save(history, batch_size, name)

    def save(self, history, batch_size, name):
        train_score = self.model.evaluate(self.x_train, self.y_train, batch_size=batch_size)
        test_score = self.model.evaluate(self.x_test, self.y_test, batch_size=batch_size)
        io = IO(history, train_score, test_score, name)
