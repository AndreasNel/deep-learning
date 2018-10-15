from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from Algorithm import Algorithm


class CNN:
    def __init__(self, dm):
        self.x_train, self.y_train = dm.load_training_data()
        self.x_test, self.y_test = dm.load_testing_data()
        self.x_valid, self.y_valid = dm.load_validation_data()

        self.model = Sequential()
        self.model.add(Conv2D(32, kernel_size=(3, 3),
                              activation='relu',
                              input_shape=dm.input_shape))
        # self.model.add(Conv2D(64, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        # self.model.add(Dropout(0.25))
        self.model.add(Flatten())
        # self.model.add(Dense(128, activation='relu'))
        # self.model.add(Dropout(0.5))
        self.model.add(Dense(dm.num_classes, activation='softmax'))

        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    def toAlg(self):
        return Algorithm(self.model,
                         self.x_train,
                         self.x_test,
                         self.x_valid,
                         self.y_train,
                         self.y_test,
                         self.y_valid)
