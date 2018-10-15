import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, BatchNormalization, Flatten, Reshape
from Algorithm import Algorithm


class LSTM:
    def __init__(self, dm):
        self.x_train, self.y_train = dm.load_training_data()
        self.x_test, self.y_test = dm.load_testing_data()
        self.x_valid, self.y_valid = dm.load_validation_data()

        self.x_train = self.x_train.reshape((len(self.x_train), dm.img_rows, dm.img_cols))
        self.x_valid = self.x_valid.reshape((len(self.x_valid), dm.img_rows, dm.img_cols))
        self.x_test = self.x_test.reshape((len(self.x_test), dm.img_rows, dm.img_cols))

        # print("Building Model...")
        self.model = Sequential()
        # self.model.add(BatchNormalization(input_shape=input_shape))
        # print("Adding layer 0")
        # self.model.add(Reshape(dm.input_shape, input_shape=(*dm.input_shape, 1)))
        # self.model.add(Dense(105, activation='relu'))
        # self.model.add(BatchNormalization())
        # self.model.add(Flatten())
        self.model.add(keras.layers.LSTM(264, activation='tanh', return_sequences=True, input_shape=(105, 105)))
        self.model.add(BatchNormalization())
        self.model.add(keras.layers.LSTM(264, activation='tanh', return_sequences=True))
        self.model.add(Flatten())
        # print("Adding output layer")
        self.model.add(Dense(dm.num_classes, activation='softmax'))
        # print("Compiling the model")
        self.model.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy'])

    def toAlg(self):
        return Algorithm(self.model,
                         self.x_train,
                         self.x_test,
                         self.x_valid,
                         self.y_train,
                         self.y_test,
                         self.y_valid)
