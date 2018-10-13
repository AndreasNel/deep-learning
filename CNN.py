from DataManager import DataManager
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Convolution2D, MaxPooling2D
import numpy as np

np.random.seed(123)  # For reproducibility

# Load pre-shuffled MNIST data into train and test setthe omniglot data

dm = DataManager()  # To access the omniglot data

(X_train, Y_train) = dm.loadTrainingData()
(X_test, Y_test) = dm.loadTestingData()

# Define the model architecture
model = Sequential()

model.add(
    Convolution2D(
        32, (3, 3),
        activation='relu',
        input_shape=(1, 105, 105),
        data_format='channels_first'))

model.add(Convolution2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='softmax'))

# Compile the model
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

# Fit the model on training data
model.fit(X_train, Y_train, batch_size=32, epochs=10, verbose=1)

# Evaluate the model on test data
score = model.evaluate(X_test, Y_test, verbose=0)
