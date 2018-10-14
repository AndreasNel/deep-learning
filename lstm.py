import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, BatchNormalization, Flatten
from keras import regularizers
from keras import utils
from DataManager import DataManager

print("Loading training data...")
dm = DataManager()
training_data, training_labels = dm.loadTrainingData()
testing_data, testing_labels = dm.loadTestingData()

print('Loaded shapes')
for i in training_data, training_labels, testing_data, testing_labels:
    print(i.shape)

input_shape = tuple(training_data.shape[1:])
num_classes = len(np.unique(training_labels))
print("input_shape: {}".format(input_shape))
print("num_classes: {}".format(num_classes))

# Convert to categorical classes
training_labels = utils.to_categorical(training_labels, num_classes)
testing_labels = utils.to_categorical(testing_labels, num_classes)

# Build the model
print("Building Model...")
model = Sequential()
model.add(BatchNormalization(input_shape=input_shape))
print("Adding layer 0")
model.add(LSTM(105, activation='sigmoid', return_sequences=True))
model.add(LSTM(105, activation='sigmoid', return_sequences=True))
# model.add(LSTM(105, activation='sigmoid', return_sequences=True))
# print("Adding hidden layer")
# model.add(LSTM(105, activation='sigmoid', return_sequences=True))
model.add(Flatten())
print("Adding output layer")
model.add(Dense(num_classes, activation='softmax'))
print("Compiling the model")
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
print("Training the model")
history = model.fit(training_data, training_labels, batch_size=320, epochs=30, verbose=1)
# Test the model with the testing data.
print("Testing the model")
score = model.evaluate(testing_data, testing_labels)

