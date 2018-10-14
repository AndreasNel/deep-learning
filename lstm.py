import numpy as np
from keras.models import Sequential
from keras.layers import LSTM
from keras import backend as K
from DataManager import DataManager

print("Loading training data...")
dm = DataManager()
training_data, training_labels = dm.loadTrainingData()
testing_data, testing_labels = dm.loadTestingData()

print('Loaded shapes')
for i in training_data, training_labels, testing_data, testing_labels:
    print(i.shape)

num_classes, num_rows, num_cols = 964, 105, 105
input_shape = (1, num_rows, num_cols) if K.image_data_format() == 'channels_first' else (num_rows, num_cols, 1)
print("input_shape: {}".format(input_shape))
training_data = training_data.reshape((training_data.shape[0],) + input_shape).astype('float32')
testing_data = testing_data.reshape((testing_data.shape[0],) + input_shape).astype('float32')

print('Reshaped shapes')
for i in training_data, training_labels, testing_data, testing_labels:
    print(i.shape)


raise Exception('asdfljkafdsljkh')

# TODO build the model in a way corresponding to the correct data
# Build the model
print("Building Model...")
model = Sequential()
print("Adding layer 1")
model.add(LSTM(20, input_shape=(105, 105), return_sequences=True, activation='relu'))
# print("Adding layer 2")
# model.add(LSTM(1, return_sequences=False, activation='softmax'))
print("Compiling the model")
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print("Training the model")
model.fit(training_data, training_labels, epochs=1000)
model.summary()
# Test the model with the testing data.
print("Testing the model")
predict = model.predict(training_data)

# TODO run the stats
