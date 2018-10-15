import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, BatchNormalization, Flatten, Reshape
from keras import regularizers
from keras.preprocessing.image import ImageDataGenerator
from keras import utils
from DataManager import DataManager

print("Loading training data...")
dm = DataManager(random_state=0)
training_data, training_labels = dm.loadTrainingData()
testing_data, testing_labels = dm.loadTestingData()
validation_data, validation_labels = dm.loadValidationData()

print('Loaded shapes')
for i in training_data, training_labels, testing_data, testing_labels, validation_data, validation_labels:
    print(i.shape)

input_shape = tuple(training_data.shape[1:])
num_classes = len(np.unique(training_labels))
print("input_shape: {}".format(input_shape))
print("num_classes: {}".format(num_classes))

# Convert to categorical classes
training_labels = utils.to_categorical(training_labels, num_classes)
testing_labels = utils.to_categorical(testing_labels, num_classes)
validation_labels = utils.to_categorical(validation_labels, num_classes)

data_generator = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=False,
    vertical_flip=False
)

data_generator.fit(training_data.reshape((len(training_data), *input_shape, 1)))

# Build the model
print("Building Model...")
model = Sequential()
# model.add(BatchNormalization(input_shape=input_shape))
print("Adding layer 0")
model.add(Reshape(input_shape, input_shape=(*input_shape, 1)))
# model.add(Dense(105, activation='relu'))
# model.add(BatchNormalization())
# model.add(Flatten())
model.add(LSTM(264, activation='tanh', return_sequences=True))
model.add(BatchNormalization())
model.add(LSTM(264, activation='tanh', return_sequences=True))
model.add(Flatten())
print("Adding output layer")
model.add(Dense(num_classes, activation='softmax'))
print("Compiling the model")
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
print("Training the model")
batch_size = 160
history = model.fit_generator(
    data_generator.flow(
        training_data.reshape((len(training_data), *input_shape, 1)),
        training_labels,
        batch_size=batch_size
    ),
    validation_data=(validation_data.reshape((len(validation_data), *input_shape, 1)), validation_labels),
    epochs=30,
    verbose=1,
)
# Test the model with the testing data.
# print("Testing the model")
# score = model.evaluate(testing_data, testing_labels)

