from keras.preprocessing.image import ImageDataGenerator

class ImageGenerator:

    def __init__(self, test_size=0.2, eval_size=0.2, random_state=0):
        self.data_generator = ImageDataGenerator(
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

    def fit(self, training_data):
        self.data_generator.fit(training_data)

    def flow(self, training_data, training_labels, batch_size):
        return self.data_generator.flow(
            training_data,
            training_labels,
            batch_size
        )