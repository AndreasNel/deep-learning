import matplotlib.pyplot as plt

class IO:

    def __init__(self, history, train_score, test_score, name):
        self.history = history
        self.train_score = test_score
        self.test_score = test_score
        self.name = name

    def run(self):
        # Plot training & validation accuracy values
        plt.plot(self.history.history['acc'])
        plt.plot(self.history.history['val_acc'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.savefig(self.name + "_accuracy")

        plt.clf()  # Clear figure

        # Plot training & validation loss values
        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.savefig(self.name + "_val_loss")

        print(self.history.history)
        print("Train score: ", self.train_score)
        print("Test score: ", self.test_score)
