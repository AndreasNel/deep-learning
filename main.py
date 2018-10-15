from CNN import CNN
from DataManager import DataManager
from ImageGenerator import ImageGenerator
from LSTM import LSTM

batch_size = 200
epochs = 30

dm = DataManager()

ig = ImageGenerator()
ig.fit(dm.x_train)
flow = ig.flow(dm.x_train, dm.y_train, batch_size)
CNN(dm).toAlg().run(batch_size, epochs, 'cnn_standard')
CNN(dm).toAlg().run_with_flow(flow, batch_size, epochs, 'cnn_IG')

LSTM(dm).toAlg().run(batch_size, epochs, 'lstm')
# LSTM(dm).toAlg().run_with_flow(flow, batch_size, epochs, 'lstm_IG')
