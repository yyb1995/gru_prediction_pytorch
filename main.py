import torch.optim as opt
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from preprocessing import generate_data
from model import GRU
from train_eval import train, predict

# Network config
input_length = 50
output_length = 1
batch_size = 256
hidden_size = 128
num_layers = 1
dropout = 0
testnum = 500
# interval is sample interval between last input and first output.
interval = 0

epoch = 100
device = 'cuda'


# Generate sin dataset for training and testing.
dataset = np.sin([i / 50 * 2 * np.pi for i in range(2000)])
x_train, y_train, x_test, y_test, normalizer = generate_data(
    dataset, 'minmax', input_length, output_length, testnum, interval)

# Build, train and predict.
model = GRU(1, hidden_size, num_layers, 1, dropout)
optimizer = opt.Adam(model.parameters())
loss = nn.MSELoss()
batch_train_loss, batch_val_loss = train(
    model, x_train, y_train, epoch, batch_size, optimizer, loss, device)
y_predict, y_real, _ = predict(model, x_test, y_test, loss, device, normalizer, batch_size)

# Draw result
plt.plot(y_predict, label='prediction')
plt.plot(y_real, label='real')
plt.legend()
plt.show()
