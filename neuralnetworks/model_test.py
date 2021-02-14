from neuralnetworks import gen_data
from keras.models import Sequential
from keras.layers import Dense, Activation, TimeDistributed, LSTM
import numpy as np

sample_size = 100
time_steps = 50
input_dim = 10

x, y = gen_data.gen_samples(sample_size, time_steps, input_dim)

# Model hyperparameters
num_epochs = 50
batch_size = 256 # Try to maximise memory per batch
num_hidden = 64

# Model definition; includes further hyperparameters, e.g. optimizer, loss function, learning rates
model = Sequential()

# LSTM
# model.add(LSTM(num_hidden, input_shape=(time_steps, input_dim), dropout=0, return_sequences=True))
# # model.add(LSTM(num_hidden, input_shape=(time_steps, num_hidden), dropout=0, return_sequences=True)) # If deep LSTM
# model.add(TimeDistributed(Dense(1, activation='relu', kernel_initializer='random_uniform')))
# model.compile(loss='mean_absolute_error', optimizer='adam')
# print(model.summary())

# ANN
model.add(Dense(num_hidden, input_shape=(time_steps, input_dim), activation='relu'))
model.add(Dense(num_hidden, activation='relu'))
model.add(Dense(num_hidden, activation='relu'))
model.add(Dense(num_hidden, activation='relu'))
model.add(Dense(1, activation='relu'))
model.compile(loss='mean_absolute_error', optimizer='adam')
print(model.summary())
print(x.shape)
print(y.shape)

# Train model; running this will take a while!
history = model.fit(x, y, epochs = num_epochs, verbose=1) # Use batch_size argument to pass data in batches