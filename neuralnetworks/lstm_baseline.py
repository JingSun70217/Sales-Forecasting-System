import pandas as pd
import numpy as np
import pickle

# Import datasets
data_directory = '../data/clean'
X = pd.read_csv('{}/nn_X.csv'.format(data_directory), index_col=0)
Y = pd.read_csv('{}/nn_Y.csv'.format(data_directory), index_col=0)

# Drop columns
keys_dates = pd.DataFrame(X['key']).join(X['date']) # Store for future lookups
drop_x_cols = ['key', 'pid_x', 'size_x', 'color', 'brand', 'rrp', 'date', 'day_of_week',
               'mainCategory', 'category', 'subCategory', 'releaseDate']
drop_y_cols = ['key', 'date']
X = X.drop(drop_x_cols, axis=1)
Y = Y.drop(drop_y_cols, axis=1)

# Convert to numpy to reshape for input
X = X.as_matrix()
Y = Y.as_matrix()

# Reshape to get data into (# samples, # timesteps, # variables)
# Training data will be shape (12,824, 92, # variables)
X_tr = X[0:92]
X_tr = X_tr[np.newaxis, :, :]

for i in range(1, X.shape[0] // 123):
    temp_x = X[i*123:i*123+92]
    temp_x = temp_x[np.newaxis, :, :]
    X_tr = np.concatenate((X_tr, temp_x), axis=0)
print(X_tr.shape)  # Check if shape is correct

# Reshape Y values the same way
Y_tr = Y[0:92]
Y_tr = Y_tr[np.newaxis, :, :]

for i in range(1, Y.shape[0]//123):
    temp_y = Y[i*123:i*123+92]
    temp_y = temp_y[np.newaxis, :, :]
    Y_tr = np.concatenate((Y_tr, temp_y), axis=0)

# Baseline model
from keras.models import Sequential
from keras.layers import Dense, Activation, TimeDistributed, LSTM

num_epochs = 20
batch_size = 8 # Try to maximise memory per batch
num_hidden = 32
timesteps = X_tr.shape[1]
input_dim = X_tr.shape[2]

model = Sequential()
model.add(LSTM(num_hidden, input_shape=(timesteps, input_dim), dropout=0, return_sequences=True))
# Output has shape (batch_size, timesteps, num_hidden), so we add a dense layer to make it (batch_size, timesteps, 1)
model.add(TimeDistributed(Dense(1, activation='relu', kernel_initializer='random_uniform'))) # With ReLu activation to clamp minimum to 0
# We can then compare predictions directly with Y of size (batch_size, timesteps, 1)
model.compile(loss='mean_absolute_error', optimizer='adam')
print(model.summary())

from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
callbacks_list = [EarlyStopping(monitor='loss', patience=5),
                  ModelCheckpoint(filepath='rnn-lstm-best_v1.h5', monitor='loss', save_best_only=True)]
# ReduceLROnPlateau(monitor='loss', factor=0.2, patience=2, verbose=1, mode='min', cooldown=0, min_lr=0)

# Train LSTM; running this will take a while!
history = model.fit(X_tr, Y_tr, batch_size=batch_size,
                    callbacks=callbacks_list,
                    epochs = num_epochs, verbose=1)

# Save model and history for future reuse
model.save('rnn-lstm_v1.h5')
with open('rnn-lstm-history_v1', 'wb') as file_pi:
    pickle.dump(history.history, file_pi)