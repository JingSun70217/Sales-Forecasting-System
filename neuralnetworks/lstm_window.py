import keras
import pandas as pd
import numpy as np
import os
import pickle

print(os.path.abspath('.')) # 'C:\Users\iamlcc\datamining2\neuralnetworks' or 'C:\Users\iamlcc\datamining2'

# Import dataset and split to train/test set
data_directory = './data/clean'
X = pd.read_csv('{}/nn_X.csv'.format(data_directory), index_col=0)
Y = pd.read_csv('{}/nn_Y.csv'.format(data_directory), index_col=0)

print(X.shape)
print(Y.shape)
print(X.columns)
print(Y.columns)

# Drop columns
drop_x_cols = ['key', 'pid_x', 'size_x', 'color', 'brand', 'rrp', 'date', 'day_of_week',
               'mainCategory', 'category', 'subCategory', 'releaseDate']
drop_y_cols = ['key', 'date']
X = X.drop(drop_x_cols, axis=1)
Y = Y.drop(drop_y_cols, axis=1)

# Convert to numpy to reshape for input
X = X.as_matrix() # Each row has shape (num_vars,)
Y = Y.as_matrix() # Each row has shape (1,)

# Model training
# We need a data generator to create mini-batches for us
# Define generator required to feed X, Y samples into model
class BatchGenerator(object):
    def __init__(self, X, Y, batch_size, window_size, num_vars, start_day, end_day):
        self.X = X
        self.Y = Y
        self.window_size = window_size
        self.batch_size = batch_size
        self.num_vars = num_vars
        self.start_day = start_day # 0 = day 1
        self.end_day = end_day
        self.current_product = 0 # To keep track of which product we are at (of 12,824)
        self.current_day = 0 # To track which day of a product we are in (of 123)

    def generate(self):
        x = np.zeros((self.batch_size, self.window_size, self.num_vars))
        y = np.zeros((self.batch_size, 1)) # Should this be 3d?
        while True:
            for i in range(self.batch_size):
                if (self.current_day+self.start_day+self.window_size) == self.end_day:
                    # Go to next product, first day
                    self.current_product += 1
                    self.current_day = 0
                if self.current_product == 12824:
                    # Go back to first product for next epoch
                    self.current_product = 0
                x[i,:,:] = self.X[self.current_product*123+self.current_day+self.start_day:
                                  self.current_product*123+self.current_day+self.start_day+self.window_size]
                y[i,:] = self.Y[self.current_product*123+self.current_day+self.window_size]
                self.current_day += 1 # Since batch_size will be = 1
            yield x, y

WINDOW_SIZE = 30
num_vars = X.shape[1]
batch_size = 8 # 12,824 divisible by 8

train_start_day = 0
train_end_day = 92 # 2017-10-01 to 2017-12-31
test_start_day = 92 - WINDOW_SIZE
test_end_day = 123 # 2018-01-01 to 2018-01-31

train_data_generator = BatchGenerator(X, Y, batch_size, WINDOW_SIZE, num_vars, train_start_day, train_end_day)
test_data_generator = BatchGenerator(X, Y, batch_size, WINDOW_SIZE, num_vars, test_start_day, test_end_day)

## Model definition; this model is a moving-window approach.
# For each product, we push 1 window-frame per batch.
# When the frame reaches the end of the window, we move to the next product.
# When we have finished all the window frames for all products, that is the end of 1 epoch.
# E.g. window_size = 30, start_day = 0, end_day = 92, then for each product, we will push in 62 window frames
# i.e. [0, 30], [1, 31], [2, 32], ..., [62, 92]

# Custom metric to get mean absolute difference
# Metric function is similar to a loss function
# except that the results from evaluating a metric are not used when training the model
import keras.backend as backend
def mean_abs_diff(y_true, y_pred):
    return backend.mean(backend.abs(y_true - y_pred))

# Using 2-layered LSTM, because I want the first layer to learn something about the whole window-frame,
# and the second layer to use the info from the first layer to make a prediction on the last day of the window-frame

# Windowing model
# Model predicts only sales unit of last day of the window
from keras.models import Sequential
from keras.layers import Dense, Activation, TimeDistributed, LSTM

num_epochs = 1 # This is too low; can increase if we push training to the cloud
num_hidden = 32

model = Sequential()
model.add(LSTM(num_hidden, input_shape=(WINDOW_SIZE, num_vars), dropout=0, return_sequences=True))
model.add(LSTM(num_hidden, dropout=0.5, return_sequences=False))
model.add(Dense(1, activation='relu')) # Need kernel_initializer?
model.compile(loss='mean_absolute_error', optimizer='adadelta', metrics=[mean_abs_diff])
print(model.summary())

from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

callbacks_list = [EarlyStopping(monitor='val_loss', patience=5),
                  ModelCheckpoint(filepath='lstm_v2_best.h5', monitor='val_loss',save_best_only=True)]
# ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, verbose=1, mode='min', cooldown=0, min_lr=0)

# Train LSTM; running this will take a while!
history = model.fit_generator(generator=train_data_generator.generate(),
                              steps_per_epoch=12824*(train_end_day-WINDOW_SIZE)/batch_size,
                              validation_data=test_data_generator.generate(),
                              validation_steps=12824*(test_end_day-WINDOW_SIZE)/batch_size,
                              callbacks=callbacks_list,
                              epochs = num_epochs, verbose=1,
                              shuffle=False)

# Save model and history for future reuse
model.save('lstm_v2.h5')
with open('lstm_history_v2', 'wb') as file_pi:
    pickle.dump(history.history, file_pi)