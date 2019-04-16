___author___ = 'Chung Chi Leung'

import pandas as pd
import os
from measures import wrangling

dataset = pd.read_csv('./data/clean/data_v0.1.csv', header=0)

print(dataset.columns)

# Drop February price-columns
dataset = wrangling.remove_price_cols(dataset, '2018-02-01', '2018-02-28')
print(dataset.columns)

# Separate Y labels from X variables
Y_train = wrangling.sales_cols(dataset, '2017-10-01', '2017-12-31')
Y_test = wrangling.sales_cols(dataset, '2018-01-01', '2018-01-31')
X_dataset = wrangling.remove_sales_cols(dataset, '2017-10-01', '2018-01-31')

# Drop more unnecessary columns from X
X_dataset = X_dataset.drop(['pid_x', 'size_x', 'stock'], axis=1)

# Flatten X so that each row is 1 day; we expect 12,824*123 = 1,577,352 rows as result
cols = ['key', 'color', 'brand', 'rrp', 'mainCategory', 'category', 'subCategory', 'releaseDate']
X_flat = pd.melt(X_dataset, id_vars=cols, var_name='date', value_name='price')
X_flat = X_flat.sort_values(['key', 'date']).reset_index(drop=True)
print(X_flat.shape)
print(X_flat.tail()) # Quick check of the result

# Flatten Y similarly, so that the rows of Y correspond to that of X
Y_train = pd.melt(Y_train, id_vars='key', var_name='date', value_name='sales')
Y_test = pd.melt(Y_test, id_vars='key', var_name='date', value_name='sales')
Y_train = Y_train.sort_values(['key', 'date'])
Y_test = Y_test.sort_values(['key', 'date'])

# Separate X to training and test data; test data should have 12,824*31 = 397,544 rows
X_train = X_flat.loc[X_flat['date'].str.startswith('2017')]
X_test = X_flat.loc[X_flat['date'].str.startswith('2018')]
print(X_test.shape)
print(Y_test.shape)

# Clean 'date' columns to keep only YYYY-MM-DD part
X_train['date'] = X_train['date'].str[0:10]
X_test['date'] = X_test['date'].str[0:10]
Y_train['date'] = Y_train['date'].str[0:10]
Y_test['date'] = Y_test['date'].str[0:10]
print(Y_test.head())

# Store dataframes to csv
out_directory = './data/clean'
if not os.path.exists(out_directory):
    os.makedirs(out_directory)

X_train.to_csv('{}/nn_X_train.csv'.format(out_directory))
X_test.to_csv('{}/nn_X_test.csv'.format(out_directory))
Y_train.to_csv('{}/nn_Y_train.csv'.format(out_directory))
Y_test.to_csv('{}/nn_Y_test.csv'.format(out_directory))