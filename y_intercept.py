import argparse
import copy
import os
import random
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from time import time
import json
import pathlib
import tensorflow as tf2
import sys
from tcn import TCN, tcn_full_summary
from tensorflow.python.keras import backend as K 
K._get_available_gpus()
from tensorflow.compat.v1.keras import backend as K
import tensorflow as tf

import pandas as pd
import numpy as np

session_config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
sess = tf.compat.v1.Session(config=session_config)
K.set_session(sess)

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

'''
The path parameter
'''
# replace the path string to get access of the data
path = 'INPUT_YOUR_PATH'

task1_data = pd.read_csv(path + 'data.csv',index_col=None)
task1_data.head()

task1_data_cp = task1_data[['date', 'last']]
task1_data_vol = task1_data[['date', 'volume']]
task1_data_cp.head()

# Delete duplicate rows based on specific columns 
task1_data_cp2 = task1_data_cp.drop_duplicates(subset=['date'], keep='last')
pd.options.display.max_rows = 22000
# with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
all_dates = task1_data_cp['date'].unique()
all_stocks = task1_data['ticker'].unique()

list_all_cp = []
for stock in all_stocks:
  this_stock_df = task1_data.loc[task1_data['ticker'] == stock]
  this_stock_df.set_index('date', inplace=True)
  this_stock_df = this_stock_df.drop(['ticker', 'volume'], axis=1)
  list_all_cp.append(this_stock_df)
all_stocks_cp = pd.concat(list_all_cp, axis=1)

list_all_vol = []
for stock in all_stocks:
  this_stock_df = task1_data.loc[task1_data['ticker'] == stock]
  this_stock_df.set_index('date', inplace=True)
  this_stock_df = this_stock_df.drop(['ticker', 'last'], axis=1)
  list_all_vol.append(this_stock_df)
all_stocks_vol = pd.concat(list_all_vol, axis=1)

list_name = [str(i) for i in range(len(all_stocks_vol.columns))]
all_stocks_vol.columns = list_name
all_stocks_cp.columns = list_name

'''
Market data
'''
cp = all_stocks_cp
adv = all_stocks_vol

'''
Preprocess the returns
'''
returns = cp.div(cp.shift()) -1
returns = returns.shift(-1)
try:
  returns.iloc[abs(returns) > 1] = 0.0
except:
  returns.head
returns.head

'''
Preprocess invalid stocks
'''
start_date = '2016-01-01'
end_date = '2019-12-31'
test_start_date = '2020-01-01'
test_end_date = '2020-12-31'

train_valid_cp = cp.loc[start_date:end_date]
train_valid_adv = adv.loc[start_date:end_date]
train_valid_returns = returns.loc[start_date:end_date]

test_cp = cp.loc[test_start_date:test_end_date]
test_adv = adv.loc[test_start_date:test_end_date]
test_returns = returns.loc[test_start_date:test_end_date]

# get the filtered stocks with too few samples in the considered period
filtered_stocks = train_valid_cp.count() / len(train_valid_cp['0']) > 0.1
filtered_train_valid_cp = train_valid_cp.loc[:, np.asarray(filtered_stocks)]
assert len(filtered_train_valid_cp.index) == len(train_valid_cp.index)
filtered_train_valid_adv = train_valid_adv.loc[:, np.asarray(filtered_stocks)]
assert len(filtered_train_valid_adv.index) == len(train_valid_cp.index)
filtered_train_valid_returns = train_valid_returns.loc[:, np.asarray(filtered_stocks)]
assert len(filtered_train_valid_returns.index) == len(train_valid_cp.index)

filtered_test_cp = test_cp.loc[:, np.asarray(filtered_stocks)]
filtered_test_adv = test_adv.loc[:, np.asarray(filtered_stocks)]
filtered_test_returns = test_returns.loc[:, np.asarray(filtered_stocks)]

# get the indices of the remained stocks
filtered_stocks_index = filtered_stocks.loc[np.asarray(filtered_stocks)]
filtered_stocks_index_str = filtered_stocks_index.index.array
filtered_stocks_index = [int(i) for i in filtered_stocks_index_str]

all_stocks_features = []
all_stocks_returns = []
test_all_stocks_features = []
test_all_stocks_returns = []
for index, i in enumerate(filtered_stocks_index_str) :
  # merge the features
  current_stock_i_features = pd.concat([filtered_train_valid_cp[i], filtered_train_valid_adv[i]], axis=1)
  test_current_stock_i_features = pd.concat([filtered_test_cp[i], filtered_test_adv[i]], axis=1)

  # normalize daily stock features 
  daily_features = current_stock_i_features.iloc[:, :1]
  test_daily_features = test_current_stock_i_features.iloc[:, :1]
  # note here the daily feature min/max are used in test
  min = daily_features.copy(deep=True).min(axis=0)
  max = daily_features.copy(deep=True).max(axis=0)

  current_stock_i_features.iloc[:, :1]=(daily_features-min)/(max-min+0.0001)
  test_current_stock_i_features.iloc[:, :1]=(test_daily_features-min)/(max-min+0.0001)
  
  # fill daily features with 1234 to help generate mask matrix later
  current_stock_i_features.iloc[:, :1] = current_stock_i_features.iloc[:, :1].fillna(1234)
  test_current_stock_i_features.iloc[:, :1] = test_current_stock_i_features.iloc[:, :1].fillna(1234)
  assert ~(current_stock_i_features.iloc[:, :1].isnull().values.any())

  # normalize vol
  daily_features = current_stock_i_features.iloc[:, 1]
  test_daily_features = test_current_stock_i_features.iloc[:, 1]
  min = daily_features.copy(deep=True).min(axis=0)
  max = daily_features.copy(deep=True).max(axis=0)
  # note here the vol min/max are used in test
  current_stock_i_features.iloc[:, 1]=(daily_features-min)/(max-min+0.0001)
  test_current_stock_i_features.iloc[:, 1]=(test_daily_features-min)/(max-min+0.0001)

  # fill vol with 1234 to help generate mask matrix later
  current_stock_i_features.iloc[:, 1] = current_stock_i_features.iloc[:, 1].fillna(1234)
  test_current_stock_i_features.iloc[:, 1] = test_current_stock_i_features.iloc[:, 1].fillna(1234)
  assert ~(current_stock_i_features.iloc[:, 1].isnull().values.any())

  all_stocks_features.append(np.asarray(current_stock_i_features))
  all_stocks_returns.append(np.asarray(filtered_train_valid_returns[i]))
  test_all_stocks_features.append(np.asarray(test_current_stock_i_features))
  test_all_stocks_returns.append(np.asarray(filtered_test_returns[i]))

all_stocks_features = np.stack(all_stocks_features)
test_all_stocks_features = np.stack(test_all_stocks_features)

all_stocks_returns = np.stack(all_stocks_returns)
test_all_stocks_returns = np.stack(test_all_stocks_returns)

all_stocks_features = np.nan_to_num(all_stocks_features)
test_all_stocks_features = np.nan_to_num(test_all_stocks_features)

all_stocks_mask = np.full(all_stocks_returns.shape, 1.0, dtype=np.float32)
for i in range(all_stocks_features.shape[0]):
    for j in range(all_stocks_features.shape[1]):
        if (np.abs(all_stocks_features[i][j]) > 1).any() or (np.abs(all_stocks_features[i][j]) == 1234).any() or np.isnan(all_stocks_returns[i][j]) or np.abs(all_stocks_returns[i][j]) > 1 or np.abs(all_stocks_returns[i][j]) == 0:
            all_stocks_mask[i][j] = 0.0
            
test_all_stocks_mask = np.full(test_all_stocks_returns.shape, 1)
for i in range(test_all_stocks_features.shape[0]):
    for j in range(test_all_stocks_features.shape[1]):
        if (np.abs(test_all_stocks_features[i][j]) > 1).any() or (np.abs(test_all_stocks_features[i][j]) == 1234).any() or np.isnan(test_all_stocks_returns[i][j]) or np.abs(test_all_stocks_returns[i][j]) > 1 or np.abs(test_all_stocks_returns[i][j]) == 0:
            test_all_stocks_mask[i][j] = 0.0    

all_stocks_returns = np.nan_to_num(all_stocks_returns)
test_all_stocks_returns = np.nan_to_num(test_all_stocks_returns)

count = 0
for i in range(all_stocks_mask.shape[0]):
  if np.sum(all_stocks_mask[i,:]) == 0:
    count+=1
print('{} stocks are masked out by the mask'.format(count))

for i in range(test_all_stocks_features.shape[0]):
  for j in range(test_all_stocks_features.shape[1]):
    for k in range(test_all_stocks_features.shape[2]):
      if abs(test_all_stocks_features[i, j, k]) > 1 and abs(test_all_stocks_features[i, j, k]) != 1234:
        # print(test_all_stocks_features[i, j, k])
        continue

def correlation(mask):  
  mask = tf.cast(mask, np.float64)
  def my_loss(x, y):
    x = tf.cast(x, np.float64)
    y = tf.cast(y, np.float64)
    x = tf.multiply(x , mask)
    y = tf.multiply(y , mask)    

    mx = tf.math.reduce_mean(x)
 
    demean_x_big_cap = x - mx
    xm = tf.multiply(demean_x_big_cap, mask)

    my = tf.math.reduce_mean(y)

    demean_y_big_cap = y - my
    ym = tf.multiply(demean_y_big_cap, mask)  

    multi = tf.multiply(xm,ym)
    r_num = tf.math.reduce_mean(multi)
    r_den = tf.math.reduce_std(xm) * tf.math.reduce_std(ym)

    return - r_num / (r_den)
  return my_loss

# hyperparameters
parameters= {}
parameters['seq'] = 8
parameters['valid_num_days'] = 252
parameters['num_of_stocks'] = all_stocks_features.shape[0]
parameters['ep'] = 3

inp = tf2.keras.Input(shape=(parameters['seq'], all_stocks_features.shape[2])) # batch_size equals to # of stocks
tcn = TCN(nb_filters = 64, kernel_size=12, dilations=[1,2,4,8,16,32,64], dropout_rate=0.4, use_layer_norm=True)(inp)
out = tf2.keras.layers.Dense(1)(tcn)

inp_mask = tf2.keras.Input(shape=(1,))
m = tf2.keras.Model([inp, inp_mask], out)

# Retrieve the config
config = m.get_config()

# At loading time, register the custom objects with a `custom_object_scope`:
custom_objects = {"correlation": correlation, 'TCN': tcn}

### prepare data for TCN input ###

def prepare_data_for_TCN(train_df, train_gt_data, all_stocks_mask, parameters):
  batch_offsets = np.arange(start=0, stop=train_df.shape[1]-parameters['seq']+1, dtype=int)

  new_train_df = np.zeros([train_df.shape[0], train_df.shape[1]-parameters['seq']+1,
                        parameters['seq'], train_df.shape[2]], dtype=np.float32)

  for offset in batch_offsets:
      new_train_df[:, offset, :, :] = train_df[:, offset:offset+parameters['seq'], :]

  new_train_df = np.transpose(new_train_df, (1, 0, 2, 3))
  new_train_df = new_train_df.reshape(new_train_df.shape[0]*new_train_df.shape[1], 
      new_train_df.shape[2], new_train_df.shape[3])

  train_mask_data = all_stocks_mask[:, (parameters['seq']-1):]
  train_mask_data = np.transpose(train_mask_data, (1, 0))
  new_train_mask_data = train_mask_data.reshape(train_mask_data.shape[0]*train_mask_data.shape[1], 1)
    
  train_gt_data = train_gt_data[:, (parameters['seq']-1):]
  train_gt_data = np.transpose(train_gt_data, (1, 0))
  new_train_gt_data = train_gt_data.reshape(train_gt_data.shape[0]*train_gt_data.shape[1], 1)

  return new_train_df, new_train_gt_data, new_train_mask_data

# prepare TCN input for the train/test part
new_train_df, new_train_gt_data, new_train_mask_data = prepare_data_for_TCN(all_stocks_features, all_stocks_returns, all_stocks_mask, parameters)
new_test_df, new_test_gt_data, new_test_mask_data = prepare_data_for_TCN(test_all_stocks_features, test_all_stocks_returns, test_all_stocks_mask, parameters)

# design loss as information coefficient with adjustments to ignore market and size exposure
my_loss = correlation(inp_mask)
m.compile(optimizer='adam', loss=my_loss)

tcn_full_summary(m, expand_residual_blocks=False)

### train TCN ###

# train the model where the validation dataset size = parameter['valid_num_days'] * parameter['num_of_stocks']
history = m.fit([new_train_df[:-parameters['valid_num_days'] * parameters['num_of_stocks'], :, :], new_train_mask_data[:-parameters['valid_num_days'] * parameters['num_of_stocks'], :]], new_train_gt_data[:-parameters['valid_num_days'] * parameters['num_of_stocks'], :], batch_size=all_stocks_features.shape[0], 
    epochs=parameters['ep'], 
    shuffle=False,
    validation_data=([new_train_df[-parameters['valid_num_days'] * parameters['num_of_stocks']:, :, :], 
                      new_train_mask_data[-parameters['valid_num_days'] * parameters['num_of_stocks']:, :]], 
                    new_train_gt_data[-parameters['valid_num_days'] * parameters['num_of_stocks']:, :]))

ypreds = m.evaluate([new_test_df, new_test_mask_data], new_test_gt_data, batch_size=parameters['num_of_stocks'])

import pickle
import math
import numpy as np
import scipy.stats as sps
from sklearn.metrics import mean_squared_error, mean_absolute_error
import collections
from scipy.stats import pearsonr

def correlation_func(prediction, ground_truth, all_mask):  
    correlation = 0
    count = 0
    for i in range(prediction.shape[0]):
        x = prediction[i, :]
        y = ground_truth[i, :]
        mask = all_mask[i, :]

        x = np.multiply(x , mask)
        y = np.multiply(y , mask)    
     
        # big_cap_masked_x = np.ma.array(x, mask=big_cap)
        # small_cap_masked_x = np.ma.array(x, mask=small_cap)
        mx = np.mean(x)
        xm = x - mx
        xm = np.multiply(xm, mask)

        my = np.mean(y)
        ym = y - my
        ym = np.multiply(ym, mask)

        multi = np.multiply(xm,ym)
        r_num = np.mean(multi) 
        r_den = np.std(xm) * np.std(ym)
 
        if np.isnan(- r_num / (r_den)):
            count += 1
            correlation += 0
        else:
            correlation += - r_num / (r_den)
    if count > 5:
        correlation = -1234
    else:
        correlation = correlation / (prediction.shape[1] - count)
    if np.isnan(correlation):
        correlation = -1234            
    return correlation

def evaluate_sr(prediction, ground_truth, mask, parameters, test=False, cutoffs_valid="", cutoffs_test="", ith_round=1):
    assert ground_truth.shape == prediction.shape, 'shape mis-match'

    if test:
        num_steps = 251 - (parameters['seq'] - 1)
    else:
        num_steps = 997

    prediction = prediction.reshape(-1, parameters['num_of_stocks'])
    ground_truth = ground_truth.reshape(-1, parameters['num_of_stocks'])
    mask = mask.reshape(-1, parameters['num_of_stocks'])

    correlation = correlation_func(prediction, ground_truth, mask)
    print('Test avg correlations: ', correlation)

    return correlation

ypreds1 = m.predict([new_test_df, new_test_mask_data], batch_size=parameters['num_of_stocks'])

test_correlation = evaluate_sr(ypreds1, new_test_gt_data, new_test_mask_data, parameters, test=True)