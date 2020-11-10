# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 10:55:02 2020

@author: atimassr
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import math
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
from keras.layers import Dropout
plt.style.use('fivethirtyeight')
#import seaborn as sns
#from sklearn import preprocessing
#import tensorflow as tf
#import statsmodels as st
#from statsmodels.tsa.holtwinters import SimpleExpSmoothing
#from statsmodels.tsa.seasonal import STL
#from sklearn.model_selection import train_test_split
#from statsmodels.tsa.ar_model import AutoReg

WEATHER_STA = 14578001
TEST_WEATHER_STA = 22005003

def normalize(dataset, target, single_param=False):
    if single_param:
        dataNorm = dataset
        dataNorm[target]=((dataset[target]-dataset[target].min())/(dataset[target].max()-dataset[target].min()))
        return dataNorm
    else:
        dataNorm=((dataset-dataset.min())/(dataset.max()-dataset.min()))
#         dataNorm[target]=dataset[target]
        return dataNorm

def segment(dataset, variable, window = 5000, future = 0):
    data = []
    labels = []
    for i in range(len(dataset)):
        start_index = i
        end_index = i + window
        future_index = i + window + future
        if future_index >= len(dataset):
            break
        data.append(dataset[variable][i:end_index])
        labels.append(dataset[variable][end_index:future_index])
    return np.array(data), np.array(labels)

def create_time_steps(length):
    return list(range(-length, 0))

def multi_step_plot(history, true_future, prediction):
    plt.figure(figsize=(12, 6))
    num_in = create_time_steps(len(history))
    num_out = len(true_future)

    plt.plot(num_in, np.array(history[:, 0]), label='History')
    plt.plot(np.arange(num_out), np.array(true_future), label='True Future')
    if prediction.any():
        plt.plot(np.arange(num_out), np.array(prediction), 'ro', label='Predicted Future')
    plt.legend(loc='upper left')
    plt.show()



weather = pd.read_csv(r'weather.csv')
weather['date'] = pd.to_datetime(weather['date'], format='%Y%m%d %H:%M')
weather.set_index('date', inplace=True)
weather = normalize(weather, 'td', single_param=False)

weather_test = pd.read_csv(r'weather_test.csv')
weather_test['date'] = pd.to_datetime(weather_test['date'], format='%Y%m%d %H:%M')
weather_test.set_index('date', inplace=True)
weather_test = normalize(weather_test, 'hu', single_param=False)

weather_ds = weather.resample('60T').mean()
weather_test_ds = weather_test.resample('60T').mean()

weather_ds = weather_ds.fillna(method='bfill')
weather_test_ds = weather_ds.fillna(method='bfill')


HISTORY_LAG = 240
FUTURE_TARGET = 120

#Scale the all of the data to be values between 0 and 1 
scaler = MinMaxScaler(feature_range=(0, 1)) 
features = np.array([weather['td'][0:1000]])
features = features.transpose()
print(features.shape)
scaled_data = scaler.fit_transform(features)

training_dataset_length = math.ceil(len(features) * .75)

train_data = scaled_data[0:training_dataset_length  , : ]

#Splitting the data
x_train=[]
y_train = []

for i in range(10, len(train_data)):
    x_train.append(train_data[i-10:i,0])
    y_train.append(train_data[i,0])

#Convert to numpy arrays
#x_train, y_train = np.array(x_train, np.array(y_train))
x_train = np.array(x_train)
y_train = np.array(y_train)

#Reshape the data into 3-D array
x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))

################# LSTM ##########################
# Initialising the RNN
model = Sequential()

model.add(LSTM(units = 50, return_sequences = True, input_shape = (x_train.shape[1], 1)))
model.add(Dropout(0.2))

# Adding a second LSTM layer and Dropout layer
model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0.2))

# Adding a third LSTM layer and Dropout layer
model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0.2))

# Adding a fourth LSTM layer and and Dropout layer
model.add(LSTM(units = 50))
model.add(Dropout(0.2))

# Adding the output layer
# For Full connection layer we use dense
# As the output is 1D so we use unit=1
model.add(Dense(units = 1))
#################################################

#compile and fit the model on 30 epochs
model.compile(optimizer = 'adam', loss = 'mean_squared_error')
model.fit(x_train, y_train, epochs = 30, batch_size = 50)

#Test data set
test_data = scaled_data[training_dataset_length - 10: , : ]

#splitting the x_test and y_test data sets
x_test = []
y_test =  features[training_dataset_length : , : ] 

for i in range(10,len(test_data)):
    x_test.append(test_data[i-10:i,0])
    
#Convert x_test to a numpy array 
x_test = np.array(x_test)

#Reshape the data into 3-D array
x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1],1))

#check predicted values
predictions = model.predict(x_test) 
#Undo scaling
predictions = scaler.inverse_transform(predictions)

#Calculate RMSE score
rmse=np.sqrt(np.mean(((predictions- y_test)**2)))
rmse

Cpredictions = np.append(features[0:int(len(features) * .75)],predictions)
plt.figure(1)
plt.plot(Cpredictions, color='red', label='Predictions', linewidth=1.0)
plt.plot(features, color='blue', label='True value', linewidth=1.0)

plt.show()
#X_train, y_train = segment(weather_ds, "td", window = HISTORY_LAG, future = FUTURE_TARGET)
#X_train = X_train.reshape(X_train.shape[0], HISTORY_LAG, 1)
#y_train = y_train.reshape(y_train.shape[0], FUTURE_TARGET, 1)
#print("Data shape: ", X_train.shape)
#print("Tags shape: ", y_train.shape)

#model = AutoReg(X_train, lags=1)
#model_fit = model.fit()

#ytest = model_fit.predict(len(X_train), len(X_train))
#print(ytest)