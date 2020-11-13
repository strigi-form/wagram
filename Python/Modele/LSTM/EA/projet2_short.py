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

WEATHER_STA = 14578001

weather = pd.read_csv(r'weather.csv')
weather['date'] = pd.to_datetime(weather['date'], format='%Y%m%d %H:%M')
weather.set_index('date', inplace=True)

HISTORY_LAG = 240
FUTURE_TARGET = 120

#Scale the all of the data to be values between 0 and 1 
scaler = MinMaxScaler(feature_range=(0, 1)) 
features = np.array([weather['t'][0:2880]])
features = features.transpose()
print(features.shape)
scaled_data = scaler.fit_transform(features)

training_dataset_length = math.ceil(len(features) * .75)

train_data = scaled_data[0:training_dataset_length  , : ]

#Splitting the data
x_train=[]
y_train = []

for i in range(HISTORY_LAG, len(train_data)):
    x_train.append(train_data[i-HISTORY_LAG:i,0])
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

#compile and fit the model on 10 epochs
model.compile(optimizer = 'adam', loss = 'mean_squared_error')
model.fit(x_train, y_train, epochs = 10, batch_size = 10)

#Test data set
test_data = scaled_data[training_dataset_length - HISTORY_LAG: , : ]

#splitting the x_test and y_test data sets
x_test = []
y_test =  features[training_dataset_length : , : ] 

for i in range(HISTORY_LAG,len(test_data)):
    x_test.append(test_data[i-HISTORY_LAG:i,0])
    
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
print(rmse)

Cpredictions = np.append(features[0:int(len(features) * .75)],predictions)
plt.figure(1)
plt.plot(Cpredictions, color='red', label='Predictions', linewidth=1.0)
plt.plot(features, color='blue', label='True value', linewidth=1.0)

plt.show()