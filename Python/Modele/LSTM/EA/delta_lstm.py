# -*- coding: utf-8 -*-
"""
Modified on Fri Nov 20 2020
"""

import math

import numpy as np # linear algebra

import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM

import array_from_uat as uat

WEATHER_STA = 14578001
START = "2016-01-01"
STOP = "2016-01-12"
HISTORY_LAG = 3

def main():
    """lstm fit and test"""
    #Scale the all of the data to be values between 0 and 1
    scaler = MinMaxScaler(feature_range=(0, 1))
    serie = uat.get_array(uat.TEMPERATURE, WEATHER_STA, START, STOP)
    serie = serie.transpose()
    features = serie[1:] - serie[:-1]
    print(features.shape)
    scaled_data = scaler.fit_transform(features)

    training_dataset_length = math.ceil(len(features) * .75)

    train_data = scaled_data[0:training_dataset_length, : ]

    #Splitting the data
    x_train = []
    y_train = []

    for i in range(HISTORY_LAG, len(train_data)):
        x_train.append(train_data[i-HISTORY_LAG:i, 0])
        y_train.append(train_data[i, 0])

    #Convert to numpy arrays
    #x_train, y_train = np.array(x_train, np.array(y_train))
    x_train = np.array(x_train)
    y_train = np.array(y_train)

    #Reshape the data into 3-D array
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    ################# LSTM ##########################
    # Initialising the RNN
    model = Sequential()

    model.add(LSTM(
        units=64,
        return_sequences=False,
        recurrent_dropout=0.2,
        input_shape=(x_train.shape[1], 1)))

    # Adding the output layer
    # For Full connection layer we use dense
    # As the output is 1D so we use unit=1
    model.add(Dense(units=1))
    #################################################

    #compile and fit the model on 10 epochs
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, epochs=200, batch_size=16)

    #Test data set
    test_data = scaled_data[training_dataset_length - HISTORY_LAG:, : ]

    #splitting the x_test and y_test data sets
    x_test = []
    y_test = serie[training_dataset_length+1 :, : ]
    previous = serie[training_dataset_length: -1, : ]
    for i in range(HISTORY_LAG, len(test_data)):
        x_test.append(test_data[i-HISTORY_LAG:i, 0])

    #Convert x_test to a numpy array
    x_test = np.array(x_test)

    #Reshape the data into 3-D array
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    #check predicted values
    predictions = model.predict(x_test)
    #Undo scaling
    predictions = scaler.inverse_transform(predictions)
    #add back previous point
    predictions += serie[training_dataset_length:-1, :]
    #Calculate RMSE score
    print("use previous RMSE:", np.sqrt(np.mean(((previous- y_test)**2))))
    print("predicted RMSE:", np.sqrt(np.mean(((predictions- y_test)**2))))

    plt.style.use('fivethirtyeight')
    plt.figure(1)
    plt.plot(np.append(serie[0:int(len(serie) * .75)], predictions),
             color='red', label='Predictions', linewidth=1.0)
    plt.plot(serie, color='blue', label='True value', linewidth=1.0)

    plt.show()

if __name__ == "__main__":
    main()
