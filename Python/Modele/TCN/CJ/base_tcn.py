# -*- coding: utf-8 -*-
"""
Create by cedric.joulain@gmail.com on Fri Nov 20 2020
"""

import math

import numpy as np # linear algebra

import matplotlib.pyplot as plt


from sklearn.preprocessing import MinMaxScaler
from keras.models import Input, Model
from keras.layers import Dense
from tcn import TCN

import array_from_uat as uat

WEATHER_STA = 14578001
START = "2016-01-01"
STOP = "2016-01-12"
HISTORY_LAG = 3
#predict serie or delta
DELTA = True

def main():
    """TCN fit and test"""
    #Scale the all of the data to be values between 0 and 1
    scaler = MinMaxScaler(feature_range=(0, 1))
    serie = uat.get_array(uat.TEMPERATURE, WEATHER_STA, START, STOP)
    serie = serie.transpose()
    if DELTA:
        features = serie[1:] - serie[:-1]
    else:
        features = serie

    print(features.shape)
    scaled_data = scaler.fit_transform(features)

    training_dataset_length = math.ceil(len(features) * .75)

    train_data = scaled_data[0:training_dataset_length, : ]

    #Splitting the data
    x_train = []
    y_train = []
    for i in range(HISTORY_LAG, len(train_data)):
        x_train.append(train_data[i-HISTORY_LAG:i])
        y_train.append(train_data[i])
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    print(x_train.shape)
    print(y_train.shape)

    i = Input(shape=(HISTORY_LAG, 1))
    m = TCN(nb_filters=64,
            kernel_size=2,
            nb_stacks=1,
            dilations=(1, 2, 4, 8, 16, 32),
            padding='causal',
            use_skip_connections=False,
            dropout_rate=0.2,
            return_sequences=False,
            activation='relu',
            kernel_initializer='he_normal',
            use_batch_norm=False,
            use_layer_norm=False)(i)

    m = Dense(1, activation='linear')(m)

    model = Model(inputs=[i], outputs=[m])

    model.compile(optimizer='adam', loss='mean_squared_error')

    model.fit(x_train, y_train, epochs=200, batch_size=16)

    #Test data set
    test_data = scaled_data[training_dataset_length - HISTORY_LAG:, : ]

    #splitting the x_test and y_test data sets
    x_test = []
    if DELTA:
        y_test = serie[training_dataset_length+1 :, : ]
        previous = serie[training_dataset_length: -1, : ]
    else:
        y_test = serie[training_dataset_length :, : ]
        previous = serie[training_dataset_length-1: -1, : ]
    for i in range(HISTORY_LAG, len(test_data)):
        x_test.append(test_data[i-HISTORY_LAG:i])

    #Convert x_test to a numpy array
    x_test = np.array(x_test)

    #check predicted values
    predict = model.predict(x_test)
    #Undo scaling
    predict = scaler.inverse_transform(predict)
    #add back previous point
    if DELTA:
        predict += serie[training_dataset_length:-1, :]

    #Calculate RMSE score
    print("use previous RMSE:", np.sqrt(np.mean(((previous- y_test)**2))))
    print("predicted RMSE:", np.sqrt(np.mean(((predict- y_test)**2))))

    plt.figure(1)
    plt.plot(predict, color='red', label='predicted', linewidth=1.0)
    plt.plot(y_test, color='blue', label='actual', linewidth=1.0)
    plt.legend(['predicted', 'actual'])
    plt.show()

if __name__ == "__main__":
    main()
