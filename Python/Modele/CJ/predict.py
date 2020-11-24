# -*- coding: utf-8 -*-
"""
Create by cedric.joulain@gmail.com on Fri Nov 20 2020
"""

import math

import multiprocessing as mp
import numpy as np # linear algebra

import matplotlib.pyplot as plt


from sklearn.preprocessing import MinMaxScaler
import tcn

import array_from_uat as uat

WEATHER_STA = 14578001
START = "2016-01-01"
STOP = "2016-01-12"
HISTORY_LAG = 10
#predict serie or delta
DELTA = False

def main():
    #/2 as hyper threading is probably on
    with mp.Pool(int(mp.cpu_count()/2)) as p:
        rmses = np.array(p.map(fit_tcn, [32]*8))
    print("RMSE avg:", np.mean(rmses), "min:", np.amin(rmses), "sd:", np.std(rmses))

def fit_tcn(nb_filters=64, nb_stacks=1, kernel_size=2, lag=HISTORY_LAG, verbose=0, epochs=200, batch_size=16):
    dilations=[2**i for i in range(0, int(math.log(lag)/math.log(2)))]
    model = tcn.compiled_tcn(
        1, # num_feat
        1, # num_classes
        nb_filters, kernel_size, dilations, nb_stacks, lag, regression=True)
    result = fit(model, lag=lag, verbose=verbose, epochs=epochs, batch_size=batch_size)
    print("TCN lag:", lag, "ksize:", kernel_size, "rmse:", result)
    return result

def fit(model, lag=HISTORY_LAG, verbose=1, epochs=200, batch_size=16):
    """Fit model and return RMSE on test data"""
    #Scale the all of the data to be values between 0 and 1
    scaler = MinMaxScaler(feature_range=(0, 1))
    serie = uat.get_array(uat.TEMPERATURE, WEATHER_STA, START, STOP)
    serie = serie.transpose()
    if DELTA:
        features = serie[1:] - serie[:-1]
    else:
        features = serie
    if verbose > 0:
        print(features.shape)
    scaled_data = scaler.fit_transform(features)

    training_dataset_length = math.ceil(len(features) * .75)

    train_data = scaled_data[0:training_dataset_length, : ]

    #Splitting the data
    x_train = []
    y_train = []
    for i in range(lag, len(train_data)):
        x_train.append(train_data[i-lag:i])
        y_train.append(train_data[i])
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    if verbose > 0:
        print(x_train.shape)
        print(y_train.shape)

    model.fit(x_train, y_train,
              epochs=epochs, batch_size=batch_size, verbose=verbose)

    #Test data set
    test_data = scaled_data

    #splitting the x_test and y_test data sets
    x_test = []
    if DELTA:
        y_test = serie[lag+1 :, : ]
        previous = serie[lag: -1, : ]
    else:
        y_test = serie[lag :, : ]
        previous = serie[lag-1: -1, : ]
    for i in range(lag, len(test_data)):
        x_test.append(test_data[i-lag:i])

    #Convert x_test to a numpy array
    x_test = np.array(x_test)

    #check predicted values
    predict = model.predict(x_test)
    #keep only next predicted value
    predict = predict[:,-1]
    #Undo scaling
    predict = scaler.inverse_transform(predict)
    #add back previous point
    if DELTA:
        predict += serie[lag:-1, :]

    #Calculate RMSE score
    result = rmse(predict, y_test, training_dataset_length, lag)
    if verbose == 1:
        print("use previous RMSE:",
              rmse(previous, y_test, training_dataset_length, lag))
        print("predicted RMSE:", result)
        print("previous vs predicted RMSE:",
              rmse(predict, previous, training_dataset_length, lag))

        plt.figure(1)
        plt.plot(predict, color='red', label='predicted', linewidth=1.0)
        plt.plot(y_test, color='blue', label='actual', linewidth=1.0)
        plt.axvline(x=training_dataset_length-lag)
        plt.legend(['predicted', 'actual'])
        plt.show()
    return result

def rmse(predict, ref, training_length, lag):
    """Compute Root Mean Square Error after training data"""
    return np.sqrt(np.mean((predict[training_length+lag:] - ref[training_length+lag:])**2))

if __name__ == "__main__":
    main()
