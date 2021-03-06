# -*- coding: utf-8 -*-
"""
Create by cedric.joulain@gmail.com on Fri Nov 20 2020
"""

import math

import multiprocessing as mp
import numpy as np # linear algebra

import matplotlib.pyplot as plt

from geneticalgorithm import geneticalgorithm as ga

from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM

import tcn

import array_from_uat as uat

WEATHER_STA = 14578001
START = "2016-01-01"
STOP = "2016-01-12"
HISTORY_LAG = 8
#predict serie or delta
DELTA = False

def main():
    """Main, first to be called"""
    fit_lstm(verbose=1)

def hist_lstm():
    """RMSE stability for LSTM"""
    #/2 as hyper threading is probably on
    with mp.Pool(int(mp.cpu_count()/2)) as pool:
        rmses = np.array(pool.map(fit_lstm, [HISTORY_LAG]*8))
    print("RMSE avg:", np.mean(rmses), "min:", np.amin(rmses), "sd:", np.std(rmses))

def genetic_lstm():
    """LSTM input/architecture genetic optimization"""
    varbound = np.array(
        [[3, 64], #histo
         [16, 64]   #units
        ])
    algorithm_parameters = {
        'max_num_iteration': None,
        'population_size':100, #default is 100
        'mutation_probability':0.1,
        'elit_ratio': 0.01,
        'crossover_probability': 0.5,
        'parents_portion': 0.3,
        'crossover_type':'uniform',
        'max_iteration_without_improv':None}
    optim = ga(
        function=rmse_lstm,
        dimension=2,
        variable_type='int',
        variable_boundaries=varbound,
        function_timeout=60*60,
        algorithm_parameters=algorithm_parameters)
    optim.run()

def rmse_lstm(arr):
    """Compte RMSE from LSTM described by a numpy array"""
    return fit_lstm(lag=int(arr[0]), units=int(arr[1]))

def fit_lstm(lag=HISTORY_LAG, units=32, verbose=0, epochs=100, batch_size=16):
    """Learn LSTM"""
    model = Sequential()
    model.add(LSTM(
        units=units,
        return_sequences=False,
        recurrent_dropout=0.2,
        input_shape=(lag, 1)))

    # Adding the output layer
    # For Full connection layer we use dense
    # As the output is 1D so we use unit=1
    model.add(Dense(units=1))
    #################################################

    #compile and fit the model on 10 epochs
    model.compile(optimizer='adam', loss='mean_squared_error')
    result = fit(model, lag=lag, verbose=verbose, epochs=epochs, batch_size=batch_size)
    print("LSTM lag:", lag, "units:", units, "rmse:", result)
    return result


def genetic_tcn():
    """TCN input/architecture genetic optimization"""
    varbound = np.array(
        [[1, 6], #histo lag 3, 4, 8, 16, 32, 64
         [4, 6],  #nb_filters 16, 32, 64
         [2, 8],  #kernel size from 2 to 8
         [1, 3] #nb_stacks
        ])
    algorithm_parameters = {
        'max_num_iteration': None,
        'population_size':100, #default is 100
        'mutation_probability':0.1,
        'elit_ratio': 0.01,
        'crossover_probability': 0.5,
        'parents_portion': 0.3,
        'crossover_type':'uniform',
        'max_iteration_without_improv':None}
    optim = ga(
        function=rmse_tcn,
        dimension=4,
        variable_type='int',
        variable_boundaries=varbound,
        function_timeout=60*60,
        algorithm_parameters=algorithm_parameters)
    optim.run()

def rmse_tcn(arr):
    """Compte RMSE from TCN described by a numpy array"""
    #histo lag 3, 4, 8, 16, 32, 64, 128, 256
    lag = 2**arr[0]
    if lag == 2:
        lag = 3
    return fit_tcn(
        lag=int(lag),
        nb_filters=int(2**arr[1]),
        kernel_size=int(arr[2]),
        nb_stacks=int(arr[3]))

def hist_tcn():
    """RMSE stability for TCN"""
    #/2 as hyper threading is probably on
    with mp.Pool(int(mp.cpu_count()/2)) as pool:
        rmses = np.array(pool.map(fit_tcn, [HISTORY_LAG]*8))
    print("RMSE avg:", np.mean(rmses), "min:", np.amin(rmses), "sd:", np.std(rmses))

def fit_tcn(lag=HISTORY_LAG, nb_filters=32, kernel_size=5, nb_stacks=1,
            verbose=0, epochs=100, batch_size=16):
    """Learn TCN"""
    #dilations can be infered from input size (lag)
    dilations = [2**i for i in range(0, int(math.log(lag)/math.log(2)))]
    model = tcn.compiled_tcn(
        1, # num_feat
        1, # num_classes
        nb_filters, kernel_size, dilations, nb_stacks, lag, regression=True)
    result = fit(model, lag=lag, verbose=verbose,
                 epochs=epochs, batch_size=batch_size, cut_predict=True)
    print("TCN lag:", lag, "units:", nb_filters, "ksize:", kernel_size,
          "stacks:", nb_stacks, "rmse:", result)
    return result

def fit(model, lag=HISTORY_LAG, verbose=1, epochs=200, batch_size=16, cut_predict=False):
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
    if cut_predict:
        predict = predict[:, -1]
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
