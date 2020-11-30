# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 10:55:02 2020

@author: atimassr
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from fbprophet import Prophet

import matplotlib.pyplot as plt
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


#df2016 = pd.read_csv(r'NW2016.csv')
#df2017 = pd.read_csv(r'NW2017.csv')
#df2018 = pd.read_csv(r'NW2018.csv')

#weather = df2016[(df2016['number_sta'] == WEATHER_STA)]
#weather = weather.append(df2017[(df2017['number_sta'] == WEATHER_STA)], ignore_index=True)
#weather = weather.append(df2018[(df2018['number_sta'] == WEATHER_STA)], ignore_index=True)
#weather['date'] = pd.to_datetime(weather['date'], format='%Y%m%d %H:%M')
#weather.set_index('date', inplace=True)
#weather['td'] = weather['td'].interpolate('linear')
#weather['precip'] = weather['precip'].interpolate('linear')
#weather['hu'] = weather['hu'].interpolate('linear')
#weather['ff'] = weather['ff'].interpolate('linear')
#weather = weather.drop(['number_sta', 'lat', 'lon', 'height_sta'], axis = 1)

#weather_test = df2016[(df2016['number_sta'] == TEST_WEATHER_STA)]
#weather_test = weather_test.append(df2017[(df2017['number_sta'] == TEST_WEATHER_STA)], ignore_index=True)
#weather_test = weather_test.append(df2018[(df2018['number_sta'] == TEST_WEATHER_STA)], ignore_index=True)
#weather_test['date'] = pd.to_datetime(weather_test['date'], format='%Y%m%d %H:%M')
#weather_test.set_index('date', inplace=True)
#weather_test['td'] = weather_test['td'].interpolate('linear')
#weather_test['precip'] = weather_test['precip'].interpolate('linear')
#weather_test['hu'] = weather_test['hu'].interpolate('linear')
#weather_test['ff'] = weather_test['ff'].interpolate('linear')
#weather_test = weather_test.drop(['number_sta', 'lat', 'lon', 'height_sta'], axis = 1)

df = pd.read_csv('example_wp_log_peyton_manning.csv')
df.head()
#print(df)

weather = pd.read_csv(r'weather.csv')

weather['date'] = pd.to_datetime(weather['date'], format='%Y%m%d %H:%M')
weather.set_index('date', inplace=True)
weather = normalize(weather, 'td', single_param=False)

weather_test = pd.read_csv(r'weather_test.csv')

weather_test['date'] = pd.to_datetime(weather_test['date'], format='%Y%m%d %H:%M')
weather_test.set_index('date', inplace=True)
weather_test = normalize(weather_test, 'hu', single_param=False)


weather_fb = weather.drop(['dd', 'ff', 'precip', 'hu', 't', 'psl'], axis = 1)
weather_fb = weather_fb.resample('720T').mean()
weather_fb = weather_fb.reset_index()

weather_fb = weather_fb.rename(columns={'date' : 'ds', 'td' : 'y'})
#weather_ds = weather.resample('720T').mean()
#weather_test_ds = weather_test.resample('720T').mean()

#weather_ds = weather_ds.fillna(method='bfill')
#weather_test_ds = weather_ds.fillna(method='bfill')

weather_ds = weather.resample('720T').mean()
weather_test_ds = weather_test.resample('720T').mean()

weather_ds = weather_ds.fillna(method='bfill')
weather_test_ds = weather_ds.fillna(method='bfill')

weather_fb_train = weather_fb[:60]
weather_fb_test = weather_fb[60:120]

#jour = 730
#periodCustom = 90

#weather_fb_train = weather_fb[:jour*2]
#weather_fb_test = weather_fb[jour*2:jour*2+periodCustom*2]

m = Prophet()
m.fit(weather_fb_train)

future = m.make_future_dataframe(periods=30)
future.tail()

forecast = m.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
fig1 = m.plot(forecast)


fig, ax = plt.subplots(figsize=(10, 10))

ax.plot(forecast['ds'], forecast['yhat'], color='purple')
ax.plot(weather_fb_train['ds'], weather_fb_train['y'], color='blue')
ax.plot(weather_fb_test['ds'], weather_fb_test['y'], color='red')

#HISTORY_LAG = 240
#FUTURE_TARGET = 120

#X_train, y_train = segment(weather_ds, "td", window = HISTORY_LAG, future = FUTURE_TARGET)
#X_train = X_train.reshape(X_train.shape[0], HISTORY_LAG, 1)
#y_train = y_train.reshape(y_train.shape[0], FUTURE_TARGET, 1)
#print("Data shape: ", X_train.shape)
#print("Tags shape: ", y_train.shape)

#model = AutoReg(X_train, lags=1)
#model_fit = model.fit()

#ytest = model_fit.predict(len(X_train), len(X_train))
#print(ytest)