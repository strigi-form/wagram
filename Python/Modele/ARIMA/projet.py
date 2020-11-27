# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 10:55:02 2020

@author: atimassr
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import lisptick
import warnings
from sklearn import preprocessing
#import tensorflow as tf
import statsmodels as st
from statsmodels.tools.validation import array_like, string_like
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.model_selection import train_test_split
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error
import math
from sklearn.preprocessing import MinMaxScaler


WEATHER_STA = 14578001
TEST_WEATHER_STA = 22005003

TEMPERATURE = '@"t"'

HOST = "uat.lisptick.org"
PORT = 12006


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
def get_array(field, station, start, stop):
    """retreive meteonet timeserie values (no timestamp) as a numpy array"""
    conn = lisptick.Socket(HOST, PORT)
    request = " ".join(["(timeserie", field, '"meteonet"', '"'+str(station)+'"', start, stop, ")"])
    array = []
    
    def inner_append(_, __, point):
        """append value to local to local array"""
        array.append(point.i)
    conn.walk_result(request, inner_append)
    return np.array([array])


#df2016 = pd.read_csv(r'NW2016.csv')
df2018 = pd.read_csv(r'D:/Anaconda/Anaconda3/Projet prediction/NW2018new.csv')
#df2018 = pd.read_csv(r'NW2018new.csv')

#df2016=get_array(TEMPERATURE, WEATHER_STA,"2016-01-01", "2016-01-12")

weather = df2018[(df2018['number_sta'] == WEATHER_STA)]
#weather = weather.append(df2016[(df2016['number_sta'] == WEATHER_STA)])
weather['date'] = pd.to_datetime(weather['date'], format='%Y%m%d %H:%M')
weather.set_index('date', inplace=True)
weather = normalize(weather, 'td', single_param=False)

weather_test = df2018[(df2018['number_sta'] == TEST_WEATHER_STA)]
weather_test['date'] = pd.to_datetime(weather_test['date'], format='%Y%m%d %H:%M')
weather_test.set_index('date', inplace=True)
weather_test = normalize(weather_test, 'hu', single_param=False)

weather_ds = weather.resample('60T').mean()
weather_test_ds = weather_test.resample('60T').mean()

weather_ds = weather_ds.fillna(method='bfill')
weather_test_ds = weather_ds.fillna(method='bfill')



HISTORY_LAG = 240
FUTURE_TARGET =120

#X_train, y_train = segment(weather_ds, "td", window = HISTORY_LAG, future = FUTURE_TARGET)
#X_train = X_train.reshape(X_train.shape[0], HISTORY_LAG, 1)
#y_train = y_train.reshape(y_train.shape[0], FUTURE_TARGET, 1)
#print("Data shape: ", X_train.shape)
#print("Tags shape: ", y_train.shape)

scaler = MinMaxScaler(feature_range=(0, 1)) 
features = np.array([weather['td'][0:15000]])
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
x_train = np.reshape(x_train , (x_train.shape[0],x_train.shape[1],1))

#plt.plot(weather, color='blue', label='Predictions', linewidth=1.0)
plt.plot(features, color='blue', label='Expected', linewidth=1.0)

#Modél arima pour prédiction des valeurs de températutre

predictions=[]

#On définit les 3 params p:les périodes prises pour autoregression model
#d:difference 'integrated' orde;q=periodes dans la moyenne mobile model

model_arima=ARIMA(features,order=(1,0,1))
model_arima_fit=model_arima.fit()
print(model_arima_fit.aic)

#predictions=model_arima_fit.forecast(steps=9)[0]
predictions=model_arima_fit.predict(start=10,end=639)

print("les prédictions sont:",predictions)

plt.plot(features)
plt.plot(predictions,color='red', label='Predictions', linewidth=1.0)
plt.legend(loc = 'best')


#multi_step_plot(X_test[20],y_test[20],prédictions[20])

#On essaye tous les combinaisons pour avoir meilleur prédiction
p=d=q=range(0,5)
pdq=list(itertools.product(p,d,q))

#Pour avoir la combinaison qui nous donne la combinaison avec le min des values
#model = ARIMA(x_train.reshape(-1).tolist(), order=(2,1,2))

#Calcul de AIC pour chaque combinaison ne permettra de choisir la meilleur
#en prenant la combinaison de faible valeur de AIC
#Pour le test il prend un peu de temps car le calcul se fait sur tous les combinaisons
#Pour cela j'ai fais cette partie sous_dessous en commentaire et laisser le choix de test

#warnings.filterwarnings('ignore')
#for param in pdq:
#    try:
#        model_arima=ARIMA(y_train, order=param) 
#        model_arima_fit=model_arima.fit()
#        print(param,model_arima_fit.aic)
#    except:
#        continue

#calcul d'erreur entre les expected et les predictions
mean_squared_error(features,predictions)
                                    
#☻model = AutoReg(X_train, lags=1)
#model_fit = model.fit()

#ytest = model_fit.predict(len(X_train), len(X_train))
#print(ytest)