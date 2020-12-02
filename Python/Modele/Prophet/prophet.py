# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 10:52:11 2020

@author: Guillaume D
"""
"""
Client of LispTick TimeSerie Streaming Server
"""

import lisptick
import pandas as pd
import numpy as np
from fbprophet import Prophet
import matplotlib.pyplot as plt

values = []
times =[]

def mean_absolute_pourcentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    #return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return np.mean((y_true - y_pred)**2)

def normalize(dataset, target, single_param=False):
    if single_param:
        dataNorm = dataset
        dataNorm[target]=((dataset[target]-dataset[target].min())/(dataset[target].max()-dataset[target].min()))
        return dataNorm
    else:
        dataNorm=((dataset-dataset.min())/(dataset.max()-dataset.min()))
#         dataNorm[target]=dataset[target]
        return dataNorm
    
def print_value(reader, uuid, value):
    global values, times, datas
    if isinstance(value, lisptick.Point):
        times.append(value.time)
        values.append(value.i)
    else:
        print(value)
    

host = "uat.lisptick.org"
port = 12006
datas = pd.DataFrame()

    
code = """
    (timeserie @"t" "meteonet" "86027001" 2016-01-01 2016-02-15)
    """
lisptick_srv = lisptick.Socket(host, port)

lisptick_srv.walk_result(code, print_value)
    
    ## Création du DataFrame + utilisation de la date comme index
datas = pd.DataFrame({'date' : times,
                          'values' : values})
    
    ## Affichage
print(datas)
    
    ## Traitement
    
datas['date'] = pd.to_datetime(datas['date'], format='%Y%m%d %H:%M')
datas = datas.rename(columns={'date' : 'ds', 'values' : 'y'})
datas.set_index('ds', inplace=True)
#datas = normalize(datas, 'values', single_param=False)
datas = datas.resample('60T').mean()
    
datas = datas.fillna(method='bfill')

errors = [0 for i in range(29)]

for i in range(1,29) :
    datas_train = datas.reset_index()[:i*24]
    datas_test = datas.reset_index()[i*24:(i+1)*24]
        
    print("On en est à : ", i)
    
    if i < 7 :
        m = Prophet(daily_seasonality=True)
    else :
        m = Prophet(daily_seasonality=True, weekly_seasonality=True)
    
    m.fit(datas_train)
        
    future = m.make_future_dataframe(periods=24, freq="H")
    future.tail()
        
    forecast = m.predict(future)
    forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
    ##fig1 = m.plot(forecast) ## Affichage gérer par Prophet
    
    if i > 10 :      
        fig, ax = plt.subplots(figsize=(10, 10))
            
        ax.plot(forecast['ds'][i*24:(i+1)*24], forecast['yhat'][i*24:(i+1)*24], color='purple')
        ##ax.plot(datas_train['ds'], datas_train['y'], color='blue')
        ax.plot(datas_test['ds'], datas_test['y'], color='red')
        plt.title(str(i) + " jours de train")
    
    errors[i] = mean_absolute_pourcentage_error(datas_test['y'], forecast['yhat'][i*24:(i+1)*24])
    print("Le taux d'erreur est de : ", round(errors[i],2) , "%")

plt.subplots()
plt.plot(range(29), errors)