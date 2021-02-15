# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 14:56:16 2021

@author: Guillaume
"""

# https://datamarket.com/data/set/22ox/monthly-milk-production-pounds-per-cow-jan-62-dec-75#!ds=22ox&display=line
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense

from tcn import TCN
import array_from_uat as uat

WEATHER_STA = 14578001
START = "2016-01-01T00:00"
STOP = "2016-01-07T23:59"
START2 = "2016-01-08T00:00"
STOP2 = "2016-01-08T23:59"
HISTORY_LAG = 8
#predict serie or delta
DELTA = False

##
# It's a very naive (toy) example to show how to do time series forecasting.
# - There are no training-testing sets here. Everything is training set for simplicity.
# - There is no input/output normalization.
# - The model is simple.
##

#milk = pd.read_csv('monthly-milk-production-pounds-p.csv.txt', index_col=0, parse_dates=True)
meteo_train = uat.get_array(uat.TEMPERATURE, WEATHER_STA, START, STOP)
meteo_test = uat.get_array(uat.TEMPERATURE, WEATHER_STA, START2, STOP2)

meteo_train = meteo_train.transpose()
meteo_test = meteo_test.transpose()


#print(milk.head())

#lookback_window = 12  # months.
lookback_window = 10

#milk = milk.values  # just keep np array here for simplicity.

x, y, x_pred, y_pred = [], [], [], []
for i in range(lookback_window, len(meteo_train)):
    x.append(meteo_train[i - lookback_window:i])
    y.append(meteo_train[i])
    
for i in range(lookback_window, len(meteo_test)):
    x_pred.append(meteo_test[i - lookback_window:i])
    y_pred.append(meteo_test[i])
    
x = np.array(x)
y = np.array(y)

x_pred = np.array(x_pred)
y_pred = np.array(y_pred)


print(x.shape)
print(y.shape)
print(x_pred.shape)
print(y_pred.shape)

i = Input(shape=(lookback_window, 1))
print(i)
m = TCN(nb_filters=2)(i)
m = Dense(1, activation='linear')(m)
model = Model(inputs=[i], outputs=[m])

model.summary()

model.compile('adam', 'mae')

print('Train...')
model.fit(x, y, epochs=50, verbose=2)


# ON  TRAIN
p = model.predict(x)

fig1, ax1 = plt.subplots()

ax1.plot(p)
ax1.plot(y)
ax1.set_title('Prédiction météo (on train)')
ax1.legend(['predicted', 'actual'])
plt.show()


## FULL PREDICTION
p2 = model.predict(x_pred)

fig2, ax2 = plt.subplots()

ax2.plot(p2)
ax2.plot(y_pred)
ax2.set_title('Prédiction météo (full test)')
ax2.legend(['predicted', 'actual'])
plt.show()