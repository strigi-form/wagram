# conda install statsmodels.api

import warnings
import itertools
import pandas as pd
import math
import statsmodels.api as sm
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')




WEATHER_STA = 14578001

df = pd.read_csv("NW_Ground_Stations_2016.csv")
weather = df[(df['number_sta'] == WEATHER_STA)]
weather['date'] = pd.to_datetime(df['date'])
weather = weather.set_index('date')
weather = weather['2016-01-01':'2016-1-12'].resample('6min').sum()
#weather = weather.resample('1H').mean()
#weather = weather.fillna(weather['t'].bfill())
print(weather['t'])

plt.ylim(270,320)
plt.plot(weather['t'])
plt.show()
# weather['t'].plot(figsize=(20, 6))
# plt.show()

# Définir les paramètres p, d et q pour prendre toute valeur comprise entre 0 et 2
p = d = q = range(0, 2)

# Générer toutes les combinaisons de p, q et q
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]

print('Examples of parameter combinations for Seasonal ARIMA...')
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))

#utilisation de la fonction SARIMAX de statsmodels pour l'adapter au modèle ARIMA saisonnier correspondant
#j'ai utilisé SARIMAX car au début mon jeu de donnée été de sept 2017 à dec 2017 avec une itération chaque 1H où 2H en fesont 
#la moy ( mean() ) des valeurs de l'intervale de 1H qui permet d'avoir aussi de bon résultat car dans une 1h la température varie de peut. 
warnings.filterwarnings("ignore")
for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(weather['t'],
                                            order=param, #argument spécifie la (P, D, Q)
                                            seasonal_order=param_seasonal, #argument spécifie la (P, D, Q, S)
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)

            results = mod.fit()

            print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
        except:
            continue
#Nous avons identifié l'ensemble de paramètres qui produit le modèle le mieux adapté à nos données de séries chronologiques
#SARIMAX(1, 1, 1)x(1, 1, 1, 12)

#warnings.filterwarnings("ignore")        
mod = sm.tsa.statespace.SARIMAX(weather['t'],
                                order=(1, 1, 1),
                                seasonal_order=(1, 1, 1, 12),
                                enforce_stationarity=False,
                                enforce_invertibility=False)

results = mod.fit()

print(results.summary().tables[1])

#avant d'appliqué le modéle en exécuter des diagnostics de modèle.
results.plot_diagnostics(figsize=(20, 12))
plt.show()

#Validation des prévisions
pred = results.get_prediction(start=pd.to_datetime('2016-01-12'), dynamic=False)
pred_ci = pred.conf_int()

ax = weather['t']['2016-01-01':].plot(label='observed')
pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7)

ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.2)


ax.set_xlabel('Date')
ax.set_ylabel('Temperature')
plt.legend()
plt.ylim(270,320)
plt.show()


weather_forecasted = pred.predicted_mean
weather_truth = weather['t']['2016-01-01':]

# mean square error
rmse = math.sqrt(((weather_forecasted - weather_truth) ** 2).mean())
print('The root Mean Squared Error of our forecasts is {}'.format(round(rmse, 4)))
