# wagram
Projet Master Objets Connectés

## Présentation

L’objectif du projet est de proposer des solutions pour le challenge Kaggle [MeteoNet](https://www.kaggle.com/katerpillar/meteonet/tasks)

Strigi-Form a déjà intégré dans LispTick les données disponibles sur météonet : les relevés de températures, humidités, pressions… de nombreuses stations météorologiques françaises.  
Plus de détails sur ces données sont disponibles [ici](https://meteofrance.github.io/meteonet/english/data/ground-observations/).

L’objectif est de prédire des valeurs du futur en utilisant les données du passé.

## Timeseries les plus longues

En utilisant le script LispTick donné dans ticks nous avons comme serie les plus longues en fonction du champ:

### direction du vent **degré (°)**

`(timeserie @"dd" "meteonet" "11260002" 2016-09-06T09:30 2018-08-23T14:12)`

### vitesse du vent **m.s-1**

`(timeserie @"ff" "meteonet" "11260002" 2016-09-06T09:30 2018-08-23T14:12)`

### précipitation **kg.m2**

`(timeserie @"precip" "meteonet" "6083005" 2016-05-24T13:06 2018-12-31T23:54)`

### humidité **pourcentage (%)**

`(timeserie @"hu" "meteonet" "11124003" 2016-05-24T15:06 2018-06-28T07:00)`

### [point de rosée](https://fr.wikipedia.org/wiki/Point_de_ros%C3%A9e) **Kelvin (K)**

`(timeserie @"td" "meteonet" "11124003" 2016-05-24T15:06 2018-06-28T07:00)`

### température **Kelvin (K)**

`(timeserie @"t" "meteonet" "84150001" 2016-05-27T14:00 2018-12-31T23:54)`

### pression atmosphérique **Pascal (Pa)**

`(timeserie @"psl" "meteonet" "83049005" 2017-10-31T14:24 2018-12-31T23:54)`
