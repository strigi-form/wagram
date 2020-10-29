# Examples LispTick

Quelques examples en LispTick pour analyser efficacement les données.

## Min et Max où et quand

Permet de trouver la valeur maximale et minimale dans toute la base MétéoNet LispTick.

```clojure
(def
  zero-kc -273.15  ;0 Kelvin in Celsius
  start 2016-01-01
  stop  2018-12-31
)
(defn find-max[field]
  (map-reduce-arg
    (fn[code] (max (timeserie field "meteonet" code start stop)))
    max
    (perimeter "meteonet")))
(defn find-min[field]
  (map-reduce-arg
    (fn[code] (min (timeserie field "meteonet" code start stop)))
    min
    (perimeter "meteonet")))
```

### Température

[**lien uat-playground**](https://uat.lisptick.org/playground?code=(def%0A%20%20zero-kc%20-273.15%20%20%3B0%20Kelvin%20in%20Celsius%0A%20%20start%202016-01-01%0A%20%20stop%20%202018-12-31%0A)%0A(defn%20find-max%5Bfield%5D%0A%20%20(map-reduce-arg%0A%20%20%20%20(fn%5Bcode%5D%20(max%20(timeserie%20field%20%22meteonet%22%20code%20start%20stop)))%0A%20%20%20%20max%0A%20%20%20%20(perimeter%20%22meteonet%22)))%0A(defn%20find-min%5Bfield%5D%0A%20%20(map-reduce-arg%0A%20%20%20%20(fn%5Bcode%5D%20(min%20(timeserie%20field%20%22meteonet%22%20code%20start%20stop)))%0A%20%20%20%20min%0A%20%20%20%20(perimeter%20%22meteonet%22)))%0A%5B%0A%20(find-min%20%40%22t%22)%0A%20(find-max%20%40%22t%22)%0A%5D)

Pour trouver les températures extrême nous avons:

```clojure
[
  (find-max @"t")
  (find-min @"t")
]
```

qui donne les 2 points:
| timestamp | station | valeur |
| --- |:---:|:---:|
| 2016-05-15T06:36 | "42023004" | 233.15 |
| 2018-11-06T10:30 | "30164001" | 337.55 |

Soit -40°C et +64.4°C ce qui semble très louche donc à regarder/filtrer.

### Humidité

[**lien uat-playground**](https://uat.lisptick.org/playground?code=(def%0A%20%20zero-kc%20-273.15%20%20%3B0%20Kelvin%20in%20Celsius%0A%20%20start%202016-01-01%0A%20%20stop%20%202018-12-31%0A)%0A(defn%20find-max%5Bfield%5D%0A%20%20(map-reduce-arg%0A%20%20%20%20(fn%5Bcode%5D%20(max%20(timeserie%20field%20%22meteonet%22%20code%20start%20stop)))%0A%20%20%20%20max%0A%20%20%20%20(perimeter%20%22meteonet%22)))%0A(defn%20find-min%5Bfield%5D%0A%20%20(map-reduce-arg%0A%20%20%20%20(fn%5Bcode%5D%20(min%20(timeserie%20field%20%22meteonet%22%20code%20start%20stop)))%0A%20%20%20%20min%0A%20%20%20%20(perimeter%20%22meteonet%22)))%0A%5B%0A%20(find-min%20%40%22hu%22)%0A%20(find-max%20%40%22hu%22)%0A%5D)

Pour trouver les températures extrême nous avons:

```clojure
[
  (find-max @"hu")
  (find-min @"hu")
]
```

qui donne les 2 points:
| timestamp | station | valeur |
| --- |:---:|:---:|
| 2016-03-23T21:30 | "7230002" | 0 |
| 2016-12-23T00:30 | "73079003" | 108 |

Soit 0% et 108% d'humidité, la aussi cela semble très louche donc à regarder/filtrer.
