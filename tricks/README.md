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

## Minimum et Maximum 6 minutes

Minimum et maximum pour un champ si tous les points de la série sont espacés de minimum 6 minutes
[**lien uat-playground**](https://uat.lisptick.org/playground?code=(def%0A%20%20start%202016-01-01%0A%20%20stop%20%202018-12-31%0A%20%20station%20%2213028001%22%0A)%0A(defn%20lag-max%5Bfield%20station%5D%0A%20%20(vget%20(last%20(max%20(delta%20(time-as-value%20(timeserie%20field%20%22meteonet%22%20station%20start%20stop)))))))%0A%0A(defn%20lag-filter%5Bcode%20field%5D%0A%20%20(def%20lmax%20(lag-max%20field%20code))%0A%20%20(cond%0A%20%20%20%20(duration%3F%20lmax)%20(%3D%206m%20lmax)%0A%20%20%20%20false))%3Bnot%20a%20duration%2C%20probably%20empty%20serie%0A%0A(defn%20field-perimeter%5Bfield%5D%0A%20%20(keep%20(perimeter%20%22meteonet%22)%20lag-filter%20field))%0A%0A(defn%20finder%5Bf%20field%5D%0A%20%20(vget%20(map-reduce%0A%20%20%20%20(fn%5Bcode%5D%20(vget%20(last%20(f%20(timeserie%20field%20%22meteonet%22%20code%20start%20stop)))))%0A%20%20%20%20f%0A%20%20%20%20(field-perimeter%20field))))%0A%0A%5B%0A%20%20(finder%20min%20%40%22hu%22)%0A%20%20(finder%20max%20%40%22hu%22)%0A%20%20(finder%20min%20%40%22t%22)%0A%20%20(finder%20max%20%40%22t%22)%0A%5D)

```clojure
(def
  start 2016-01-01
  stop  2018-12-31
  station "13028001"
)
(defn lag-max[field station]
  (vget (last (max (delta (time-as-value (timeserie field "meteonet" station start stop)))))))

(defn lag-filter[code field]
  (def lmax (lag-max field code))
  (cond
    (duration? lmax) (= 6m lmax)
    false));not a duration, probably empty serie

(defn field-perimeter[field]
  (keep (perimeter "meteonet") lag-filter field))

(defn finder[f field]
  (vget (map-reduce
    (fn[code] (vget (last (f (timeserie field "meteonet" code start stop)))))
    f
    (field-perimeter field))))

[
  (finder min @"hu")
  (finder max @"hu")
  (finder min @"t")
  (finder max @"t")
]
```

## Série la plus longue pour un champ

Cet exemple permet de trouvé, pour un champ donné, le code station, la date de début et la date de fin de la plus longue série où tous les points sont exactement espacés de 6 minutes.  
Avec ici comme exemple la température `@"t"` [**lien uat-playground**](https://uat.lisptick.org/playground?code=(def%0A%20%20start%20%20%202016-01-01%0A%20%20stop%20%20%20%202018-12-31%0A)%0A(defn%20ts%5Bfield%20code%5D%0A%20%20(timeserie%20field%20%22meteonet%22%20code%20start%20stop))%0A(defn%20is6m%5Bfield%20code%5D%0A%20%20(sign(%3D%20(delta(time-as-value(ts%20field%20code)))%206m)))%0A(defn%20usable-size%5Bfield%20code%5D%0A%20%20(def%20lis6m%20(is6m%20field%20code))%0A%20%20(-%0A%20%20%20%20(%2B%20lis6m)%0A%20%20%20%20(keep%20(%2B%20lis6m)%20(%3D%20lis6m%200))))%0A(defn%20longer-usable%5Bfield%5D%0A%20%20(map-reduce-arg%0A%20%20%20%20(fn%5Bcode%5D%20(last(prune(max(usable-size%20field%20code)))))%0A%20%20%20%20max%0A%20%20%20%20(perimeter%20%22meteonet%22)))%0A(defn%20longer-info%5Bfield%5D%0A%20%20(def%20res%20(last%20(longer-usable%20field)))%0A%20%20(str%0A%20%20%20%20%20%20%22(timeserie%20%22%20field%20%22%20%5C%22meteonet%5C%22%20%5C%22%22%0A%20%20%20%20%20%20(first(first(vget%20res)))%20%22%5C%22%20%22%0A%20%20%20%20%20%20(-%20(tget%20res)%20(*%206m%20(last%20(vget%20res))))%20%22%20%22%0A%20%20%20%20%20%20(tget%20res)%20%22)%22))%0A(longer-info%20%40%22t%22))

```clojure
(def
  start   2016-01-01
  stop    2018-12-31
)
(defn ts[field code]
  (timeserie field "meteonet" code start stop))
(defn is6m[field code]
  (sign(= (delta(time-as-value(ts field code))) 6m)))
(defn usable-size[field code]
  (def lis6m (is6m field code))
  (-
    (+ lis6m)
    (keep (+ lis6m) (= lis6m 0))))
(defn longer-usable[field]
  (map-reduce-arg
    (fn[code] (last(prune(max(usable-size field code)))))
    max
    (perimeter "meteonet")))
(defn longer-info[field]
  (def res (last (longer-usable field)))
  (str
      "(timeserie " field " \"meteonet\" \""
      (first(first(vget res))) "\" "
      (- (tget res) (* 6m (last (vget res)))) " "
      (tget res) ")"))
(longer-info @"t")
```
