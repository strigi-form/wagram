# Prophet

Voici la partie pour Prophet. Plus d'informations sur [Prophet](https://facebook.github.io/prophet/) sur le site de Facebook.

# Installation

Pour utiliser Prophet, j'utilise Anaconda et Spyder (Python 3.8).  L'utilisation de ces outils simplifie grandement l'installation de Prophet :
- Ouvrir une console Anaconda (Anaconda Prompt)
- Écrire : ```conda install -c conda-forge fbprophet.```

Une fois les dépendances et Prophet d'installer, il suffit d'ouvrir une nouvelle instance de Spyder et d'inclure  :
- ```from fbprophet import Prophet```

## Algorithme

Dans un premier temps, je récupère les données de température depuis Lisptick. Je les convertis en DataFrame pour pouvoir les utiliser par la suite.

Une fois les données récupérées et stockées, je mets à jour le format de la date pour être sûr qu'elle soit conforme pour Prophet. De plus, il est nécessaire de renommer les colonnes pour faire les prédictions. On renomme donc la colonne des dates en **ds** et la colonne des données en **y**. Par des soucis de mémoire, je réduis mes données à 1 par heure, en gardant une valeur moyenne sur cette durée. Puis j'utilise une fonction permettant de combler les possibles N/A.

Prophet se base sur des tendances pour faire ces prédictions. On retrouve 3 types de tendance : journalier, hebdomadaire et annuel. J'ai donc décidé de me baser sur les deux premières pour mes tests. J'ai pris comme hypothèse de toujours prédire les prochaines 24h mais en variant le nombre de jours dans la base de train. 

Pour les tests, j'utilise les premières données disponibles dans la BD à savoir les relevées météos de Janvier 2016. Je choisis ensuite de faire varier ma base de train entre 1 jour et 4 semaines. Pour les prédictions, tant que l'on ne dépasse pas une semaine dans la base de train, je ne fais pas intervenir la tendance hebdomadaire. Enfin, je stocke à chaque fois le taux d'erreur que j'affiche à la fin sur une courbe d'erreur. 


## Modification du code

Voici les différentes modifications possibles à faire pour varier les tests :
- Dans la requête envoyée au serveur Lisptick, on peut modifier les dates de début et de fin pour changer la période de  test. Il faut cependant veiller à toujours récupérer au moins 30 jours de données. Je prends donc généralement 1 mois de données + les 15 jours du mois suivant.
- Pour faire varier le nombre de jours dans la base de train, cela se passe au niveau de la boucle **for**. Pour cela, il suffit de modifier la borne min et max dans le **range**. Il faut cependant veiller à **ne pas dépasser 29** en borne max.
- Avec cette version de Spyder, l'affichage de graphes est dans une fenêtre à part et on ne peut pas dépasser 20 graphes. Il faut donc prendre cela en compte et, si on s'intéresse par exemple aux résultats entre 1 et 7 jours, modifier le **if** dans la boucle for en conséquence.

## Résultats

Tout d'abord, avant 7 jours dans la base de train, les prédictions ne sont pas très bonnes. Ceci est surement dû au fait que l'on ne fait pas intervenir le caractère hebdomadaire qui fournit une caractéristique supplémentaire intéressante. 

Une fois le caractère hebdomadaire prit en compte, on voit que la courbe d'erreur reste plutôt faible. De plus, le meilleur taux est souvent obtenu pour une base de train de 24 ou 25 jours. Je n'ai pour l'instant pas encore d'explications, notamment cela ne correspond pas à un nombre de semaines "rond" (3 semaines et 3/4 jours).

