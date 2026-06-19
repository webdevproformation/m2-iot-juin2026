# scikit learn

- librairie centrale dans le Machine Learning en Python
- librairie qui fait partie des librairies scientifique de Python : Numpy, MatplotLib et pandas
- elle propose une grande quantité algorithme pour faire du ML
- une des grande force de scikit learn
    - API => une fois que l'on a compris 
    - le processus est facilement modifiable 
    - la documentation est complète / très bonne qualité

- couvre l'ensemble du processus de ML
    - prétraitement des données (préprocessing) 
    - entrainement du modèle (`.fit()`)
    - évaluation des performances du modèle ( `.score()` / sklearn.mesure )
    - sélection des modèles : beaucoup de modèles 
    - Dans la documentation les modèles s'appellent des estimateurs
        - <https://scikit-learn.org/stable/machine_learning_map.html>


# Machine Learning / cas d'usage de la librairie

- Classification : des données qui ne sont pas continues
    - reconnaissance d'image
    - autorisé oui / non
    - email spam oui / non
    - titanic survecu oui / non 
    - prédire ce type de valeur

- Régression : données qui sont continues
    - durée
    - température
    - prix
    - consommation d'énergie
    - prédire ce type de valeur

# Structure général de l'API 

- quelquesoit le type d'estimateur que vous allez utiliser
- votre code va avoir toujours la même organisation


```py
# LinearRegression => nuage de point qui ont la même tendance
# récupérer des données 
# X,y  = pd.read_csv()
# X,y = np.array()


model = LinearRegression( ..... ) # hyper paramètre

# apprendre => meilleurs paramètres pour votre modèle
model.fit(X,y)

# note de la performance du modèle
model.score(X,y)

# utiliser le modèle

y_predict = model.predict(notre_X)
```

```py
# KNeighborsClassifier => nuage de point qui ont la même tendance
# récupérer des données 
# X,y  = pd.read_csv()
# X,y = np.array()

model = KNeighborsClassifier( ..... ) # hyper paramètre

# apprendre => meilleurs paramètres pour votre modèle
model.fit(X,y)

# note de la performance du modèle
model.score(X,y)

# utiliser le modèle

y_predict = model.predict(notre_X)
```


```py
# SVR => nuage de point qui ont la même tendance
# récupérer des données 
# X,y  = pd.read_csv()
# X,y = np.array()

model = SVR( ..... ) # hyper paramètre

# apprendre => meilleurs paramètres pour votre modèle
model.fit(X,y)

# note de la performance du modèle
model.score(X,y)

# utiliser le modèle

y_predict = model.predict(notre_X)
```

# scikit learn en action