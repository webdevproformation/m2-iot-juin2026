# exo 2

- Créez un jeu de données multiclasse en utilisant la fonction de génération de spirales  (voir le code ci-dessous).
- Construisez un modèle capable d’apprendre à partir de ces données (une combinaison de couches linéaires et non linéaires pourrait être nécessaire).
- Mettez en place une fonction de perte et un optimiseur capables de gérer des données multiclasses (extension facultative : utilisez l’optimiseur Adam au lieu de SGD ; il faudra peut-être tester différentes valeurs de taux d’apprentissage pour obtenir un résultat satisfaisant).
- Mettez en place une boucle d’entraînement et de test pour les données multiclasses et entraînez un modèle afin d’atteindre une précision de test supérieure à 95 % (vous pouvez utiliser la fonction de mesure de précision de votre choix).
- Tracez les frontières de décision sur le jeu de données « spirales » à partir des prédictions de votre modèle ; la fonction plot_decision_boundary() devrait également fonctionner pour ce jeu de données.


```py
import numpy as np
N = 100 # number of points per class
D = 2 # dimensionality
K = 3 # number of classes
X = np.zeros((N*K,D)) # data matrix (each row = single example)
y = np.zeros(N*K, dtype='uint8') # class labels
for j in range(K):
  ix = range(N*j,N*(j+1))
  r = np.linspace(0.0,1,N) # radius
  t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2 # theta
  X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
  y[ix] = j
# lets visualize the data
plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
plt.show()
```