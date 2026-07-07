# Etape 1

```py
# 1 récupérer des données / split / transformer en tensor (si nécessaire)

# premier exemple on va générer nos données qui vont être une ligne droite (que l'on va générer nous même)
import torch

# y = a * x + b
# le paramètre a de la fonction => weight
# le paramètre b de la fonction => bias 

weight = 0.7
bias = 0.3 

# scikit learn
# X feature
# y target
X = torch.arange(0 , 1 , 0.02).unsqueeze(dim=1)
y = weight * X + bias

# partage entre de données de entrainement / données de test 
# entrainement 80% de données / test 20% de données

train_split = int( 0.8 * len(X)  )
X_train , y_train = X[:train_split] , y[:train_split]
X_test , y_test = X[train_split:] , y[train_split:]

# visualiser nos données
import matplotlib.pyplot as plt 

plt.figure()
plt.scatter( X_train, y_train , c="b", label="valeur entrainement" )
plt.scatter( X_test, y_test , c="y", label="valeur test" )
plt.xlabel("X feature", fontsize="14")
plt.ylabel("Y target", fontsize="14")
```

