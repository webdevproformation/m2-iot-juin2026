# classification

le faire d'arriver à faire distinguer des concepts à notre réseau de Neurone

# Classification binaire

- spam oui / non
- fruit oui / non


# Classification multi class

- fruit : orange / pomme / poire / banane ...
- plusieurs valeurs possibles

# processus

- décupérer les données / split / tensor
- model
- fonction de loss / optimiser
- loop d'entrainement / loop de test 
- evaluation 
- bonus sauvegarder le model dans un fichier dédié .pt 


# utilisation de Seaborn pour représenter des données

```py
from sklearn.datasets import load_iris

# gérer nos données
fleurs = load_iris( )

#print(fleurs.data)

fleurs = pd.DataFrame({
    'X1' : fleurs.data[:, 0] ,
    'X2' : fleurs.data[:, 1] ,
    'X3' : fleurs.data[:, 2] ,
    'X4' : fleurs.data[:, 3] ,
    'label' : fleurs.target
})

sns.pairplot(fleurs , hue="label")

# sns.violinplot(x="X1" , y="X2" , data=fleurs , hue="label")

# sns.swarmplot(x="X1" , y="X2" , data=fleurs , hue="label")

sns.boxplot(x="X1" , y="X2" , data=fleurs , hue="label")
```