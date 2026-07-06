# element de base pour créer des réseaux de neuronne artificiel 

- Tensor 
- un peu comme sur NumPy voir les fondamentaux de Tensor / les opérations de base que l'on  peut utiliser

- résoudre des problèmes de régression 
- résoudre des problèmes de classication 
    - classification binaire (survecu oui / non)
    - classification multi class (fleur iris => 4 types de fleurs)

- CNN Convolutional Neural Network => les réseaux de neuronne dédiés à la comprehension / détection d'image 

---

# Tensor 

- données (excel / photos / texte / image) 
- transformer en Tensor 
- rappeler les tableaux NDArray (N Dimension Array) de NumPy 


```py
# comme pour NumPy

import torch

scalaire = torch.tensor(5)

scalaire.ndim # 0
scalaire.shape # torch.Size([]) liste vide
scalaire.item() # permettre de récupérer la valeur stockée => tensor => repasser sur une valeur classique

vecteur = torch.tensor([1,2]) 
# utilise BEAUCOUP lorsque l'on va utiliser nn.Linear()
# ndim 1
vecteur.shape  # torch.Size([2]) # target 

# MATRICE 
MATRICE = torch.tensor([ [1,2,3], [3,4, 5] ])
MATRICE.shape  # torch.Size([2 , 3])

# 2 => nombre de ligne / exemple dans ton dataset
# 3 => colonnes / caractéristiques / Features du dataset

# TENSOR tableau à 3 dimensions

TENSOR = torch.tensor([ [  [1,2] , [3,4] ] ])

# forme du tensor
TENSOR.shape  # torch.Size([1, 2 , 2])
```


```txt
EXO 

créer un tensor de dimension 3 de taille (1,2,4) contenant des chiffres entiers de votre choix

- afficher ses dimensions
- afficher son contenu
```

```txt
Énoncé

À l'aide de PyTorch :

Crée un tenseur 3D de taille (2, 5, 10).
Ce tenseur doit contenir exactement 50 zéros et 50 uns (soit 100 éléments au total).
Mélange aléatoirement les valeurs afin que les 0 et les 1 ne soient pas regroupés.

Vérifie :

- la forme du tenseur ;
- le nombre de zéros ;
- le nombre de uns.
```

```py
resultat = torch.cat([
    torch.zeros(50),
    torch.ones(50)
])
resultat = resultat[torch.randperm(resultat.size(0))]

resultat = resultat.reshape((2,5,10))

resultat
```


## Exo

```txt
Énoncé
À l'aide de PyTorch :

Crée un tenseur 3D de forme (4, 2, 5).
Ce tenseur doit contenir exactement :

10 zéros ;
30 uns.


- Mélange les valeurs aléatoirement.
- Transforme le vecteur obtenu en tenseur 3D.
- Crée ensuite un tenseur mask avec torch.zeros_like() ayant la même forme que ton tenseur.
- Dans mask, remplace les positions correspondant aux 1 du tenseur original par la valeur 9.
```
