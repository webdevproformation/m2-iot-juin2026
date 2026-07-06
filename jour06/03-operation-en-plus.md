# opération supplémentaire sur les Tensor

## Agregation

```
.min()
.max()
.mean()
.argmax() 
```

# Opération qui change la shape de vos tensor

- reshape() changer les dimensions d'un tensor
- view() copie d'une valeur / copie de son adresse mémmoire (comme en javascript lorsque l'on crée une copie d'un tableau)
- stack() empiler horizontalement / verticalement
- squeeze() enlever une dimension 1 à un tensor
- unsqueeze() ajouter une dimension 1 à un tensor
- permute() changer les dimensions d'un tensor


# code device agnostic

```py
# view copie d'une tensor en partageant son adresse mémoire
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"

print(device)

Y = torch.tensor([1,2,3]).to(device)
print(Y.device)

Y = Y.cpu() # changer à la volé le device qui gère la variable
# code device agnostic 

print(Y.device)

Z = torch.tensor([1,2,3] , device=device) # Torch not compiled with CUDA enabled
print(Z.device)
# Y , Z 
```

# correction

```py
A = torch.zeros((2,2))
B = torch.ones((2,2))

ligne1 = torch.hstack((A,B))
ligne2 = torch.hstack(( B, A ))

resultat = torch.vstack(( ligne1 , ligne2 ))
resultat
```


## correction


```py
# correction 
import torch
A = torch.zeros(2,2)
B = torch.ones(2,2)

haut = torch.hstack((A,B))

ligne_3_un = torch.ones_like(torch.zeros(4)).view(1,4)
ligne_4_zero = torch.zeros_like(torch.ones(4)).view(1,4)
haut , ligne_3_un, ligne_4_zero

resultat = torch.vstack( ( haut , ligne_3_un , ligne_4_zero ) )

resultat
```

## Correction

```py
# créer les block de base
import torch
A = torch.ones((2,2))
B = torch.zeros((2,2))

ligne_1 = torch.hstack( (A,B,A) )

ligne_2 = torch.hstack( (B,A,B) )

derniere_ligne = torch.ones_like(torch.zeros(6)).view(1,6)

# résultat final 
resultat = torch.vstack( (ligne_1 ,  ligne_2 , derniere_ligne) )
resultat
```

# Enoncé

Exercice : 

On te donne le tableau NumPy suivant :

```py
import numpy as np

data = np.array([
    [2, 4, 6],
    [8, 10, 12]
])
```
Objectif
Obtenir le tenseur PyTorch suivant :

```py
tensor([[ 2,  4,  6],
        [ 8, 10, 12],
        [ 1,  1,  1]])
```

Contraintes

- Utiliser torch.from_numpy()
- Utiliser torch.ones_like()
- Utiliser torch.vstack()
- Utiliser view() pour que la ligne de 1 ait la bonne forme