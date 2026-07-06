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