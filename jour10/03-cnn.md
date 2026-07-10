# CNN

Convolutional Neural Network : famille de modèle de Deep Learning qui est spécialisé dans la reconnaissance d'image 
Traitement des images via informatique

# processus reste le même que d'habitude

- récupérer des données
- model
- fonction de loss / optimizer
- train / test loop
- éval / présentation 


# pour la partie données => récupérer des images => transformé en tensor 

google colab 

```py
# dépendance du projet
import torch
from torch import nn

# torchvision
from torchvision import datasets # récupérer nos images
from torchvision.transforms import ToTensor

import matplotlib.pyplot as plt
```