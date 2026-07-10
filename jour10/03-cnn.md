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

```py
# dataset récupérer nos données

# utiliser une librairie qui s'appelle MNIST : Modified National Institute of Standards and Technology

# ET FashionMNIST <https://github.com/zalandoresearch/fashion-mnist>
# voir les images : <https://github.com/zalandoresearch/fashion-mnist/blob/master/doc/img/fashion-mnist-sprite.png>

train_data = datasets.FashionMNIST(
    root="data", # le nom du dossier dans lequel les images doivent être téléchargée
    train=True,
    download=True,
    transform=ToTensor(),                               
    target_transform=None
)

test_data = datasets.FashionMNIST(
    root="data", # le nom du dossier dans lequel les images doivent être téléchargée
    train=False,
    download=True,
    transform=ToTensor()
)

# une fois que les images sont téléchargée / bien rangée / déjà en noir et blanc ET 
# disponible sous forme de tensor

image, label = train_data[0]
image, label
```

```py
image, label = train_data[0]
image.shape # torch.Size([1, 28, 28])

# 1 => 1 seul niveau de couleur noir et black 0 255 color_channel 1 niveau de gris grayscale
# 28 => width
# 28  => height de chaque image

# combien de données dans chaque set train / test

len( train_data ), len(test_data) # 60 000 , 10 000

class_names = train_data.classes

class_names

"""
10 classes / 10 catégories de produit dans cette base 
classification multi classe
['T-shirt/top',
 'Trouser',
 'Pullover',
 'Dress',
 'Coat',
 'Sandal',
 'Shirt',
 'Sneaker',
 'Bag',
 'Ankle boot']
"""
```