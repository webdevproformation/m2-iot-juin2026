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

# Model

```py
# 2 model
from torch import nn 

class ModelLinaire( nn.Module ):

  def __init__(self):
    super().__init__()

    # modèle à 1 seul neurone
    self.weight =  nn.Parameter(torch.randn(1, requires_grad=True , dtype=torch.float32))
    self.bias = nn.Parameter(torch.randn(1, requires_grad=True , dtype=torch.float32))
    # ... créer les paramètres

  def forward(self , x):
    return self.weight * x + self.bias
```

# les fonctions de loss / optimizer 

```py
# 3 fonction de loss / optimizer

# En fonction du cas que vous devez chercher 
loss_fn = nn.L1Loss() 
# fonction qui va évaluer l'écart en les prédictions de notre model et 
# les valeurs d'entrainement
# idéal pour les problèmes de régression 
# utiliser MAE : Erreur Absolue Moyenne 
device = "cuda" if torch.cuda.is_available() else "cpu"
model_0 = ModelLinaire().to(device)

optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.1)
#  stochastic gradient descent 
# param => qu'est ce que l'on veut modifier => les paramètres de notre model
# lr = learning rate => l'augmentation / diminution dans l'avancée
# de la descente de gradiant 
```

# 4 boucle de entrainement / de test

```py
# 4 boucle de entrainement / de test
torch.manual_seed(42)
torch.cuda.manual_seed(42)

# permettre à notre code d'être device agnostique

device = "cuda" if torch.cuda.is_available() else "cpu"

# faire en sorte que nos variables utilisent le bon device
X_train , y_train = X_train.to(device) , y_train.to(device)
X_test , y_test = X_test.to(device) , y_test.to(device)

epochs = 100

for epoch in range(epochs):
  model_0.train()

  # forward
  y_pred = model_0(X_train)

  # loss
  loss = loss_fn(y_pred, y_train )

  # zero
  optimizer.zero_grad()

  # retour au début du reseau de neuronne
  loss.backward()

  # mis à jour des paramètres
  optimizer.step()

  model_0.eval()
  with torch.inference_mode():
    y_pred_test = model_0( X_test )
    loss_test   = loss_fn(y_pred_test , y_test  )

  # pendant la phrase de train et test on peut afficher des
  # indicateurs pour voir si on va dans le bon sens
  if epoch % 10 == 0 :
    print(f"epoch {epoch} - loss : {loss} - loss test : {loss_test}")
```

```txt
epoch 0 - loss : 1.9269698858261108 - loss test : 2.7384707927703857
epoch 10 - loss : 0.7748699188232422 - loss test : 1.3913705348968506
epoch 20 - loss : 0.23453089594841003 - loss test : 0.5960526466369629
epoch 30 - loss : 0.19346952438354492 - loss test : 0.4453505873680115
epoch 40 - loss : 0.15908153355121613 - loss test : 0.36262649297714233
epoch 50 - loss : 0.12472259998321533 - loss test : 0.28677138686180115
epoch 60 - loss : 0.0904335230588913 - loss test : 0.2040473222732544
epoch 70 - loss : 0.05604551360011101 - loss test : 0.12132315337657928
epoch 80 - loss : 0.02170329913496971 - loss test : 0.045467983931303024
epoch 90 - loss : 0.07049746066331863 - loss test : 0.07777298986911774
```


```py
model_0.state_dict()
```