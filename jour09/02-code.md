```py

```

```py

# make out our data

from sklearn.datasets import make_circles

n_samples= 1000

X, y = make_circles(n_samples, noise=0.03, random_state=42)
len(X), len(y) , X[:5],y[:5]

# X contient 2 features
# y contient 0 1 => binary

# transformer les données en Dataframe
import pandas as pd

circles = pd.DataFrame({ "X1" : X[:,0] ,
                          "X2" : X[:,1],
                          "label" : y})

# il faut convertir en tensor
import torch

X = torch.from_numpy(X).type(torch.float32)
y = torch.from_numpy(y).type(torch.float32)

X[0],y[0]
# (tensor([0.7542, 0.2315], dtype=torch.float64), tensor(1))
X[0].dtype # torch.float32


# split des données
from sklearn.model_selection import train_test_split

X_train, X_test , y_train, y_test = train_test_split(X, y , test_size=0.2 , random_state=42)

# premier modèle non linéaire
from torch import nn

class CircleModelV3(nn.Module):
  def __init__(self):
    super().__init__()
    self.layer1 = nn.Linear(in_features=2 , out_features=90)
    self.layer2 = nn.Linear(in_features=90 , out_features=90)
    self.layer3 = nn.Linear(in_features=90 , out_features=1)
    self.relu = nn.ReLU() # tous les chiffres négatif = 0 tous les chiffres positif garde leur valeur

  def forward(self, x):
    return self.layer3(self.relu(self.layer2(self.relu(self.layer1(x)))))



model_3 = CircleModelV3().to(device)

loss_fn = nn.BCEWithLogitsLoss() # have sigmoid activation function build-in
optimizer = torch.optim.SGD(params=model_3.parameters() , lr=0.1)

torch.cuda.manual_seed(42)
torch.manual_seed(42)

 # + tour d essaie !
epochs = 1000

# put agnostique code on data
X_train = X_train.to(device)
y_train = y_train.to(device)
X_test = X_test.to(device)
y_test = y_test.to(device)

for epoch in range(epochs):
  model_3.train()

  # forward
  y_logits = model_3(X_train).squeeze()
  # torch.sigmoid fonction d'activation
  # raw logits -> prediction probabilites -> prediction labels
  y_pred = torch.round(torch.sigmoid(y_logits))

  # loss
  # attention y_logits et y_pred
  # loss_fn contient BCEWithLogitsLoss() expect raw logits
  train_loss = loss_fn(y_logits , y_train)

  acc = accuracy(y_true = y_train, y_pred = y_pred)

  # optimizer
  # reset every epcoch
  optimizer.zero_grad()

  # back propagration
  train_loss.backward()

  # optimize step (gradient descent)
  optimizer.step()

  # evaluation mode
  # test loop
  model_3.eval()
  with torch.inference_mode():
    y_logits_test = model_3(X_test).squeeze()
    y_pred_test = torch.round(torch.sigmoid(y_logits_test))

    test_loss = loss_fn(y_logits_test , y_test)
    acc_test = accuracy(y_true = y_test, y_pred = y_pred_test)

  # afficher ce qu'il se passe

  if epoch % 100 == 0 :
    print(f"epoch {epoch} -- train_loss : {train_loss:.5f} -- acc: {acc:.2f}% -- test_loss : {test_loss:.5f} -- test_acc : {acc_test:.2f}%  ")
```