# classification binaire / entrain de classification multi class

- data
- model
- fonction de loss / optimiser
  loss_fn = nn.CrossEntropyLoss()
 
- train / test

   forward
  y_logits = model( X_train ) []
  y_pred = torch.softmax(y_logits, dim=1).argmax( dim=1 ) 

    [ 0.20 , 0.05 , 0.40 , 0.15  ].argmax( dim=1 ) 

- 
- eval
- sauvegarder



```py
# dépendances
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
```


```py
def accuracy_fn(y_true, y_pred):
    """Calculates accuracy between truth labels and predictions.

    Args:
        y_true (torch.Tensor): Truth labels for predictions.
        y_pred (torch.Tensor): Predictions to be compared to predictions.

    Returns:
        [torch.float]: Accuracy value between y_true and y_pred, e.g. 78.45
    """
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc


def plot_decision_boundary(model: torch.nn.Module, X: torch.Tensor, y: torch.Tensor):
    """Plots decision boundaries of model predicting on X in comparison to y.

    Source - https://madewithml.com/courses/foundations/neural-networks/ (with modifications)
    """
    # Put everything to CPU (works better with NumPy + Matplotlib)
    model.to("cpu")
    X, y = X.to("cpu"), y.to("cpu")

    # Setup prediction boundaries and grid
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 101), np.linspace(y_min, y_max, 101))

    # Make features
    X_to_pred_on = torch.from_numpy(np.column_stack((xx.ravel(), yy.ravel()))).float()

    # Make predictions
    model.eval()
    with torch.inference_mode():
        y_logits = model(X_to_pred_on)

    # Test for multi-class or binary and adjust logits to prediction labels
    if len(torch.unique(y)) > 2:
        y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)  # mutli-class
    else:
        y_pred = torch.round(torch.sigmoid(y_logits))  # binary

    # Reshape preds and plot
    y_pred = y_pred.reshape(xx.shape).detach().numpy()
    plt.contourf(xx, yy, y_pred, cmap=plt.cm.RdYlBu, alpha=0.7)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
```

```py
# jeu de données
X , y = make_blobs(
    n_samples= 1000, 
    n_features= 2 , # colonnes de X
    centers= 4 , # 4 colonnes possible 
    cluster_std=1.5,
    random_state=42
)
```

```py
# visualiser les données sur un graphique
import pandas as pd
import seaborn as sns
blob_dataframe = pd.DataFrame({
    "X1" : X[:,0],
    "X2" : X[:,1],
    "target" : y
})
sns.pairplot( blob_dataframe , hue="target" )

```

```py
# transformation en tensor et split

X = torch.from_numpy(X).type(torch.float32)
y = torch.from_numpy(y).type(torch.int64)

X_train, X_test , y_train, y_test = train_test_split(X, y , test_size=0.2 , random_state=42 )

```

```py
# model

device = "cuda" if torch.cuda.is_available() else "cpu"

class MultiClass(nn.Module):

  def __init__(self , in_feature :int , out_feature : int , hidden:int ):
    super().__init__()
    self.linear_layer_stack = nn.Sequential(
        nn.Linear(in_features=in_feature, out_features=hidden),
        nn.ReLU(),
        nn.Linear(in_features=hidden, out_features=hidden),
        nn.ReLU(),
        nn.Linear(in_features=hidden, out_features=out_feature)
    )

  def forward(self, x):
    return self.linear_layer_stack(x)


model = MultiClass(2 , 4 , 10).to(device)

model.state_dict()
```


```py
# loss et optimiser

loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(params=model.parameters(), lr=0.1)
```

```py
# entrainement et test

epochs = 200 

# seed pour le random

torch.manual_seed(42)
torch.cuda.manual_seed(42)

# rendre nos variables device agnostic

X_train, X_test =  X_train.to(device), X_test.to(device)
y_train, y_test  = y_train.to(device), y_test.to(device)

for epoch in range(epochs):
  model.train()

  # forward
  y_logits = model( X_train )
  y_pred = torch.softmax(y_logits, dim=1).argmax( dim=1 )

  # loss
  loss = loss_fn(y_logits , y_train  )
  acc = accuracy_fn( y_true=y_train , y_pred=y_pred )

  
  # zero
  optimizer.zero_grad()

  # backward
  loss.backward()

  # step
  optimizer.step()

  # test 
  model.eval()
  with torch.inference_mode():
    # forward_test
    y_logits_test = model( X_test )
    y_pred_test = torch.softmax(y_logits_test, dim=1).argmax( dim=1 )

    # loss_test
    loss_test = loss_fn(y_logits_test , y_test )
    #  loss_test = loss_fn(y_logits_test , y_test.long() )
    acc_test = accuracy_fn( y_true=y_test , y_pred=y_pred_test )

  # afficher l'évolution des calculs pendant l'exécution de la boucle 
  if epoch % 20 == 0 :
    print(f"epoch {epoch} - loss {loss:.5f} - acc {acc:.2f}% - loss_test {loss_test:.5f} - acc_test {acc_test:.2f}%")

```