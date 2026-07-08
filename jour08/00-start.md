# le workflow (etapes dans le code)

- récupérer les données / transformer en tensor / split
- model
- 2 fonctions loss / optimizer
- train / test pour améliorer les paramètres du modèle
- eval 
- sauvegarde

# model

```
class Linaire( nn.Module ):
    
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.randn(1, require_grad=True , dtype=float32  ) )
        self.b = nn.Parameter(torch.randn(1, require_grad=True , dtype=float32  ) )

    def forward(self , x):
        return self.a * x + self.b

```

```
class Linaire( nn.Module ):
    
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(in_features=1, out_features=1)

    def forward(self , x):
        return self.layer(x)

```



```
class Linaire( nn.Module ):
    
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(in_features=2, out_features=4)

    def forward(self , x):
        return self.layer(x)
```



```

input = torch.tensor([[ 1,2,3 ], [4,5,6]])
input_dim = input.shape[1]

class Linaire( nn.Module ):
    
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(in_features=input_dim, out_features=5)
        self.layer2 = nn.Linear(in_features=5, out_features=1)

    def forward(self , x):
        return self.layer2(self.layer1(x) )
```

```
class Linaire( nn.Module ):
    
    def __init__(self):
        super().__init__()
        self.suite_layers = nn.Sequential(
            nn.Linear(in_features=2, out_features=5)
            nn.Linear(in_features=5, out_features=5)
            nn.Linear(in_features=5, out_features=5)
            nn.Linear(in_features=5, out_features=1)
        )

    def forward(self , x):
        return self.suite_layers( x )
```


```py
class Linaire( nn.Module ):
    
    def __init__(self , in , out , hidden):
        super().__init__()
        self.suite_layers = nn.Sequential(
            nn.Linear(in_features=in, out_features=hidden)
            nn.Linear(in_features=hidden, out_features=hidden)
            nn.Linear(in_features=hidden, out_features=hidden)
            nn.Linear(in_features=hidden, out_features=out)
        )

    def forward(self , x):
        return self.suite_layers( x )


model_1 = Linaire( 2 , 1 , 10 )
```