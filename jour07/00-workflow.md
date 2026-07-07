# récupérer les données et transformer en tensor

from_numpy()

# et tu splits train / test

# model

```
class TOTOTO( nn.Module ):

    def __init__(self):
        super().__init__(); 

        // 
        self.layer 
        self.layer // fonction d'activation / sigmoid / 
        self.layer 

    def forward(self, X) -> torch.Tensor :
        return // fonction 

```

# deux fonctions à créer 

```py
# fonction de cost # écart entre la valeur prédite par TOTOTO et la valeur d'entrainement

# optimiser => qui permet de faire rapprocher les paramètres à la valeur d'entrainement
```

# il faut écrire un boucle train / test 

```py
# boucle for epochs

epochs=100

instance = TOTOTO()

for epoch in range(epochs):

    # entrainement
    # étapes 
    # 1 activer le require_grad
    instance.train() # activer 
    
    # 2 forward => prédiction (train)

    # 3 fonction de cout => écart

    # 4 zero => remise à zéro optimiser

    # 5 backward => retour au début 

    # 6 step => update les paramètres de ton model 
    
    instance.eval()
    # le test 
        # 2 forward => prédiction (test)
        # 3 fonction de cout => écart (test)
    # étapes
# valeur de visualisationd
```

# Eval 

- graph (matplot lib / librairie / tes propres fonction)

# savegarder ton mode entrainé 

- fichier .pt / .pth (Pickel)

model = KNNModel()
model.fit()
model.eval()