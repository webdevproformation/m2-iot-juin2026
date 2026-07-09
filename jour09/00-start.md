# courageux pour le rappeler les étapes d'apprentissage d'une modele deep learn

1 récupérer de donnée / conversion de tensor (avec un device GPU / CPU) / split (train test) / 


2. modèle

```py
class Yolo(nn.Module):

    def __init__(self):
        # ici tu vas décrire ton réseau de neurone

        self.reseau = nn.Sequential(
            nn.Linear( in_features=2 , out_features=4 ),
            nn.Linear( in_features=4 , out_features=2 ),
            nn.Linear( in_features=2 , out_features=1 )
        )

    def forward(self, x):
        return self.reseau( x )
```
 
 

3. 2 fonctions loss / optimizer

4. entrainement / test (loop)

```py
epochs = 200

for epoch in range(epochs):
    # entrainement (amélioration des paramètres du modèle pour qu'ils fit des valeurs de la target)
    # 6 étapes
    model.train() # activer la descente de gradient dans les tensor

    # forward => obtenir une prédiction
    y_pred = model( data_train )
    
    # voir écart entre prediction ET le y de test
    loss = loss_fn( y_pred , y_train )

    # remise à 0 de l'optimizer
    optimizer.zero_grad()

    # backward 
    # revenir au début de notre reseau de neurone
    loss.backward()

    # step() => mettre à jour les paramètres du modèle
    optimizer.step()
    
    # test 
    model.eval()
    
    with torch.infered_mode():
        y_pred_test = model( data_test )
        # voir écart entre prediction ET le y de test
        loss_test = loss_fn( y_pred_test , y_test )

# qu'est ce qu'il se passe ????
    if epoch % 30 == 0 :
        print(f"epoch {epoch} - loss {loss} - loss_train {loss_test}")


```