# exo 1

- Créez un jeu de données de classification binaire à l’aide de la fonction make_moons() de Scikit-Learn.
- Pour assurer la cohérence, le jeu de données doit comporter 1000 échantillons et utiliser random_state=42.
- Convertissez les données en tenseurs PyTorch. Divisez-les en ensembles d’entraînement et de test à l’aide de train_test_split (80 % pour l’entraînement et 20 % pour le test). Construisez un modèle en créant une sous-classe de nn.Module ; ce modèle doit intégrer des fonctions d’activation non linéaires et être capable d’apprendre à partir des données créées à l’étape 1.
- N’hésitez pas à utiliser la combinaison de couches PyTorch (linéaires nnLinear et non linéaires nn.ReLu ) de votre choix.
- Configurez une fonction de perte et un optimiseur compatibles avec la classification binaire pour l’entraînement du modèle.
- Créez une boucle d’entraînement et de test pour ajuster le modèle (créé à l’étape 2) aux données (créées à l’étape 1).
- Pour mesurer la précision du modèle, vous pouvez créer votre propre fonction de précision ou utiliser celle de TorchMetrics.
- Entraînez le modèle suffisamment longtemps pour qu’il atteigne une précision supérieure à 96 %.
- La boucle d’entraînement doit afficher la progression (perte et précision sur les ensembles d’entraînement et de test) toutes les 10 époques.
- Effectuez des prédictions avec votre modèle entraîné et visualisez-les à l’aide de la fonction plot_decision_boundary() définie dans ce notebook.
