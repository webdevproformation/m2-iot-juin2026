# Fonctionnalités ML avancées pour ton projet RFID

## Détection comportementale avancée (User Behavioral Profiling)

Aller plus loin que “anomalie basique”

Apprendre les patterns individuels d’accès :

- horaires habituels
- fréquence d’accès
- séquence de déplacements

Détecter :

- accès inhabituel (ex : 3h du matin)
- badge potentiellement volé


Astuces :

- Isolation Forest
- Autoencoder
- HDBSCAN (UEBA – User & Entity Behavior Analytics)

---

## Identification implicite de l’utilisateur (biométrie comportementale)

Même sans RFID fiable

Reconnaitre un utilisateur à partir de :

- timing
- vitesse d’accès
- séquences d’usage

Chaque utilisateur a un pattern d’utilisation unique [github.com]

Objectif :

- détection badge cloné
- sécurité avancée

---

## Modélisation des flux et prédiction de fréquentation

Modéliser le système comme une série temporelle multi-zones

Prédire :

- nombre d’entrées par heure
- zones saturées

Objectif :

- optimisation des accès
- sécurité

📌 ML :

Prophet / ARIMA
LSTM

---

## Détection de séquences anormales

Pas seulement un événement → mais une suite logique d’événements

- entrée sans sortie
- accès incohérent entre zones

Modèle à utiliser :

Hidden Markov Models (HMM)
LSTM séquentiel
graph-based anomaly detection


Bon café rdv dans 15 min @ toute suite !! rdv 14h45 