# utiliser le dataset bitcoin

- vous devez réaliser une script pandas qui permet de déterminer si on doit acheter / ou vendre du bitcoin en 2026

- la stratégie est la suivante
    - utiliser la fonction rolling() sur les 28 derniers jours
    - déterminer le min des 28 derniers jour
    - déterminer le max des 28 derniers jour
- si le cours du jours est > max des 28 jours => acheter 
- si le cours du jours est < min des 28 jours => vendre 

- Turtle Strategy !!  