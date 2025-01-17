Documentation de développement
1. src/
Rôle : Contient le code source des modèles et des outils utilisés dans le projet.

image_model/
Contient :

image_model.py : Définit le modèle de traitement des images.
__init__.py : Fichier d’initialisation du module.
Documentation : Décrit comment le modèle est conçu, quels blocs sont utilisés, et comment les constantes mathématiques sont intégrées.
text_model/
Contient :

text_model.py : Définit le modèle de traitement du texte.
__init__.py : Fichier d’initialisation du module.
Documentation : Explique les étapes de prétraitement textuel, l’architecture du modèle, et la logique derrière les fonctions de perte et de régularisation.
audio_model/
Contient :

audio_model.py : Définit le modèle de traitement de l’audio.
__init__.py : Fichier d’initialisation du module.
Documentation : Décrit comment les données audio sont transformées en spectrogrammes, comment les couches récurrentes sont configurées, et comment le modèle apprend des séquences temporelles.
video_model/
Contient :

video_model.py : Définit le modèle de traitement de la vidéo.
__init__.py : Fichier d’initialisation du module.
Documentation : Présente les couches spatio-temporelles, les normalisations spécifiques, et les mécanismes pour capter les relations entre les cadres successifs.
fused_model/
Contient :

fusion.py : Code pour combiner les représentations multimodales.
final_model.py : Définit le modèle final fusionné.
Documentation : Expose les stratégies de fusion, les fonctions d’évaluation globale, et l’utilisation des métriques pour guider l’entraînement.
utils/
Contient :

data_utils.py : Fournit des outils pour traiter les données.
constants.py : Définit les constantes mathématiques utilisées dans le projet.
evaluation.py : Implémente les méthodes d’auto-évaluation.
Documentation : Explique comment utiliser ces outils, pourquoi les constantes sont définies de cette manière, et comment les évaluations internes sont calculées.
2. train/
Rôle : Scripts pour entraîner les différents modèles.

train_image_model.py : Script d’entraînement pour le modèle image.
train_text_model.py : Script d’entraînement pour le modèle texte.
train_audio_model.py : Script d’entraînement pour le modèle audio.
train_video_model.py : Script d’entraînement pour le modèle vidéo.
train_fused_model.py : Script d’entraînement pour le modèle fusionné final.
Documentation : Explique les processus d’entraînement, les hyperparamètres utilisés, et les configurations recommandées pour chaque modèle.

3. fintuning/
Rôle : Scripts pour affiner les modèles après l’entraînement initial.

finetune_image_model.py : Ajuste le modèle image.
finetune_text_model.py : Ajuste le modèle texte.
finetune_audio_model.py : Ajuste le modèle audio.
finetune_video_model.py : Ajuste le modèle vidéo.
finetune_fused_model.py : Ajuste le modèle fusionné.
Documentation : Décrit comment les ajustements sont effectués, quels paramètres sont modifiés, et comment l’affinement contribue à améliorer les performances globales.

4. data/
Rôle : Contient toutes les données nécessaires à l’entraînement et au test.

raw/ : Données brutes provenant des sources initiales.

images/ : Images non traitées.
text/ : Données textuelles brutes.
audio/ : Fichiers audio d’origine.
video/ : Séquences vidéo non éditées.
synth/ : Données générées artificiellement.
processed/ : Données nettoyées et normalisées prêtes pour l’entraînement.

synthetic/ : Données synthétiques générées par le modèle ou par d’autres algorithmes.

Documentation : Explique le flux de données, de leur origine brute à leur transformation en ensembles prêts pour l’entraînement. Indique également comment les données synthétiques sont produites et validées.

5. final_model/
Rôle : Contient le modèle final fusionné et prêt à l’utilisation.

final_model.pth : Fichier du modèle fusionné entraîné.
init.py : Fichier d’initialisation du module.
Documentation : Décrit comment charger et utiliser ce modèle pour des prédictions, des tests, ou des intégrations dans d’autres pipelines.

6. cdc.md
Rôle : Cahier des charges du projet.

Documentation : Liste les spécifications, objectifs et fonctionnalités du projet. Il s’agit du document de référence pour s’assurer que le développement suit les attentes initiales.

