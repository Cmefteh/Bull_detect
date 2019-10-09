# La détection d'événements sonores pour le suivi de l'Alose feinte du Rhône 
Un projet proposé par l'Association Migrateurs Rhône Méditerranée (MRM) basée à Arles et effectué à l'Institut de Recherche en Informatique de Toulouse (IRIT) dans le cadre de l'obtention du diplôme d'ingénieur en systèmes électroniques industrielle. 

En bref,
Il s’agit d’une tâche de classification audio binaire, permettant de déterminer si un segment audio de dix secondes provenant d'un jeu de données spécifié contient un son de bull ou non. 

Environnement de travail: 

Logiciels et bibliothèques requis: 
 + Ubuntu 16.04 or later (64-bit)
 + Python version 3
 + Tensorflow 
 + Keras
 + Librosa
 + Numpy
 
 Fonctionnement du projet: 
 
 1) 
 
 Pour exécuter ce projet sur vos données, il faut organiser tout d'abord tous vos enregistrements audios et vos fichiers d'annotations dans un même répertoire selon l'année puis l'endroit de leurs enregistrement. 
 
En fait, nos enregistrements n’ont pas tous la même longueur, or dans une première étape, nous avons besoin d’avoir des entrées de tailles similaires pour notre réseau de neurones. Nous commençons alors par exécuter le code (Data_preparation.ipynb). Ce code permet à la fois de découper tous les enregistrements en bout de 10 secondes et de déterminer leur vérités terrain dans des fichiers ".txt". 

Il suffit juste de préciser le lien du répertoire qui contient toutes les années et le modifier dans la variable "audio_dir" dans le code (Data_preparation.ipynb). Puis, modifier les deux variables: 
s_dir = ' l'endroit (path)  où  vous souhaitez enregistrer les audios découpés '. 
bull_test = ' l'endroit (path)  où  vous souhaitez enregistrer les vérités terrain '. 

2) 
 
 Aprés le découpage des audios, nous calculons maintenant le spectrogramme de chaque audio découpé et nous enregistrons nos fichiers 

 

