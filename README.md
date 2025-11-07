# translation_latin_french

-faire de l'OCR sur une édition imprimée d'une traduction de la Guerre des Gaules de Jules César avec en colonne de gauche, le latin, en colonne de droite, le français
-constitution du dataset d'entraînement avec, à chaque ligne du tableau, une phrase latine dans une colonne, sa traduction française dans l'autre (phase d'alignement)
-entraînement du modèle labse (https://huggingface.co/sentence-transformers/LaBSE) spécialisé dans l'alignement de traductions sur le corpus d'entraînement 
-test du modèle entraîné : l'objectif est de reconnaître quelle phrase française est la traduction de telle phrase latine de la Guerre des Gaules
-le modèle labse datant un peu, possibilité de réaliser un benchmark de modèles de similarité de phrases ou tout autre modèle de traduction
