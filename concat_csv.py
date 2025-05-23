import os
import pandas as pd

# Dossier contenant les fichiers CSV
dossier = "traitements_csv"
fichiers_csv = [f for f in os.listdir(dossier) if f.endswith(".csv")]

dfs = []

for i, fichier in enumerate(fichiers_csv):
    chemin_fichier = os.path.join(dossier, fichier)
    print(f"Lecture de : {chemin_fichier}")
    try:
        # Lire le fichier CSV
        df = pd.read_csv(chemin_fichier)
        dfs.append(df)
    except Exception as e:
        print(f"❌ Erreur lors de la lecture du fichier : {chemin_fichier}")
        print(f"Détail de l'erreur : {e}")
        continue  # Passe au fichier suivant

# Concaténer les fichiers valides
if dfs:
    df_concatene = pd.concat(dfs, ignore_index=True)
    df_concatene.to_csv("csv_concatene.csv", index=False)
    print("✅ Fichier concaténé écrit dans 'csv_concatene.csv'")
else:
    print("⚠️ Aucun fichier n'a pu être concaténé.")
