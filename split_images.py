import os
from PIL import Image

# === Configuration ===
input_dir = "bgallv"      # Dossier contenant les images d'entrée
output_dir = "moities_bgallv"      # Dossier pour enregistrer les moitiés droites
os.makedirs(output_dir, exist_ok=True)

# === Traitement des images ===
for filename in os.listdir(input_dir):
    if filename.lower().endswith(".jpg"):
        image_path = os.path.join(input_dir, filename)
        try:
            with Image.open(image_path) as img:
                width, height = img.size
                mid_x = width // 2
                # Définir la boîte de découpe pour la moitié droite
                box = (mid_x, 0, width, height)
                right_half = img.crop(box)
                # Enregistrer l'image découpée
                output_path = os.path.join(output_dir, filename)
                right_half.save(output_path)
                print(f"✅ {filename} traité → {output_path}")
        except Exception as e:
            print(f"❌ Erreur lors du traitement de {filename} : {e}")
