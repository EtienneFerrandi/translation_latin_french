import base64
from vllm import LLM, SamplingParams
from PIL import Image
from io import BytesIO
import os

# === Configuration ===
input_dir = "De_Bello_Gallico_Caesar"
output_dir = "traitements"
os.makedirs(output_dir, exist_ok=True)

# Initialise le modèle
model_name = "mistralai/Pixtral-12B-2409"
llm = LLM(model=model_name, max_num_seqs=1, enforce_eager=True, tensor_parallel_size=2, tokenizer_mode="mistral")

# Paramètres du prompt
prompt = """on est sur des pages en mode paysage ;
la partie de droite de chaque page contient des bouts de phrase latin sur la gauche et des bouts de phrase français sur la droite qui sont la traduction de chaque ;
on veut récupérer sous la forme d'un tableau chaque bout de phrase latine et leur traduction française """

sampling_params = SamplingParams(max_tokens=8092)

# Fonction pour encoder une image en base64
def encode_image(image_obj):
    if isinstance(image_obj, Image.Image):
        img = image_obj
    else:
        img = Image.open(image_obj)
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

# Boucle sur toutes les images JPG dans le dossier
for filename in os.listdir(input_dir):
    if filename.lower().endswith(".jpg"):
        image_path = os.path.join(input_dir, filename)
        image_base64 = encode_image(image_path)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                ]
            },
        ]

        # Génération
        outputs = llm.chat(messages, sampling_params=sampling_params)
        result_text = outputs[0].outputs[0].text

        # Enregistrement du résultat
        output_filename = os.path.splitext(filename)[0] + ".txt"
        output_path = os.path.join(output_dir, output_filename)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(result_text)

        print(f"✅ Traitement terminé : {filename} → {output_filename}")
