import os
import csv
from PIL import Image
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# === Configuration ===
input_dir = "moities_bgallv"
output_dir = "traitements_csv"
os.makedirs(output_dir, exist_ok=True)

# === Modèle multimodal LLaMA 3.2 Vision ===
model_name = "unsloth/Llama-3.2-11B-Vision-Instruct"

class ImageInference:
    def __init__(self, model_name=model_name):
        self.llm = LLM(model=model_name, 
                       max_model_len=4096, 
                       max_num_seqs=1, 
                       tensor_parallel_size=1,
                       enforce_eager=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def generate_image_output(self, image: Image.Image) -> str:
        messages = [{
            'role': 'user',
            'content': (
                "<|image|>\n"
                "1. Dans cette image, on a des bouts de phrases en latin sur la gauche et des bouts de phrase en français sur la droite qui sont la traduction de chaque ;\n"
                "2. on veut récupérer sous la forme d'un tableau chaque bout de phrase en latin et leur traduction en français;\n"
                "3. produis toi-même ce tableau."
            )
        }]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        sampling_params = SamplingParams(max_tokens=2048, temperature=0.2)

        outputs = self.llm.generate({
            "prompt": prompt,
            "multi_modal_data": {"image": image},
        }, sampling_params=sampling_params)

        return outputs[0].outputs[0].text if outputs else "No output generated."

def parse_table_to_csv(text_output):
    """
    Convertit une sortie texte de tableau markdown en liste de lignes CSV.
    Ignore les lignes de séparation avec des `---`.
    """
    lines = text_output.strip().split('\n')
    rows = []
    for line in lines:
        line = line.strip()
        if line.startswith("|") and line.endswith("|"):
            parts = [cell.strip() for cell in line[1:-1].split("|")]
            if not all(p.startswith("---") for p in parts):  # Ignore ligne de séparation
                rows.append(parts)
    return rows

# === Exécution ===
inference_engine = ImageInference()

for filename in os.listdir(input_dir):
    if filename.lower().endswith(".jpg"):
        image_path = os.path.join(input_dir, filename)
        image = Image.open(image_path)

        result_text = inference_engine.generate_image_output(image)
        csv_rows = parse_table_to_csv(result_text)

        if csv_rows:
            output_filename = os.path.splitext(filename)[0] + ".csv"
            output_path = os.path.join(output_dir, output_filename)
            with open(output_path, "w", encoding="utf-8", newline='') as f:
                writer = csv.writer(f)
                writer.writerows(csv_rows)
            print(f"✅ Tableau sauvegardé : {filename} → {output_filename}")
        else:
            print(f"⚠️ Aucun tableau trouvé dans : {filename}")
