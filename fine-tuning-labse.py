import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    TranslationEvaluator,
    losses
)
from sentence_transformers.evaluation import TranslationEvaluator
from sentence_transformers.readers import InputExample
import argparse
from datetime import datetime
from datasets import load_dataset, Dataset
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


train_path = 'train.csv'
test_path = 'test.csv'
output_dir = 'model_labse_finetuned'

epochs = 4
batch_size = 4
max_seq_length = 128
evaluation_steps = 50
use_cuda = True


train_df = pd.read_csv(train_path, sep=',', encoding='utf-8')
print(f"Nombre d'exemples d'entraînement: {len(train_df)}")
print("\nAperçu des données:")
train_df.head()

train_df['latin_length'] = train_df['Latin'].fillna('').apply(len)
train_df['french_length'] = train_df['Français'].fillna('').apply(len)

print(f"\nLongueur moyenne des textes latins: {train_df['latin_length'].mean():.2f} caractères")
print(f"Longueur moyenne des textes français: {train_df['french_length'].mean():.2f} caractères")
print(f"Longueur maximale des textes latins: {train_df['latin_length'].max()} caractères")
print(f"Longueur maximale des textes français: {train_df['french_length'].max()} caractères")


train_df['Latin'] = train_df['Latin'].fillna('')
train_df['Français'] = train_df['Français'].fillna('')

print(f"Données d'entraînement chargées: {len(train_df)} paires")

# Créez une liste d'InputExample
train_examples = []
for _, row in train_df.iterrows():
    # Ensure both Latin and French are present and non-empty strings
    latin_text = str(row.get('Latin', '')).strip()
    french_text = str(row.get('Français', '')).strip()

    if latin_text and french_text: # Only add if both texts are non-empty
         train_examples.append(InputExample(texts=[latin_text, french_text]))
    else:
        # Optional: print a warning or log if a pair is skipped
        print(f"Skipping row with empty text: Latin='{row.get('Latin')}', Français='{row.get('Français')}'")


# Add a check to ensure all examples have exactly two texts
for i, example in enumerate(train_examples):
    if len(example.texts) != 2:
        raise ValueError(f"InputExample at index {i} does not have exactly 2 texts: {example.texts}")

print(f"Nombre d'InputExamples valides créés: {len(train_examples)}")

# Convertir la liste d'InputExample en datasets.Dataset
# train_dataset = Dataset.from_list([{'texts': example.texts} for example in train_examples])

# Créer le dataset avec des colonnes séparées pour le latin et le français
# Cela facilite le traitement par le trainer pour les paires de phrases
train_dataset = Dataset.from_dict({
    'sentence1': [example.texts[0] for example in train_examples],
    'sentence2': [example.texts[1] for example in train_examples]
})


print(f"Dataset créé avec {len(train_dataset)} exemples et les colonnes: {train_dataset.column_names}")


args = SentenceTransformerTrainingArguments(
    num_train_epochs = epochs,
    per_device_train_batch_size = batch_size,
    eval_steps = evaluation_steps,
    output_dir = output_dir
)



model = SentenceTransformer('sentence-transformers/LaBSE')
model.max_seq_length = 128

train_loss = losses.MultipleNegativesRankingLoss(model)

output_dir = 'output/finetuned-labse'
os.makedirs(output_dir, exist_ok=True)

# evaluator = None
# if os.path.exists(dev_path):
#     dev_df, _ = load_data_as_examples(dev_path)

#     evaluator = EmbeddingSimilarityEvaluator(
#         sentences1=dev_df['greek'].tolist(),
#         sentences2=dev_df['french'].tolist(),
#         scores=[1.0] * len(dev_df),
#         name='dev-eval',
#         batch_size=batch_size,
#         show_progress_bar=True
#     )
#     print(f"Évaluateur créé avec {len(dev_df)} paires de validation")

evaluator = TranslationEvaluator(
    source_sentences=valid_df['Latin'].tolist(),
    target_sentences=valid_df['Français'].tolist(),
    source_lang="la",
    target_lang="fr",
    name='val-eval'
)


print("Début du finetuning...")
trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    loss=train_loss,
    eval_dataset=valid_dataset,
    evaluator=evaluator
)
trainer.train()

print(f"Finetuning terminé. Modèle sauvegardé dans: {output_dir}")