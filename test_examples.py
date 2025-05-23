import os
import pandas as pd
from sentence_transformers import (
    SentenceTransformer,
)
from datetime import datetime
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def evaluate_cross_lingual_retrieval(model, test_df, k_values=[1, 5, 10]):

    print("Démarrage de l'évaluation par récupération bilingue...")

    print("Encodage des phrases latines...")
    source_embeddings = model.encode(test_df['Latin'].tolist(), show_progress_bar=True,
                                     batch_size=32, convert_to_numpy=True)

    print("Encodage des phrases françaises...")
    target_embeddings = model.encode(test_df['Français'].tolist(), show_progress_bar=True,
                                     batch_size=32, convert_to_numpy=True)

    print("Calcul des similarités cosinus...")
    similarities = cosine_similarity(source_embeddings, target_embeddings)

    results = {}

    print("\nRésultats de l'évaluation:")
    for k in k_values:
        hits = 0
        for i in range(len(similarities)):
            top_indices = np.argsort(similarities[i])[::-1][:k]
            if i in top_indices:
                hits += 1

        accuracy = hits / len(similarities)
        results[f'top_{k}_accuracy'] = accuracy
        print(f"Top-{k} Accuracy: {accuracy:.4f}")

    diagonal_similarities = np.diagonal(similarities)
    mean_similarity = np.mean(diagonal_similarities)
    results['mean_diagonal_similarity'] = mean_similarity
    print(f"Similarité moyenne des paires correctes: {mean_similarity:.4f}")

    reciprocal_ranks = []
    for i in range(len(similarities)):
        ranks = np.argsort(np.argsort(-similarities[i]))
        reciprocal_ranks.append(1.0 / (ranks[i] + 1))

    mrr = np.mean(reciprocal_ranks)
    results['mean_reciprocal_rank'] = mrr
    print(f"Mean Reciprocal Rank (MRR): {mrr:.4f}")

    return results



def show_retrieval_examples(model, test_df, num_examples=5):

    sample_indices = np.random.choice(len(test_df), min(num_examples, len(test_df)), replace=False)

    target_sentences = test_df['Français'].tolist()
    target_embeddings = model.encode(target_sentences, show_progress_bar=True,
                                     batch_size=32, convert_to_numpy=True)

    print("\nExemples de recherche de traduction:")
    for idx in sample_indices:
        latin_sentence = test_df.iloc[idx]['Latin']
        correct_french = test_df.iloc[idx]['Français']

        query_embedding = model.encode([latin_sentence], convert_to_numpy=True)[0]

        similarities = cosine_similarity([query_embedding], target_embeddings)[0]

        top3_indices = np.argsort(-similarities)[:3]

        print(f"\nPhrase latine: {latin_sentence}")
        print(f"Traduction correcte: {correct_french}")
        print("Top 3 des traductions candidates:")
        for i, top_idx in enumerate(top3_indices):
            candidate = test_df.iloc[top_idx]['Français']
            sim_score = similarities[top_idx]
            correct_mark = " ✓" if top_idx == idx else ""
            print(f"  {i+1}. ({sim_score:.4f}) {candidate}{correct_mark}")



best_model = SentenceTransformer("model_finetuned/checkpoint-200")

test_df = pd.read_csv("test.csv", sep=',', encoding='utf-8')
test_df['Latin'] = test_df['Latin'].fillna('')
test_df['Français'] = test_df['Français'].fillna('')

print("\n" + "="*50)
print("ÉVALUATION DU MODÈLE")
print("="*50)

evaluation_results = evaluate_cross_lingual_retrieval(best_model, test_df)

show_retrieval_examples(best_model, test_df, num_examples=3)