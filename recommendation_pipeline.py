import pandas as pd
import numpy as np
import tensorflow as tf
import os
import sys
from pathlib import Path

# Add project directories to the Python path
HM_TWO_STEP_RECO_DIR = Path(__file__).resolve().parent
TWO_TOWER_DIR = HM_TWO_STEP_RECO_DIR / 'two_tower_cg' / 'refactor'
MMOE_RANKER_DIR = HM_TWO_STEP_RECO_DIR / 'mmoe_ranker'

sys.path.append(str(TWO_TOWER_DIR))
sys.path.append(str(MMOE_RANKER_DIR))

# It's better to import the necessary modules after adding to path
from two_tower_cg.refactor.__inference__ import run_inference as generate_candidates
from mmoe_ranker.load_data import load_data as mmoe_load_data
from mmoe_ranker.preprocess import build_lookups, build_normalization_layers, PreprocessedHmData
from mmoe_ranker.features import Features
from mmoe_ranker.basic_ranker import BasicRanker
from two_tower_cg.refactor.img_feature_extraction import get_image_paths, SwinModel, SwinConfig, AutoImageProcessor
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from tqdm import tqdm
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin
from torchvision import transforms as v2


# Define the same encoder class used for generating the embeddings
class ImageEncoder(nn.Module, PyTorchModelHubMixin):
    def __init__(self, encoder_config):
        super(ImageEncoder, self).__init__()
        self.swin = SwinModel(config=encoder_config)
        self.embedding_layer = nn.Linear(encoder_config.hidden_size, 128)

    def forward(self, image_tensor):
        features = self.swin(image_tensor).pooler_output
        embeddings = self.embedding_layer(features)
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings


def run_candidate_generation(model_version='v1', top_k=100):
    """Step 1: Generate candidate articles using the Two-Tower model."""
    print("--- Running Step 1: Candidate Generation ---")
    candidates_df = generate_candidates(model_version=model_version, top_k=top_k, 
                                      output_path=str(HM_TWO_STEP_RECO_DIR / 'output' / 'inference_results.parquet'))
    print(f"Candidate generation complete.")
    return candidates_df

def run_reranking(candidates_df, customer_id_to_rerank: str | None = None):
    """Step 2: Re-rank the candidates using the MMOE model."""
    print("\n--- Running Step 2: Re-ranking ---")

    # 1. Load data and build components to reconstruct the model
    print("Loading data for model reconstruction...")
    hm_data = mmoe_load_data()
    lookups = build_lookups(hm_data.train_df, hm_data.article_df)
    normalization_layers = build_normalization_layers(hm_data)

    dummy_preprocessed = PreprocessedHmData(None, 0, None, 0, lookups, normalization_layers)

    # 2. Rebuild model and load weights
    print("Rebuilding MMOE model and loading weights...")
    mmoe_model = BasicRanker(dummy_preprocessed)
    # Build the model by calling it on a dummy batch
    dummy_input = {f: l(tf.constant([''])) for f, l in lookups.items()}
    dummy_input.update({f: tf.constant([0.0]) for f in normalization_layers.keys()})
    mmoe_model(dummy_input)
    
    model_path = MMOE_RANKER_DIR / 'output' / 'parquet' / 'model'
    mmoe_model.load_weights(str(model_path))
    print("Model rebuilt and weights loaded successfully.")

    # If a specific customer is requested, filter the candidates
    if customer_id_to_rerank:
        print(f"Filtering candidates for customer_id: {customer_id_to_rerank}")
        candidates_df = candidates_df[candidates_df['customer_id'] == customer_id_to_rerank]
        if candidates_df.empty:
            print(f"Customer {customer_id_to_rerank} not found in candidates file.")
            return pd.DataFrame()

    # 3. Process each customer's candidates one by one to avoid memory issues
    all_reranked_results = []
    print(f"Starting re-ranking for {len(candidates_df)} customers...")
    
    for _, row in tqdm(candidates_df.iterrows(), total=candidates_df.shape[0]):
        customer_id = row['customer_id']
        article_ids = row['predicted_article_ids'].split(' ')
        
        customer_candidates_df = pd.DataFrame({'customer_id': customer_id, 'article_id': article_ids})
        
        # Merge with features
        rerank_data = customer_candidates_df.merge(hm_data.customer_df, on='customer_id', how='left')
        rerank_data = rerank_data.merge(hm_data.article_df, on='article_id', how='left')
        rerank_data = rerank_data.merge(hm_data.engineered_customer_features, on='customer_id', how='left')
        rerank_data = rerank_data.merge(hm_data.engineered_article_features, on='article_id', how='left')
        rerank_data.fillna(0, inplace=True)
        
        # Create model inputs
        model_inputs = {}
        for col in Features.ALL_CATEG_FEATURES:
            if col in rerank_data.columns:
                model_inputs[col] = lookups[col](tf.constant(rerank_data[col].astype(str).tolist()))
        for col in Features.ALL_CONTI_FEATURES + hm_data.engineered_columns:
             if col in rerank_data.columns:
                model_inputs[col] = tf.constant(rerank_data[col].values, dtype=tf.float32)

        # Get predictions
        scores = mmoe_model.predict(model_inputs, verbose=0)
        rerank_data['score'] = scores
        
        # Sort and aggregate results
        rerank_data = rerank_data.sort_values(by='score', ascending=False)
        reranked_ids = ' '.join(rerank_data['article_id'].tolist())
        
        all_reranked_results.append({'customer_id': customer_id, 'predicted_article_ids': reranked_ids})

    reranked_df = pd.DataFrame(all_reranked_results)
    
    output_path = HM_TWO_STEP_RECO_DIR / 'output' / 'reranked_recommendations.parquet'
    os.makedirs(HM_TWO_STEP_RECO_DIR / 'output', exist_ok=True)
    reranked_df.to_parquet(output_path, index=False)
    print(f"Re-ranking complete. Results saved to {output_path}")
    return reranked_df

def search_by_image(image_path, reranked_df, image_embeddings_df):
    """Step 3: Search for visually similar articles based on an input image."""
    print("\n--- Running Step 3: Image Search ---")

    # 1. Replicate the exact image embedding logic from the feature extraction script
    print(f"Extracting embedding for image: {image_path}")
    device = torch.device('cpu')
    ckpt = "yainage90/fashion-image-feature-extractor"
    encoder_config = SwinConfig.from_pretrained(ckpt)
    image_processor = AutoImageProcessor.from_pretrained(ckpt)

    encoder = ImageEncoder(encoder_config).from_pretrained(ckpt).to(device)
    encoder.eval()  # Set to evaluation mode

    # Define the same image transformations
    transform = v2.Compose([
        v2.Resize((encoder_config.image_size, encoder_config.image_size)),
        v2.ToTensor(),
        v2.Normalize(mean=image_processor.image_mean, std=image_processor.image_std),
    ])

    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        image_embedding = encoder(image_tensor).cpu().numpy()

    # 2. Filter embeddings for the articles in the user's feed
    user_articles = reranked_df['predicted_article_ids'].iloc[0].split(' ')
    feed_embeddings = image_embeddings_df[image_embeddings_df['article_id'].isin(user_articles)]

    # 3. Calculate cosine similarity
    print("Calculating cosine similarity...")
    article_embeddings = feed_embeddings.drop('article_id', axis=1).values
    similarities = cosine_similarity(image_embedding, article_embeddings).flatten()

    feed_embeddings['similarity'] = similarities

    # 4. Sort articles by similarity
    sorted_articles = feed_embeddings.sort_values(by='similarity', ascending=False)

    final_recommendations = sorted_articles[['article_id', 'similarity']]
    print("Image search complete.")
    return final_recommendations

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run the H&M recommendation pipeline.")
    parser.add_argument("--customer_id", type=str, default="aa51fd04db21c0d2620a351dc5b94b704922d674b1c52a37225dd25a7a166ee0", 
                        help="A specific customer_id to rerank for (simulates online inference).")
    args = parser.parse_args()

    # Step 1: Load pre-generated candidates instead of running generation
    print("--- Loading pre-generated candidates ---")
    candidates_path = TWO_TOWER_DIR / 'output' / 'inference' / 'inference_results.parquet'
    if not candidates_path.exists():
        print(f"Candidates file not found at {candidates_path}.")
        print("Please run candidate generation first or ensure the file is in the correct location.")
        sys.exit(1)

    candidates = pd.read_parquet(candidates_path)
    print("Candidates loaded successfully.")

    # Step 2: Re-rank candidates
    reranked_recommendations = run_reranking(candidates, customer_id_to_rerank=args.customer_id)

    if reranked_recommendations.empty:
        print("No recommendations were generated. Exiting.")
        sys.exit(0)

    # Step 3: Perform image search for the first user as an example
    print("\n--- Preparing for Image Search Example ---")
    example_image_path = HM_TWO_STEP_RECO_DIR / 'data' / 'images' / '010' / '0108775015.jpg'
    image_embeddings_path = HM_TWO_STEP_RECO_DIR / 'data' / 'image_embeddings.parquet'

    if example_image_path.exists() and image_embeddings_path.exists():
        image_embeddings = pd.read_parquet(image_embeddings_path)
        # Get recommendations for the first user for the demo
        if not reranked_recommendations.empty:
            first_user_recs = reranked_recommendations.head(1)
            final_recs = search_by_image(str(example_image_path), first_user_recs, image_embeddings)
            print("\nFinal Recommendations after Image Search:")
            print(final_recs)
        else:
            print("\nNo re-ranked recommendations generated, skipping image search.")
    else:
        print(f"\nExample image or embeddings not found, skipping image search.")

    print("\nPipeline finished.")
