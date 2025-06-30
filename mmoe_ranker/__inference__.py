from pathlib import Path
import os
import sys
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
CUR_DIR = Path(__file__).resolve().parent
PRJ_ROOT = CUR_DIR.parent
MMOE_RANKER_DIR = CUR_DIR
sys.path.insert(0, PRJ_ROOT)

from mmoe_ranker.load_data import load_data as mmoe_load_data
from mmoe_ranker.preprocess import build_lookups, build_normalization_layers, PreprocessedHmData
from mmoe_ranker.features import Features
from mmoe_ranker.basic_ranker import BasicRanker

def run_reranking(candidates_df, customer_id_to_rerank: str | None = None, output_path = None):
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
    dummy_input = {f: l(tf.constant([''])) for f, l in lookups.items()}
    dummy_input.update({f: tf.constant([0.0]) for f in normalization_layers.keys()})
    mmoe_model(dummy_input)
    
    model_path = MMOE_RANKER_DIR / 'output' / 'parquet' / 'model'
    mmoe_model.load_weights(str(model_path))
    print("Model rebuilt and weights loaded successfully.")

    if customer_id_to_rerank:
        print(f"Filtering candidates for customer_id: {customer_id_to_rerank}")
        candidates_df = candidates_df[candidates_df['customer_id'] == customer_id_to_rerank]
        if candidates_df.empty:
            print(f"Customer {customer_id_to_rerank} not found in candidates file.")
            return pd.DataFrame()

    # 3. Process each customer's candidates one by one
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
        rerank_data = rerank_data.sort_values(by='score', ascending=False)
        reranked_ids = ' '.join(rerank_data['article_id'].tolist())
        
        all_reranked_results.append({'customer_id': customer_id, 'predicted_article_ids': reranked_ids})

    reranked_df = pd.DataFrame(all_reranked_results)
    
    os.makedirs(output_path.parent, exist_ok=True)
    reranked_df.to_parquet(output_path, index=False)
    print(f"Re-ranking complete. Results saved to {output_path}")
    return reranked_df