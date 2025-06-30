import pandas as pd
import numpy as np
import tensorflow as tf
import os
import sys
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

HM_TWO_STEP_RECO_DIR = Path(__file__).resolve().parent
TWO_TOWER_DIR = HM_TWO_STEP_RECO_DIR / 'two_tower_cg' / 'refactor'
MMOE_RANKER_DIR = HM_TWO_STEP_RECO_DIR / 'mmoe_ranker'

sys.path.append(str(TWO_TOWER_DIR))
sys.path.append(str(MMOE_RANKER_DIR))

from two_tower_cg.refactor.__inference__ import run_inference as generate_candidates
from image_query_model.image_query import *
from text_query_model.text_query import *

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
import argparse
import json

ARTICLES_DF = HM_TWO_STEP_RECO_DIR / 'data' / 'articles.csv'

def run_online_recommend(customer_id, method = 'text', query = None):
    """Run the online recommendation pipeline for a specific customer_id."""
    print(f"\n--- Running online recommendation for customer_id: {customer_id} ---")
    RERANKED_CANDIDATES = HM_TWO_STEP_RECO_DIR / 'output' / f'reranked_recommendations_{customer_id}.parquet'
    reranked_cand = pd.read_parquet(RERANKED_CANDIDATES)
    reranked_cand = reranked_cand[reranked_cand['customer_id'] == customer_id]

    map_method = {
        'text': search_by_text,
        'image': search_by_image
    }
    
    if method == "image":
        feature_path = HM_TWO_STEP_RECO_DIR / 'data' / 'image_embeddings.parquet'
        feature_df = pd.read_parquet(feature_path)
    elif method == "text":
        feature_path = HM_TWO_STEP_RECO_DIR / 'data' / 'articles.csv'
        feature_df = pd.read_csv(feature_path)
    else:
        raise ValueError("Method must be either 'text' or 'image'.")
    
    if query is not None and feature_path.exists():
        final_recs = map_method[method](query, reranked_cand, feature_df, threshold=0.2)
        final_recs["article_id"] = final_recs["article_id"].astype(str)
        print("\nFinal Recommendations:")
        print(final_recs)
        art_desc = pd.read_csv(ARTICLES_DF)[['article_id', 'detail_desc']]
        art_desc['article_id'] = art_desc['article_id'].astype(str)
        final_recs = final_recs.merge(art_desc, on='article_id', how='left')
        final_recs = final_recs[['article_id', 'detail_desc', 'similarity']]
        
        with open(HM_TWO_STEP_RECO_DIR / 'output' / f'final_rec_with_{method}.json', 'w') as f:
            json.dump(final_recs.to_dict(orient='records'), f, indent=4)
        print(f"Final recommendations saved to {HM_TWO_STEP_RECO_DIR / 'output' / f'final_rec_with_{method}.json'}")
    else:
        print(f"Feature data not found at {feature_path}. Please check the path or run the feature extraction step first.")
        return None
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run the H&M online recommendation pipeline.")
    parser.add_argument("--customer_id", type=str, default="aa51fd04db21c0d2620a351dc5b94b704922d674b1c52a37225dd25a7a166ee0", 
                        help="A specific customer_id to rerank for (simulates online inference).")
    parser.add_argument("--method", type=str, choices=['text', 'image'], default='text',
                        help="Method to use for querying: 'text' or 'image'.")
    parser.add_argument("--query", type=str, default="long trousers",
                        help="Query string for text search or image path for image search.")
    
    args = parser.parse_args()
    
    run_online_recommend(args.customer_id, method=args.method, query=args.query)
