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
import argparse


PRJ_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PRJ_ROOT)
from text_query_model.text_query import search_by_text
from image_query_model.image_query import search_by_image
from mmoe_ranker.__inference__ import run_reranking

def run_candidate_generation(model_version='v1', top_k=100):
    """Step 1: Generate candidate articles using the Two-Tower model."""
    print("--- Running Step 1: Candidate Generation ---")
    candidates_df = generate_candidates(model_version=model_version, top_k=top_k, 
                                      output_path=str(HM_TWO_STEP_RECO_DIR / 'output' / 'inference_results.parquet'))
    print(f"Candidate generation complete.")
    return candidates_df

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run the H&M recommendation pipeline.")
    parser.add_argument("--customer_id", type=str, default="aa51fd04db21c0d2620a351dc5b94b704922d674b1c52a37225dd25a7a166ee0", 
                        help="A specific customer_id to rerank for (simulates online inference).")
    args = parser.parse_args()

    # Step 1: Load pre-generated candidates
    print("--- Loading pre-generated candidates ---")
    candidates_path = TWO_TOWER_DIR / 'output' / 'inference' / 'inference_results.parquet'
    if not candidates_path.exists():
        print(f"Candidates file not found at {candidates_path}.")
        print("Please run candidate generation first or ensure the file is in the correct location.")
        sys.exit(1)

    candidates = pd.read_parquet(candidates_path)
    print("Candidates loaded successfully.")

    # Step 2: Re-rank candidates for the given customer_id
    reranked_recommendations = run_reranking(candidates, customer_id_to_rerank=args.customer_id, output_path=HM_TWO_STEP_RECO_DIR / 'output' / 'reranked_recommendations.parquet')

    if reranked_recommendations.empty:
        print("No recommendations were generated. Exiting.")
        sys.exit(0)
