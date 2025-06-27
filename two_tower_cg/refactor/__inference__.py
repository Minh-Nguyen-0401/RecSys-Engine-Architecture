"""Run inference with the trained two-tower model.

Usage (example):
    python -m __inference__ \
        --model_dir output/models/model_v1 \
        --top_k 12

This script rebuilds the model architecture, loads the trained weights, applies
all preprocessing steps, and outputs top-k article recommendations for each
input customer.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import List

import pandas as pd
import tensorflow as tf

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
import sys

sys.path.insert(0, CUR_DIR)

from src.config import Config, Variables
from src.preprocess import preprocess, process_features, PreprocessedHmData
from src.train import build_model
from src.custom_recall import CustomRecall
from filter_candidates import apply_candidate_filters


logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s:%(name)s: %(message)s'
    )

PARENT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PARENT_DIR / "output"
MODEL_DIR = OUTPUT_DIR / "models"

def _predict_batch(model: tf.keras.Model,
                   batch: dict[str, tf.Tensor],
                   top_k: int,
                   threshold: float | None,
                   article_lookup_inverse: tf.keras.layers.StringLookup) -> List[List[str]]:
    """Return string article_id predictions for each sample in the batch."""
    logits = model(batch, training=False)  # (batch, nb_articles)

    if threshold is not None:
        # boolean mask where similarity >= threshold
        mask = logits >= threshold
        predictions = []
        for i in range(tf.shape(logits)[0]):
            idx = tf.boolean_mask(tf.range(tf.shape(logits)[1]), mask[i])
            if tf.size(idx) == 0:
                idx = tf.math.top_k(logits[i], k=top_k).indices  # fallback
            preds = article_lookup_inverse(idx).numpy().astype(str).tolist()
            predictions.append(preds)
        return predictions
    else:
        top_indices = tf.math.top_k(logits, k=top_k).indices  # (batch, k)
        flat_indices = tf.reshape(top_indices, [-1])
        flat_article_ids = article_lookup_inverse(flat_indices)
        article_ids = tf.reshape(flat_article_ids, top_indices.shape)  # (batch, k)
        return article_ids.numpy().astype(str).tolist()


def run_inference(model_version: str,
                  batch_size: int = 1024,
                  embedding_dim: int = 64,
                  top_k: int = 12,
                  threshold: float | None = None,
                  output_path: str | None = None) -> pd.DataFrame:
    """Generate top-k recommendations for each customer."""

    logger.info("Loading parquet datasets from ./output/parquet …")
    parquet_dir = OUTPUT_DIR / "parquet"
    available_parts = ["train", "val"]
    frames: list[pd.DataFrame] = []
    for part in available_parts:
        file_path = parquet_dir / f"{part}.parquet"
        if file_path.exists():
            logger.info(f"Reading {file_path}")
            if part == "train":
                train_df = pd.read_parquet(file_path)
                frames.append(train_df)
            else:
                frames.append(pd.read_parquet(file_path))
    if not frames:
        raise FileNotFoundError("No parquet datasets found in ./output/parquet")

    trans_df = pd.concat(frames, ignore_index=True)
    # Keep latest record per customer
    if "t_dat" in trans_df.columns:
        trans_df.sort_values("t_dat", inplace=True)
    latest_trans_df = trans_df.drop_duplicates("customer_id", keep="last")

    latest_trans_df.to_parquet(
        os.path.join(OUTPUT_DIR, "inference", "latest_transactions.parquet"),
        index=False
    )
    logger.info(f"Total unique customers for inference: {len(latest_trans_df)}")

    # Articles parquet
    articles_path = parquet_dir / "articles.parquet"
    if not articles_path.exists():
        raise FileNotFoundError("articles.parquet not found in ./output/parquet")
    article_df = pd.read_parquet(articles_path)

    preprocessed: PreprocessedHmData = preprocess(
        train_df=train_df,
        val_df=latest_trans_df,
        article_df=article_df,
        batch_size=batch_size,
    )

    model_dir_path = MODEL_DIR / f"model_{model_version}"
    config = Config(
        embedding_dimension=embedding_dim,
        batch_size=batch_size,
        learning_rate=0.01,
        nb_epochs=1,
        model_save_dir=model_dir_path,
        recall_k_values=[top_k],
    )

    logger.info("Re-building model architecture …")
    model = build_model(preprocessed, config)

    dummy = next(iter(preprocessed.train_ds.take(1)))
    _ = model(dummy)

    weights_path = model_dir_path / "model_weights.h5"
    logger.info(f"Loading weights from {weights_path} …")
    model.load_weights(weights_path, by_name=True, skip_mismatch=True)
    logger.info("Weights loaded.")

    model.compile(
        metrics=[CustomRecall(k=top_k, name=f"recall@{top_k}")],
    )

    logger.info("Running inference …")
    # Build reverse lookup layer for indices→string
    article_lookup_inverse = tf.keras.layers.StringLookup(
        vocabulary=preprocessed.lookups["article_id"].get_vocabulary(include_special_tokens=False),
        invert=True,
        num_oov_indices=0,
    )

    customer_lookup_inverse = tf.keras.layers.StringLookup(
        vocabulary=preprocessed.lookups["customer_id"].get_vocabulary(include_special_tokens=False),
        invert=True,
        num_oov_indices=0,
    )

    predictions: List[List[str]] = []
    customer_ids: List[str] = []

    logger.info("Applying candidate filters before inference …")
    # Chuyển DataFrame từ lookup
    candidates_df = preprocessed.article_df.copy()
    candidates_df["customer_id"] = "dummy"  # hoặc replicate mỗi khách hàng nếu cần

    # Apply filters
    filtered_articles_df = apply_candidate_filters(
        candidates_df=candidates_df,
        article_df=preprocessed.article_df,
        customer_df=preprocessed.customer_df
    )

    # Cập nhật lại vocabulary nếu cần filter bằng lookup layer (phức tạp hơn)
    logger.info(f"Filtered articles: {len(filtered_articles_df)} / {len(preprocessed.article_df)}")

    for batch in preprocessed.val_ds:
        batch_preds = _predict_batch(model, batch, top_k, threshold, article_lookup_inverse)
        predictions.extend(batch_preds)
        original_customer_ids = customer_lookup_inverse(batch["customer_id"])
        customer_ids.extend(original_customer_ids.numpy().astype(str).tolist())

    result_df = pd.DataFrame(
        {
            "customer_id": customer_ids,
            "predicted_article_ids": [" ".join(preds) for preds in predictions],
        }
    )

    if output_path:
        output_path = Path(output_path)
        if not output_path.is_absolute():
            output_path = OUTPUT_DIR / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result_df.to_parquet(output_path, index=False)
        logger.info(f"Inference results written to {output_path}")

    logger.info("Inference completed.")
    return result_df


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run two-tower model inference.")
    parser.add_argument("-mv", "--model_version", required=True, help="Model version, e.g. 'v1'")
    parser.add_argument("--top_k", type=int, default=12, help="Number of candidates per customer")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size for inference")
    parser.add_argument("--embedding_dim", type=int, default=128, help="Embedding dimension (must match training)")
    parser.add_argument("--output", default="inference/inference_results.parquet", help="Path to write predictions")
    parser.add_argument("--threshold", type=float, default=None, help="Similarity threshold for recommendations")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_inference(
        model_version=args.model_version,
        batch_size=args.batch_size,
        embedding_dim=args.embedding_dim,
        top_k=args.top_k,
        threshold=args.threshold,
        output_path=args.output,
    )
