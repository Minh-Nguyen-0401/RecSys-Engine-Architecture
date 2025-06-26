from pathlib import Path
import logging
import pandas as pd
import numpy as np
from typing import Tuple
import os

from .config import Variables
from .features import add_rolling_features

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s:%(name)s: %(message)s'
    )

# Helper paths
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"
PARQUET_DIR = OUTPUT_DIR / 'parquet'
PARQUET_DIR.mkdir(parents=True, exist_ok=True)

RAW_PARQUET_PATH = PARQUET_DIR / 'raw_transactions.parquet'
TRAIN_PARQUET_PATH = PARQUET_DIR / 'train.parquet'
VAL_PARQUET_PATH = PARQUET_DIR / 'val.parquet'
ARTICLE_PARQUET_PATH = PARQUET_DIR / 'articles.parquet'
CUSTOMER_PARQUET_PATH = PARQUET_DIR / 'customers.parquet'

IMG_EMB_PARQUET_PATH = r"D:\Study\UNIVERSITY\THIRD YEAR\Business Analytics\final assignment\hm-two-step-reco\data\image_embeddings.parquet"
IMG_EMB_PARQUET_PATH = Path(IMG_EMB_PARQUET_PATH).resolve()

# Date ranges:
TRAIN_START, TRAIN_END = '2020-06-20', '2020-08-20'
VAL_START, VAL_END = '2020-08-21', '2020-09-22'

def create_age_interval(x):
    if x <= 25:
        return '[16, 25]'
    if x <= 35:
        return '[26, 35]'
    if x <= 45:
        return '[36, 45]'
    if x <= 55:
        return '[46, 55]'
    if x <= 65:
        return '[56, 65]'
    return '[66, 99]'


def preprocess_customer_data(customer_df):
    customer_df["FN"].fillna("UNKNOWN", inplace=True)
    customer_df["Active"].fillna("UNKNOWN", inplace=True)
    customer_df["club_member_status"].fillna("UNKNOWN", inplace=True)
    customer_df["fashion_news_frequency"] = customer_df["fashion_news_frequency"].replace({"None": "NONE"})
    customer_df["fashion_news_frequency"].fillna("UNKNOWN", inplace=True)
    customer_df["age"].fillna(customer_df["age"].median(), inplace=True)
    customer_df["age_interval"] = customer_df["age"].apply(lambda x: create_age_interval(x))
    return customer_df


def _split_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    logger.info('Splitting dataset by date ranges …')
    train_mask = (df['t_dat'] >= TRAIN_START) & (df['t_dat'] <= TRAIN_END)
    val_mask = (df['t_dat'] >= VAL_START) & (df['t_dat'] <= VAL_END)
    return df[train_mask], df[val_mask]


def _load_raw_transactions() -> pd.DataFrame:
    """Load raw transactions from cached parquet or CSV source."""
    if RAW_PARQUET_PATH.exists():
        logger.info('Reading cached parquet: %s', RAW_PARQUET_PATH)
        return pd.read_parquet(RAW_PARQUET_PATH)

    logger.info('Reading raw transaction CSV files…')
    tx_df = pd.read_csv(r"D:\Study\UNIVERSITY\THIRD YEAR\Business Analytics\final assignment\hm-two-step-reco\data\transactions_train.csv",
                        usecols=['article_id', 'customer_id', 't_dat', 'price', 'sales_channel_id'])
    tx_df['article_id'] = tx_df['article_id'].astype(str)
    tx_df['customer_id'] = tx_df['customer_id'].astype(str)
    tx_df["t_dat"] = pd.to_datetime(tx_df["t_dat"])
    tx_df = tx_df[tx_df["t_dat"] >= TRAIN_START].copy()
    tx_df.to_parquet(RAW_PARQUET_PATH, index=False)
    logger.info('Cached parquet saved: %s', RAW_PARQUET_PATH)
    return tx_df


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    logger.info("Loading data...")
    """Return train, val,  article DataFrames with engineered features."""
    if TRAIN_PARQUET_PATH.exists() and VAL_PARQUET_PATH.exists():
        logger.info('Reading cached parquets: %s, %s', TRAIN_PARQUET_PATH, VAL_PARQUET_PATH)
        train_df = pd.read_parquet(TRAIN_PARQUET_PATH)
        val_df = pd.read_parquet(VAL_PARQUET_PATH)
        article_df = pd.read_parquet(ARTICLE_PARQUET_PATH)

        # ensure for reloading dfs
        train_df = train_df[(train_df["t_dat"] >= TRAIN_START) & (train_df["t_dat"] <= TRAIN_END)]
        val_df = val_df[(val_df["t_dat"] >= VAL_START) & (val_df["t_dat"] <= VAL_END)]
        logger.info('Train:%d  Val:%d', len(train_df), len(val_df))
        return train_df, val_df, article_df
    else: 
        # Articles
        if ARTICLE_PARQUET_PATH.exists():
            logger.info('Reading cached articles parquet: %s', ARTICLE_PARQUET_PATH)
            article_df = pd.read_parquet(ARTICLE_PARQUET_PATH)
        else:
            logger.info('Reading articles CSV...')
            article_df = pd.read_csv(r"D:\Study\UNIVERSITY\THIRD YEAR\Business Analytics\final assignment\hm-two-step-reco\data\articles.csv")
            for col in Variables.ARTICLE_CATEG_VARIABLES:
                article_df[col] = article_df[col].astype(str)
        
            # Image embeddings
            if IMG_EMB_PARQUET_PATH.exists():
                img_emb_df = pd.read_parquet(IMG_EMB_PARQUET_PATH)
                logger.info(f"Image embeddings loaded. Shape: {img_emb_df.shape}")
                img_emb_df['article_id'] = img_emb_df['article_id'].astype(str)
                
                article_df = article_df.merge(img_emb_df, on='article_id', how='left')
                mapped_articles = article_df[article_df["img_embd_0"].notna()].shape[0]
                nonmapped_articles = article_df[article_df["img_embd_0"].isna()].shape[0]
                logger.info(f"Mapped articles: {mapped_articles}, Non-mapped articles: {nonmapped_articles}")
            else:
                logger.warning("Image embeddings not found. Please run img_feature_extraction.py first.")
            
            article_df.to_parquet(ARTICLE_PARQUET_PATH, index=False)

        # Customers (for categorical joins)
        if CUSTOMER_PARQUET_PATH.exists():
            logger.info('Reading cached customers parquet: %s', CUSTOMER_PARQUET_PATH)
            customer_df = pd.read_parquet(CUSTOMER_PARQUET_PATH)
        else:
            logger.info('Reading customers CSV...')
            customer_df = pd.read_csv(r"D:\Study\UNIVERSITY\THIRD YEAR\Business Analytics\final assignment\hm-two-step-reco\data\customers.csv")
            customer_df = preprocess_customer_data(customer_df)
            customer_df[Variables.CUSTOMER_CATEG_VARIABLES] = customer_df[Variables.CUSTOMER_CATEG_VARIABLES].astype(str)
            customer_df.to_parquet(CUSTOMER_PARQUET_PATH, index=False)

        # Transactions
        transactions_df = _load_raw_transactions()
        transactions_df = transactions_df[['article_id', 'customer_id', 't_dat', 'price']]

        cust_meta_cols = [c for c in Variables.CUSTOMER_CATEG_VARIABLES if c != 'customer_id']
        art_meta_cols  = [c for c in Variables.ARTICLE_CATEG_VARIABLES + Variables.IMG_EMB_VARIABLES
                        if c != 'article_id']

        cust_idx = customer_df.set_index('customer_id')[cust_meta_cols]
        art_idx  = article_df.set_index('article_id')[art_meta_cols]

        transactions_df = (
            transactions_df
            .join(cust_idx, on='customer_id', how='left')
            .join(art_idx,  on='article_id',  how='left')
        )
        
        del customer_df, cust_idx, art_idx
        
        numeric_cols = [c for c in transactions_df.columns
                        if pd.api.types.is_numeric_dtype(transactions_df[c])]
        for col in numeric_cols:
            transactions_df[col] = transactions_df[col].astype(np.float32)
            logger.info(f"Downcasted {col} to float32.")

        # Rolling stats Calculation
        base_cols = [
            'article_id', 'customer_id', 't_dat', 'price'
        ]
        rolling_df = add_rolling_features(transactions_df[base_cols].copy())
        rolling_feature_cols = [c for c in rolling_df.columns if c not in base_cols]

        for col in rolling_feature_cols:
            transactions_df[col] = rolling_df[col].values

        # Splits
        train_df, val_df = _split_data(transactions_df)

        # Save parquet
        train_df.to_parquet(TRAIN_PARQUET_PATH, index=False)
        val_df.to_parquet(VAL_PARQUET_PATH, index=False)
        
        logger.info('Train:%d  Val:%d', len(train_df), len(val_df))
        return train_df, val_df, article_df
