import logging
import pandas as pd
import numpy as np
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s:%(name)s: %(message)s'
    )

ROLLING_WINDOW_DAYS = 30

def add_rolling_features(transactions_df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Adding rolling features...")
    """
    Creates daily aggregated and rolling window features with memory optimization.
    This prevents memory errors by using efficient merges and downcasting data types.
    """
    logger.info("Starting feature engineering with daily aggregation and memory optimization...")
    transactions_df['t_dat'] = pd.to_datetime(transactions_df['t_dat'])
    transactions_df['date'] = transactions_df['t_dat'].dt.date
    transactions_df['date'] = pd.to_datetime(transactions_df['date'])
    transactions_df['__idx'] = np.arange(len(transactions_df))
    transactions_df.sort_values('t_dat', inplace=True)
    

    # Customer Features
    logger.info("Calculating rolling features for customers...")
    # 1. Aggregate daily
    customer_daily_agg = transactions_df.groupby(['customer_id', 'date']).agg(
        customer_daily_spend=('price', 'sum'),
        customer_daily_articles=('article_id', 'nunique')
    ).reset_index()
    # 2. Calculate rolling features on the daily aggregate
    customer_daily_agg_indexed = customer_daily_agg.set_index('date')
    rolling_features = customer_daily_agg_indexed.groupby('customer_id')[['customer_daily_spend', 'customer_daily_articles']].rolling(window=f'{ROLLING_WINDOW_DAYS}D', min_periods=7).sum()
    rolling_features = rolling_features.rename(columns={'customer_daily_spend': 'customer_spend_30d', 'customer_daily_articles': 'customer_articles_30d'}).reset_index()
    # 3. Merge rolling features back to daily aggregate
    customer_daily_agg = pd.merge(customer_daily_agg, rolling_features, on=['customer_id', 'date'], how='left')
    # 4. Merge daily and rolling customer features back to main transactions table
    transactions_df = pd.merge(transactions_df, customer_daily_agg[['customer_id', 'date', 'customer_spend_30d', 'customer_articles_30d', 'customer_daily_spend', 'customer_daily_articles']], on=['customer_id', 'date'], how='left')
    logger.info("Rolling features added successfully.")


    logger.info("Calculating time-delta features...")
    # Days since last transaction for customer
    transactions_df['days_since_last_customer_trans'] = transactions_df.groupby('customer_id')['t_dat'].diff().dt.days.fillna(0).astype('int32')

    # Final Cleanup
    logger.info("Cleaning up final dataset and optimizing memory...")
    feature_cols = [
        'customer_daily_spend', 'customer_daily_articles', 'customer_spend_30d', 'customer_articles_30d',
        'days_since_last_customer_trans'
    ]
    transactions_df[feature_cols] = transactions_df[feature_cols].fillna(0)
    
    # Downcast for memory efficiency
    for col in feature_cols:
        if 'spend' in col or 'price' in col:
            transactions_df[col] = transactions_df[col].astype('float32')
        else:
            transactions_df[col] = transactions_df[col].astype('int32')
    
    transactions_df = transactions_df.drop(columns=['date'])
    transactions_df.sort_values(by=['__idx'], inplace=True)
    transactions_df.drop(columns=['__idx'], inplace=True)
    
    logger.info("Feature engineering completed successfully.")
    return transactions_df
