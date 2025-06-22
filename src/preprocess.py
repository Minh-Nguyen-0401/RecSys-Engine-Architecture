import pandas as pd
import tensorflow as tf
from typing import Dict, List
import logging

from .config import Variables


logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s:%(name)s: %(message)s'
    )


class PreprocessedHmData:
    def __init__(self,
                 train_ds: tf.data.Dataset,
                 val_ds: tf.data.Dataset,
                 lookups: Dict[str, tf.keras.layers.StringLookup],
                 all_articles: Dict[str, tf.Tensor],
                 label_probs_hash_table: tf.lookup.StaticHashTable,
                 nb_train_obs: int,
                 nb_val_obs: int):
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.lookups = lookups
        self.all_articles = all_articles
        self.label_probs_hash_table = label_probs_hash_table
        self.nb_train_obs = nb_train_obs
        self.nb_val_obs = nb_val_obs


def process_features(inputs: Dict[str, tf.Tensor],
                     lookups: Dict[str, tf.keras.layers.StringLookup]) -> Dict[str, tf.Tensor]:
    """Applies string lookups to categorical features and casts numerical features."""
    outputs = {}
    for key, value in inputs.items():
        if key in lookups:
            outputs[key] = lookups[key](value)
        else:
            # It's a numerical feature, cast to float32
            outputs[key] = tf.cast(value, tf.float32)
    return outputs


def get_label_probs_hash_table(train_df: pd.DataFrame,
                               article_lookup: tf.keras.layers.StringLookup) -> tf.lookup.StaticHashTable:
    article_counts_dict = train_df.groupby('article_id')['article_id'].count().to_dict()
    nb_transactions = train_df.shape[0]
    keys = list(article_counts_dict.keys())
    values = [count / nb_transactions for count in article_counts_dict.values()]

    keys = tf.constant(keys, dtype=tf.string)
    keys = article_lookup(keys)
    values = tf.constant(values, dtype=tf.float32)

    return tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(keys, values),
                                     default_value=0.0)


def build_lookups(train_df, article_df) -> Dict[str, tf.keras.layers.StringLookup]:
    lookups = {}
    for categ_variable in Variables.ALL_CATEG_VARIABLES:
        if categ_variable == 'article_id':
            unique_values = article_df[categ_variable].unique()
            lookups[categ_variable] = tf.keras.layers.StringLookup(vocabulary=unique_values, num_oov_indices=0)
        else:
            unique_values = train_df[categ_variable].unique()
            lookups[categ_variable] = tf.keras.layers.StringLookup(vocabulary=unique_values)
    return lookups


def preprocess(train_df: pd.DataFrame,
               val_df: pd.DataFrame | None = None,
               article_df: pd.DataFrame | None = None,
               batch_size: int = 1024) -> PreprocessedHmData:
    logger.info("Starting preprocessing...")

    if val_df is None:
        val_df = train_df.iloc[:0].copy()
    if article_df is None:
        raise ValueError("article_df must be provided for preprocessing.")

    nb_train_obs = train_df.shape[0]
    nb_val_obs = val_df.shape[0]

    lookups = build_lookups(train_df, article_df)
    logger.info(f"Built {len(lookups)} lookups for categorical variables.")

    all_features = Variables.ALL_CATEG_VARIABLES + Variables.ROLLING_FEATURES + Variables.IMG_EMB_VARIABLES

    logger.info("Creating TensorFlow datasets for train, validation sets...")
    train_ds = tf.data.Dataset.from_tensor_slices(dict(train_df[all_features])) \
        .shuffle(100_000) \
        .batch(batch_size) \
        .map(lambda inputs: process_features(inputs, lookups)) \
        .repeat()
    val_ds = tf.data.Dataset.from_tensor_slices(dict(val_df[all_features])) \
        .batch(batch_size) \
        .map(lambda inputs: process_features(inputs, lookups))
    logger.info("TensorFlow datasets created.")

    logger.info("Building all_articles tensor for inference...")
    article_lookup = lookups['article_id']

    # Build article features for inference
    all_articles_df = article_df[Variables.ARTICLE_CATEG_VARIABLES + Variables.IMG_EMB_VARIABLES].copy()

    for col in Variables.IMG_EMB_VARIABLES:
        if col in all_articles_df.columns:
            all_articles_df[col].fillna(0.0, inplace=True)

    article_features_all = Variables.ARTICLE_CATEG_VARIABLES + Variables.IMG_EMB_VARIABLES
    article_lookups = {key: lkp for key, lkp in lookups.items() if key in Variables.ARTICLE_CATEG_VARIABLES}

    article_ds = tf.data.Dataset.from_tensor_slices(dict(all_articles_df[article_features_all])) \
        .batch(len(all_articles_df)) \
        .map(lambda inputs: process_features(inputs, article_lookups))
    article_batch = next(iter(article_ds))
    all_articles = {k: v for k, v in article_batch.items()}

    logger.info("all_articles tensor built successfully.")

    logger.info("Calculating label probabilities for log(q) correction...")
    label_probs_hash_table = get_label_probs_hash_table(train_df, article_lookup)
    logger.info("Label probabilities calculated.")

    logger.info("Preprocessing finished.")
    return PreprocessedHmData(train_ds=train_ds,
                              val_ds=val_ds,
                              lookups=lookups,
                              all_articles=all_articles,
                              label_probs_hash_table=label_probs_hash_table,
                              nb_train_obs=nb_train_obs,
                              nb_val_obs=nb_val_obs
    )