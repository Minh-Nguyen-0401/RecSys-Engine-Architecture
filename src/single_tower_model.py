import math
from typing import Dict, List
import tensorflow as tf
from tensorflow import keras
import logging

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s:%(name)s: %(message)s'
    )


class SingleTowerModel(keras.models.Model):
    """A single tower model for either customers or articles."""
    def __init__(self,
                 lookups: Dict[str, tf.keras.layers.StringLookup],
                 embedding_dimension: int,
                 numerical_features: List[str] = None):
        super().__init__()
        self._numerical_features = numerical_features if numerical_features else []
        logger.info(f"Initializing SingleTowerModel for categorical features: {list(lookups.keys())} "
                    f"and numerical features: {self._numerical_features}")

        self._all_embeddings = {}
        for categ_variable in lookups.keys():
            lookup = lookups[categ_variable]
            vocab_size = lookup.vocabulary_size()
            if categ_variable == 'article_id' or categ_variable == 'customer_id':
                cat_var_emb_dim = 128
            else:
                cat_var_emb_dim = int(3 * math.log(vocab_size, 2))
            embedding_layer = tf.keras.layers.Embedding(vocab_size, cat_var_emb_dim)
            self._all_embeddings[categ_variable] = embedding_layer

        self._dense1 = tf.keras.layers.Dense(512, activation='relu')
        self._dense2 = tf.keras.layers.Dense(256, activation='relu')
        self._dense3 = tf.keras.layers.Dense(embedding_dimension, activation='relu')

    def call(self, inputs):
        # Process categorical features
        categorical_embeddings = []
        for variable, embedding_layer in self._all_embeddings.items():
            embeddings = embedding_layer(inputs[variable])
            categorical_embeddings.append(embeddings)

        # Process numerical features
        numerical_features_processed = []
        if self._numerical_features:
            for feature in self._numerical_features:
                numerical_features_processed.append(tf.reshape(inputs[feature], (-1, 1)))
        
        # Concatenate all features
        all_features_concat = tf.concat(categorical_embeddings + numerical_features_processed, axis=1)
        
        outputs = self._dense1(all_features_concat)
        outputs = self._dense2(outputs)
        outputs = self._dense3(outputs)
        return outputs
