import tensorflow as tf
from typing import Dict
from tensorflow import keras
import logging
import os

from .config import Config, Variables
from .model import Basic2TowerModel
from .single_tower_model import SingleTowerModel
from .preprocess import PreprocessedHmData
from .custom_recall import CustomRecall


logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s:%(name)s: %(message)s'
    )


def build_tower_sub_model(vocab_size: int, embedding_dimension: int) -> tf.keras.Model:
    return tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dimension),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(embedding_dimension, activation='relu')
    ])


def get_callbacks():
    return [keras.callbacks.TensorBoard(log_dir='logs', update_freq=100)]


def build_model(data: PreprocessedHmData, config: Config) -> keras.models.Model:
    logger.info("Building the two-tower model...")
    # Customer tower
    customer_lookups = {key: lkp for key, lkp in data.lookups.items() if key in Variables.CUSTOMER_CATEG_VARIABLES}
    customer_numerical_features = Variables.ROLLING_FEATURES
    customer_model = SingleTowerModel(lookups=customer_lookups,
                                    embedding_dimension=config.embedding_dimension,
                                    numerical_features=customer_numerical_features)

    # Article tower
    article_lookups = {key: lkp for key, lkp in data.lookups.items() if key in Variables.ARTICLE_CATEG_VARIABLES}
    article_numerical_features = Variables.IMG_EMB_VARIABLES
    article_model = SingleTowerModel(lookups=article_lookups,
                                   embedding_dimension=config.embedding_dimension,
                                   numerical_features=article_numerical_features)

    model = Basic2TowerModel(customer_model=customer_model,
                           article_model=article_model,
                           data=data)
    logger.info("Two-tower model built successfully.")
    return model

def run_training(model, data, config) -> keras.callbacks.History:
    logger.info("Starting model training...")
    model.compile(optimizer=keras.optimizers.Adagrad(learning_rate=config.learning_rate),
                  metrics=[CustomRecall(k=k, name=f'recall@{k}') for k in config.recall_k_values],
                  run_eagerly=False)

    history = model.fit(x=data.train_ds,
                        epochs=config.nb_epochs,
                        steps_per_epoch=data.nb_train_obs // config.batch_size,
                        validation_data=data.val_ds,
                        # validation_steps=data.nb_val_obs // config.batch_size,
                        callbacks=get_callbacks(),
                        verbose=1)
    logger.info("Model training finished.")
    return history

def save_model(model: keras.models.Model, config: Config):
    """Saves the model weights to the specified directory."""
    weights_path = os.path.join(config.model_save_dir, 'model_weights.h5')
    logger.info(f"Saving model weights to {weights_path}")
    model.save_weights(weights_path)
    logger.info("Model weights saved successfully.")