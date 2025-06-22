import tensorflow as tf
from tensorflow import keras
import logging

from .preprocess import PreprocessedHmData
from .custom_cross_entropy_loss import CustomCrossEntropyLoss

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s:%(name)s: %(message)s'
    )


class Basic2TowerModel(keras.models.Model):
    """A basic two-tower model for recommendation."""
    def __init__(self,
                 customer_model: keras.models.Model,
                 article_model: keras.models.Model,
                 data: PreprocessedHmData):
        super().__init__()
        logger.info("Initializing Basic2TowerModel...")
        self.customer_model = customer_model
        self.article_model = article_model
        self.all_articles = data.all_articles
        self.loss_fn = CustomCrossEntropyLoss(label_probs=data.label_probs_hash_table)

    def call(self, inputs, training=False):
        customer_embeddings = tf.math.l2_normalize(self.customer_model(inputs), axis=-1)

        if training:
            article_embeddings = tf.math.l2_normalize(self.article_model(inputs), axis=-1)
        else:
            raw_art = self.article_model(self.all_articles)
            article_embeddings = tf.math.l2_normalize(raw_art, axis=-1)

        return tf.matmul(customer_embeddings, tf.transpose(article_embeddings))

    def train_step(self, inputs):
        # Forward pass
        with tf.GradientTape() as tape:
            logits = self(inputs, training=True)
            loss_val = self.loss_fn((inputs['article_id'], logits, True))

        # Backward pass
        self.optimizer.minimize(loss_val, self.trainable_variables, tape=tape)
        return {'loss': loss_val}

    def test_step(self, inputs): 
        # Forward pass
        logits = self(inputs, training=False)
        loss_val = self.loss_fn((inputs['article_id'], logits, False))

        num_articles = tf.shape(logits)[1]
        k = tf.minimum(5000, num_articles)
        top_indices = tf.math.top_k(logits, k=k).indices

        # Compute metrics
        metric_results = self.compute_metrics(x=None, y=inputs['article_id'], y_pred=top_indices, sample_weight=None)

        return {'loss': loss_val, **metric_results}
