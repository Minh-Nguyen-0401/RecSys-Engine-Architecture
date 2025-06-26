import tensorflow as tf
import logging

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s:%(name)s: %(message)s'
    )


class CustomCrossEntropyLoss(tf.keras.layers.Layer):
    """Custom loss to apply log(q) correction for sampled softmax."""
    def __init__(self, label_probs: tf.lookup.StaticHashTable = None, **kwargs):
        super().__init__(**kwargs)
        logger.info("Initializing CustomCrossEntropyLoss...")
        self._label_probs = label_probs

    def call(self, inputs):
        true_labels, logits, training = inputs
        batch_size = tf.shape(logits)[0]

        if training:
            # Apply log q correction
            label_probs = self._label_probs.lookup(true_labels)
            logits -= tf.math.log(label_probs)
            # Override true labels to apply the softmax as if we only had "batch size" classes
            true_labels = tf.range(
                0,
                batch_size,
                dtype=true_labels.dtype
            )

        # Compute softmax cross entropy
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=true_labels, logits=logits)
        return tf.reduce_mean(loss)

    def get_config(self):
        config = super().get_config()
        return config
