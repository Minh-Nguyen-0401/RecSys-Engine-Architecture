# %% [markdown]
# Two-Tower Recommendation Pipeline on Colab
This notebook covers preprocessing, model definition, training, and evaluation in one file.  
You should first upload or mount `train_df`, `val_df`, and `article_df`.

# %%
# Install dependencies (run once)
!pip install tensorflow pandas numpy

# %%
# Imports and logging setup
import tensorflow as tf
import pandas as pd
import numpy as np
import logging
from typing import Dict, List

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# %%
# --- USER: load your DataFrames train_df, val_df, article_df here ---
# Example: mount Google Drive and read pickled DataFrames
# from google.colab import drive
# drive.mount('/content/drive')
# train_df = pd.read_pickle('/content/drive/MyDrive/train_df.pkl')
# val_df   = pd.read_pickle('/content/drive/MyDrive/val_df.pkl')
# article_df = pd.read_pickle('/content/drive/MyDrive/article_df.pkl')

# %% 
# Preprocessing utilities
class PreprocessedHmData:
    def __init__(self,
                 train_ds,
                 val_ds,
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

def process_features(inputs, lookups):
    outputs = {}
    for key, value in inputs.items():
        if key in lookups:
            outputs[key] = lookups[key](value)
        else:
            outputs[key] = tf.cast(value, tf.float32)
    return outputs

def get_label_probs_hash_table(train_df, article_lookup):
    counts = train_df.groupby('article_id').size().to_dict()
    total = len(train_df)
    keys = tf.constant(list(counts.keys()), dtype=tf.string)
    keys = article_lookup(keys)
    values = tf.constant([c/total for c in counts.values()], tf.float32)
    return tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(keys, values),
        default_value=0.0
    )

def build_lookups(train_df, article_df, all_categ_vars):
    lookups = {}
    for var in all_categ_vars:
        if var == 'article_id':
            uniq = article_df[var].unique()
            lookups[var] = tf.keras.layers.StringLookup(vocabulary=uniq, num_oov_indices=0)
        else:
            uniq = train_df[var].unique()
            lookups[var] = tf.keras.layers.StringLookup(vocabulary=uniq)
    return lookups

def preprocess(train_df, val_df, article_df,
               all_categ_vars, rolling_features, img_emb_vars,
               batch_size=1024):
    # Build lookups
    lookups = build_lookups(train_df, article_df, all_categ_vars)
    # Create tf.data.Datasets
    features = all_categ_vars + rolling_features + img_emb_vars
    train_ds = tf.data.Dataset.from_tensor_slices(dict(train_df[features]))         .shuffle(100_000).batch(batch_size).map(lambda x: process_features(x, lookups)).repeat()
    val_ds   = tf.data.Dataset.from_tensor_slices(dict(val_df[features]))         .batch(batch_size).map(lambda x: process_features(x, lookups))
    # Build all_articles tensor
    all_articles_df = article_df[all_categ_vars + img_emb_vars].fillna(0.0)
    article_ds = tf.data.Dataset.from_tensor_slices(dict(all_articles_df))         .batch(len(all_articles_df)).map(lambda x: process_features(x, lookups))
    all_articles = next(iter(article_ds))
    # Label prob table
    label_probs = get_label_probs_hash_table(train_df, lookups['article_id'])
    return PreprocessedHmData(train_ds, val_ds, lookups, all_articles,
                              label_probs, len(train_df), len(val_df))

# %%
# --- Define your feature lists ---
# Example:
all_categ_vars = ['customer_id', 'article_id', 'age_interval', 'FN', 'Active', 'club_member_status', 'fashion_news_frequency']
rolling_features = ['days_since_last', 'purchase_count_7d']  # replace with your rolling feature names
img_emb_vars = ['img_embd_0', 'img_embd_1', 'img_embd_2']    # replace with your image embedding cols

# %%
# Define CustomRecall metric
class CustomRecall(tf.keras.metrics.Metric):
    def __init__(self, k, name='recall', **kwargs):
        super().__init__(name=f'{name}_at_{k}', **kwargs)
        self._cumulative = tf.Variable(0.)
        self._count = tf.Variable(0.)
        self._k = k

    def _recall_at_k(self, y_true, top_idx):
        k = tf.minimum(self._k, tf.shape(top_idx)[1])
        top = top_idx[:, :k]
        hits = tf.cast(tf.equal(tf.expand_dims(y_true, -1), top), tf.float32)
        return tf.reduce_any(hits, axis=1)

    def update_state(self, y_true, y_pred, sample_weight=None):
        hits = self._recall_at_k(y_true, y_pred)
        hits = tf.cast(hits, tf.float32)
        if sample_weight is not None:
            hits *= sample_weight
        self._cumulative.assign_add(tf.reduce_sum(hits))
        self._count.assign_add(tf.reduce_sum(tf.ones_like(hits)))

    def result(self):
        return tf.math.divide_no_nan(self._cumulative, self._count)

    def reset_states(self):
        self._cumulative.assign(0.)
        self._count.assign(0.)

# %%
# Define CustomCrossEntropyLoss
class CustomCrossEntropyLoss(tf.keras.losses.Loss):
    def __init__(self, label_probs, **kwargs):
        super().__init__(**kwargs)
        self.label_probs = label_probs

    def call(self, y_true, logits, training=False):
        if training:
            probs = self.label_probs.lookup(y_true)
            logits -= tf.math.log(probs)
            y_true = tf.range(tf.shape(logits)[0], dtype=y_true.dtype)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true, logits=logits)
        return tf.reduce_mean(loss)

# %%
# Define single-tower model
class SingleTowerModel(keras.Model):
    def __init__(self, lookups, num_features, embedding_dim):
        super().__init__()
        self.emb_layers = {}
        for var, lookup in lookups.items():
            dim = 128 if var in ['customer_id','article_id'] else int(3 * np.log2(lookup.vocabulary_size()))
            self.emb_layers[var] = keras.layers.Embedding(lookup.vocabulary_size(), dim)
        self.dense1 = keras.layers.Dense(512, activation='relu')
        self.dense2 = keras.layers.Dense(256, activation='relu')
        self.dense3 = keras.layers.Dense(embedding_dim, activation='relu')

    def call(self, inputs):
        embs = [self.emb_layers[k](inputs[k]) for k in self.emb_layers]
        x = tf.concat(embs + [tf.reshape(inputs[f],(-1,1)) for f in num_features], axis=1)
        x = self.dense1(x); x = self.dense2(x); return self.dense3(x)

# %%
# Define two-tower model
class Basic2TowerModel(keras.Model):
    def __init__(self, cust_model, art_model, data, top_k=5000):
        super().__init__()
        self.cust = cust_model; self.art = art_model
        self.all_articles = data.all_articles
        self.loss_fn = CustomCrossEntropyLoss(data.label_probs_hash_table)
        self.top_k = top_k

    def call(self, inputs, training=False):
        cust_emb = tf.math.l2_normalize(self.cust(inputs),axis=-1)
        if training:
            art_emb = tf.math.l2_normalize(self.art(inputs),axis=-1)
        else:
            art_emb = tf.math.l2_normalize(self.art(self.all_articles),axis=-1)
        return tf.matmul(cust_emb, art_emb, transpose_b=True)

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            logits = self(x, True)
            loss = self.loss_fn(y, logits, training=True)
        self.optimizer.minimize(loss, self.trainable_variables, tape=tape)
        return {'loss': loss}

    def test_step(self, data):
        x, y = data
        logits = self(x, False)
        loss = self.loss_fn(y, logits, training=False)
        # top-k dynamic
        k = tf.minimum(self.top_k, tf.shape(logits)[1])
        top_idx = tf.math.top_k(logits, k=k).indices
        # metrics
        metrics = {m.name: m.result() for m in self.metrics}
        for m in self.metrics:
            m.update_state(y, top_idx)
        return {'loss': loss, **metrics}

# %%
# =========== Run preprocessing ===========
prep = preprocess(train_df, val_df, article_df,
                  all_categ_vars, rolling_features, img_emb_vars,
                  batch_size=512)

# %%
# =========== Build models ===========
cust_model = SingleTowerModel(prep.lookups, rolling_features, 128)
art_model = SingleTowerModel(prep.lookups, img_emb_vars, 128)
two_tower = Basic2TowerModel(cust_model, art_model, prep, top_k=5000)

# %%
# Compile and train
two_tower.compile(
    optimizer=keras.optimizers.Adam(0.001),
    metrics=[CustomRecall(k) for k in [100,500,1000,5000]]
)
history = two_tower.fit(prep.train_ds,
                        steps_per_epoch=prep.nb_train_obs//512,
                        validation_data=prep.val_ds,
                        validation_steps=prep.nb_val_obs//512,
                        epochs=4)

