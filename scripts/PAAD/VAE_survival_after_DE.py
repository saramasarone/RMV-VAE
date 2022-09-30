import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

# this goes in the notebook

import os
import random
from keras import backend as K
import umap.umap_ as umap
import re

# read filtered data
counts = pd.read_csv('/Users/smasarone/Documents/Omics_datasets_integration/counts_after_preprocessing.csv', index_col= 0)
print("counts:", counts.shape)
counts = counts.T

# change the index back to how ot was (R changed it)
new_idx =[]
for i in counts.index:
    new_idx.append(re.sub("[ .]", "-", i))   
counts.index = new_idx

# read clin data and filter it
clin = pd.read_csv('/Users/smasarone/Documents/Omics_datasets_integration/TCGA-PAAD.GDC_phenotype.csv', index_col=0, header=0)
clin = clin.filter(items=counts.index, axis=0)
print("clin", clin.shape)

# scale the data
scaled_counts = StandardScaler().fit_transform(counts)
scaled_counts = pd.DataFrame(scaled_counts, index=counts.index)
scaled_counts.columns = counts.columns

#we are gonna use dead or alive as target 
##### MODEL here ######
seed_value = 0
os.environ['PYTHONHASHSEED'] = str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)
session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
K.set_session(sess)

class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z,
       the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# ENCODER
INPUT_SHAPE = scaled_counts.shape[1]
latent_dim = 30
encoder_inputs = layers.Input(shape=(INPUT_SHAPE), name="x")
h0 = layers.Dense(100, activation="relu")(encoder_inputs)
h1 = layers.Dense(80, activation="relu")(h0)
h2 = layers.Dense(60, activation="relu")(h1)
z_mean = layers.Dense(latent_dim, name="z_mean")(h2)
z_log_var = layers.Dense(latent_dim, name="z_log_var", bias_initializer='zeros')(h2)
z = Sampling()([z_mean, z_log_var])

# summary encoder
encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
print(encoder.summary())

# DECODER
latent_inputs = keras.Input(shape=(latent_dim))
dec_h1 = layers.Dense(60, activation="relu")(latent_inputs)
dec_h2 = layers.Dense(80, activation="relu")(dec_h1)
dec_h3 = layers.Dense(100, activation="relu")(dec_h2)
decoder_outputs = layers.Dense(INPUT_SHAPE, activation="linear")(dec_h3)

# summary decoder
decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
decoder.summary()

class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.mse(data, reconstruction)
                )
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {

            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

#### PLOT
vae = VAE(encoder, decoder)
vae.compile(optimizer=keras.optimizers.Adam())
history = vae.fit(scaled_counts, epochs=100, batch_size=32)
plt.title("Train loss")
plt.plot(history.history['loss'])
plt.plot(history.history['reconstruction_loss'])
plt.plot(history.history['kl_loss'])
plt.show()

print(np.min(history.history['loss']))
print("Done")

get_z = keras.models.Model(inputs=[encoder_inputs], outputs=vae.get_layer("encoder").output, name="VAE")
z_output = get_z.predict({"x": scaled_counts})

# extract only the relevant output
embedding = z_output[2]
pca = PCA(n_components=9)
pca_res = pca.fit_transform(embedding)

pca_raw = PCA(n_components=9)
pca_res_r = pca_raw.fit_transform(scaled_counts)

y = clin['vital_status.demographic']
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,4))
fig.suptitle("PCA of input data vs embedding")
ax1.set_title("PCA of expression data")
sns.scatterplot(x = pca_res_r[:, 0], y=pca_res_r[:, 1], hue=y, palette = 'coolwarm', ax = ax1)
ax2.set_title("PCA of embedding")
sns.scatterplot(x=pca_res[:, 0], y=pca_res[:, 1], hue=y, palette='coolwarm', ax = ax2)
plt.show()

# reducer umap
reducer = umap.UMAP(random_state=42)
embedding_umap = reducer.fit_transform(embedding)
original_umap= reducer.fit_transform(scaled_counts)

# UMAP embedding
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,4))
fig.suptitle("Umap of input data vs embedding")
ax1.set_title("Umap of embedding data")
sns.scatterplot(x = embedding_umap[:, 0], y=embedding_umap[:, 1], hue=y, palette = 'coolwarm', ax = ax1)
ax2.set_title("Umap of original ")
sns.scatterplot(x=original_umap[:, 0], y=original_umap[:, 1], hue=y, palette='coolwarm', ax = ax2)
plt.show()
print(embedding.shape)

# compare VAE embedding with full dataset and filtered dataset without embedding
clf_embedding = LogisticRegression(random_state=0)
scores = cross_val_score(clf_embedding, embedding, y, cv=5)
print(np.mean(scores))

# original counts
clf_orig = LogisticRegression(random_state=0)
scores_orig = cross_val_score(clf_orig, counts, y, cv=5)
print(np.mean(scores_orig))

# Random Forest
rf_emb = RandomForestClassifier(max_depth=2, random_state=0)
scores_rf_emb = cross_val_score(rf_emb, embedding, y, cv=5)
print(np.mean(scores_rf_emb))

# Random Forest
rf_orig = RandomForestClassifier(max_depth=2, random_state=0)
scores_rf_orig = cross_val_score(rf_orig, counts, y, cv=5)
print(np.mean(scores_rf_orig))

# export counts embedding
embedding_df = pd.DataFrame(embedding, index = counts.index)
# embedding_df.to_csv("/Users/smasarone/Documents/Omics_datasets_integration/counts_embedding_PAAD.csv")


















