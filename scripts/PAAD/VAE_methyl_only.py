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

import os
import random
from keras import backend as K
import umap.umap_ as umap

#use this in the end

# read methylation data and map it
methy = pd.read_csv('/Users/smasarone/Documents/Omics_datasets_integration/TCGA-PAAD.methylation450.tsv', delim_whitespace=True, index_col=0)
map_methy = pd.read_csv('/Users/smasarone/Documents/Omics_datasets_integration/illuminaMethyl450_hg38_GDC', delim_whitespace=True, index_col=0)

# only get the idx for which there's a geneid 
ref_data = map_methy[map_methy['gene']!="."]
gene_ids=ref_data.index
methy = methy.filter(items = gene_ids, axis=0)
methy.index = ref_data['gene']

# read and map counts data to filter methylation data
counts = pd.read_csv('/Users/smasarone/Documents/Omics_datasets_integration/TCGA-PAAD.htseq_counts.tsv', delim_whitespace=True, index_col=0)
counts_map = pd.read_csv('/Users/smasarone/Documents/Omics_datasets_integration/gencode.v22.annotation.gene.probeMap', delim_whitespace=True, index_col=0)
counts = counts.filter(items = counts_map.gene.index, axis =0)
counts.index = counts_map.gene
methy = methy.filter(items = counts.columns, axis =1)

# read clin data
clin = pd.read_csv('/Users/smasarone/Documents/Omics_datasets_integration/TCGA-PAAD.GDC_phenotype.csv', index_col=0, header=0)
clin = clin.filter(items=counts.columns, axis=0)

# scale and filter
methy = methy.T
methy = methy.dropna(how = 'all')
methy.fillna(value = 0, inplace=True)
clin = clin.filter(items= methy.index, axis=0)

#filter columns with Variance Threshold
selector = VarianceThreshold(threshold=0.05)
methy_filtered = selector.fit_transform(methy)
mask = selector.get_support()  # to get the right idx
final_col = methy.columns[mask]

scaled_methyl = StandardScaler().fit_transform(methy_filtered)
scaled_methyl = pd.DataFrame(scaled_methyl, index=methy.index)
scaled_methyl.columns = final_col
print(scaled_methyl)

# we are gonna use dead or alive as target variable later on
###### MODEL here ######
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
INPUT_SHAPE = scaled_methyl.shape[1]
latent_dim = 40
encoder_inputs = layers.Input(shape=(INPUT_SHAPE), name="x")
h0 = layers.Dense(500, activation="relu")(encoder_inputs)
h1 = layers.Dense(300, activation="relu")(h0)
h2 = layers.Dense(200, activation="relu")(h1)
z_mean = layers.Dense(latent_dim, name="z_mean")(h2)
z_log_var = layers.Dense(latent_dim, name="z_log_var", bias_initializer='zeros')(h2)
z = Sampling()([z_mean, z_log_var])

# summary encoder
encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
print(encoder.summary())

# DECODER
latent_inputs = keras.Input(shape=(latent_dim))
dec_h1 = layers.Dense(200, activation="relu")(latent_inputs)
dec_h2 = layers.Dense(300, activation="relu")(dec_h1)
dec_h3 = layers.Dense(500, activation="relu")(dec_h2)
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
history = vae.fit(scaled_methyl, epochs=100, batch_size=32)
plt.title("Train loss")
plt.plot(history.history['loss'])
plt.plot(history.history['reconstruction_loss'])
plt.plot(history.history['kl_loss'])
plt.ylim(-3, 300)
plt.show()

print(np.min(history.history['loss']))
print("Done")

get_z = keras.models.Model(inputs=[encoder_inputs], outputs=vae.get_layer("encoder").output, name="VAE")
z_output = get_z.predict({"x": scaled_methyl})

# extract only the relevant output
embedding = z_output[2]
pca = PCA(n_components=9)
pca_res = pca.fit_transform(embedding)

pca_raw = PCA(n_components=9)
pca_res_r = pca_raw.fit_transform(scaled_methyl)

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
embedding_original= reducer.fit_transform(scaled_methyl)

# UMAP embedding
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,4))
fig.suptitle("Umap of input data vs embedding")
ax1.set_title("Umap of expression data")
sns.scatterplot(x = embedding_umap[:, 0], y=embedding_umap[:, 1], hue=y, palette = 'coolwarm', ax = ax1)
ax2.set_title("PCA of embedding")
sns.scatterplot(x=embedding_original[:, 0], y=embedding_original[:, 1], hue=y, palette='coolwarm', ax = ax2)
plt.show()

print(embedding.shape)

# compare VAE embedding with full dataset and filtered dataset without embedding
clf_embedding = LogisticRegression(random_state=0, penalty = 'l1', solver = 'saga')
scores = cross_val_score(clf_embedding, embedding, y, cv=5)
print(np.mean(scores))

# filtered methylation
clf_orig = LogisticRegression(random_state=0, penalty = 'l1', solver = 'saga')
scores_orig = cross_val_score(clf_orig, scaled_methyl, y, cv=5)
print(np.mean(scores_orig))

# export the embedding
embedding_df = pd.DataFrame(embedding, index = methy.index)
print(embedding_df.shape)
embedding_df.to_csv('/Users/smasarone/Documents/Omics_datasets_integration/methylation_embedding.csv')







