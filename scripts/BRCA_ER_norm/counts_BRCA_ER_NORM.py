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
import os
import random
from keras import backend as K
import umap.umap_ as umap
import re

# read filtered data
counts = pd.read_csv('corrected_counts_BRCA_ER.csv', index_col= 0)
print("counts after EdgeR filtering:", counts.shape)
counts = counts.T
clin = pd.read_csv('clin_for_DE.csv', index_col=0, header=0)
print("clin", clin.shape)

# read old label and use it for col names
idx = pd.read_csv('idx_ER.csv', index_col =0)
counts.index = idx.iloc[:,0]

# read old label and use it for col names
columns = counts.columns
new_cols =[]
for i in counts.columns:
    new_cols.append(i.split("|")[0])
counts.columns = new_cols
print(counts.columns)

# scale the data
scaled_counts = StandardScaler().fit_transform(counts)
scaled_counts = pd.DataFrame(scaled_counts, index=counts.index)
scaled_counts.columns = counts.columns

# check for correlated genes with ER status
from scipy import stats
d ={'corr':[], 'p_val':[]}

new = clin.breast_carcinoma_estrogen_receptor_status.replace({"positive":1, "negative":0})
for i in range(scaled_counts.shape[1]):
    a = np.array(scaled_counts.iloc[:,i])
    res = stats.pointbiserialr(a, new)
    d['corr'].append(res[0])
    d['p_val'].append(res[1])
print(len(d['corr']))

corr_df = pd.DataFrame(d)
corr_df.index = scaled_counts.columns
corr_df = corr_df.sort_values(by='corr', ascending=False)
### RESULTS are FOXA1, GATA3, ANKRA2, DNLI1, TTCB, C16orf57, MSL3, FAM171A1, B3GNT5


# select the gene_name
gene_i_want = None
genes_selected=[]
for i, z in enumerate(scaled_counts.columns):
    genes_selected.append(z)
    if z == 'FOXA1':  
        gene_i_want = i
print(gene_i_want)

# we are gonna use dead or alive as target  
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
INPUT_SHAPE = scaled_counts.shape[1]
latent_dim = 10
encoder_inputs = layers.Input(shape=(INPUT_SHAPE), name="x")
h0 = layers.Dense(1000, activation="relu")(encoder_inputs)
h1 = layers.Dense(500, activation="relu")(h0)
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
dec_h2 = layers.Dense(500, activation="relu")(dec_h1)
dec_h3 = layers.Dense(1000, activation="relu")(dec_h2)
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
        self.sign_loss_tracker = keras.metrics.Mean(name="sign_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
            self.sign_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.mse(data, reconstruction)
                )/INPUT_SHAPE
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))/latent_dim
            
            #regularisation loss
            x = tf.reshape(z[:, 0], [-1,1])
            x = tf.repeat(x, tf.shape(x)[0], axis = 1)
            x_diff_sign = (x - tf.transpose(x)) 
            x_diff_sign = tf.reshape(x_diff_sign, [-1,1]) 
            x_diff_sign = tf.tanh(x_diff_sign * 100)
            
            y = tf.reshape(data[:, gene_i_want], [-1,1])  
            y = tf.repeat(y, tf.shape(y)[0], axis = 1)
            y = tf.cast(y, tf.float32)
            y_diff_sign = tf.sign(y - tf.transpose(y))
            y_diff_sign = tf.reshape(y_diff_sign, [-1,1]) 
            sign_loss = tf.keras.losses.mean_absolute_error(x_diff_sign, y_diff_sign)

            total_loss = reconstruction_loss + kl_loss*0.1 +sign_loss #0.1

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.sign_loss_tracker.update_state(sign_loss)

        return {

            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
            "sign_loss": self.kl_loss_tracker.result(),
        }

#### PLOT
vae = VAE(encoder, decoder)
vae.compile(optimizer=keras.optimizers.Adam())
history = vae.fit(scaled_counts, epochs=60, batch_size=32)
plt.title("Train loss")
plt.plot(history.history['loss'])
plt.plot(history.history['reconstruction_loss'])
plt.plot(history.history['kl_loss'])
plt.plot(history.history['sign_loss'])
plt.show()

print(np.min(history.history['loss']))
print("Done")

get_z = keras.models.Model(inputs=[encoder_inputs], outputs=vae.get_layer("encoder").output, name="VAE")
z_output = get_z.predict({"x": scaled_counts})

# extract only the relevant output
embedding = z_output[2]
pca = PCA(n_components=4)
pca_res = pca.fit_transform(embedding)

pca_raw = PCA(n_components=4)
pca_res_r = pca_raw.fit_transform(scaled_counts)

y = clin.breast_carcinoma_estrogen_receptor_status
y_for_plot = ["ER+" if i =='positive' else 'ER-' for i in y]

fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20,4))
fig.suptitle("PCA of input data vs embedding")
ax1.set_title("PCA of expression data")
sns.scatterplot(x = pca_res_r[:, 0], y=pca_res_r[:, 1], hue=y_for_plot, palette = 'magma', ax = ax1, s = 30)
ax1.legend(loc="lower right")
ax2.set_title("PCA of expression data")
sns.scatterplot(x = pca_res_r[:, 0], y=pca_res_r[:, 1], hue=scaled_counts.iloc[:,gene_i_want], palette = 'coolwarm', ax = ax2, s = 30)
ax2.legend(loc="lower right")
ax3.set_title("PCA of embedding")
sns.scatterplot(x=pca_res[:, 0], y=pca_res[:, 1], hue=y_for_plot, palette='magma', ax = ax3, s = 30)
ax3.legend(loc="lower right")
ax4.set_title("PCA of embedding by expr.")
sns.scatterplot(x=pca_res[:, 0], y=pca_res[:, 1], hue=scaled_counts.iloc[:,gene_i_want], palette='coolwarm', ax = ax4, s = 30)
ax4.legend(loc="lower right")
plt.show()


# # reducer umap
# reducer = umap.UMAP(random_state=42)
# embedding_umap = reducer.fit_transform(embedding)
# original_umap = reducer.fit_transform(scaled_counts)

# #UMAP embedding
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,4))
# fig.suptitle("Umap of input data vs embedding")
# ax1.set_title("Umap of embedding data")
# sns.scatterplot(x = embedding_umap[:, 0], y=embedding_umap[:, 1], hue=y, palette = 'coolwarm', ax = ax1)
# ax2.set_title("Umap of original ")
# sns.scatterplot(x=original_umap[:, 0], y=original_umap[:, 1], hue=y, palette='coolwarm', ax = ax2)
# plt.show()
# print(embedding.shape)

# Random Forest
rf_emb = RandomForestClassifier(max_depth=2, random_state=0)
scores_rf_emb = cross_val_score(rf_emb, embedding, y, cv=10)
print(np.mean(scores_rf_emb))

# Random Forest
rf_orig = RandomForestClassifier(max_depth=2, random_state=0)
scores_rf_orig = cross_val_score(rf_orig, counts, y, cv=10)
print(np.mean(scores_rf_orig))
print(scores_rf_orig)

# export counts embedding
embedding_df = pd.DataFrame(embedding, index = counts.index)
# embedding_df.to_csv("/Users/smasarone/Documents/Omics_datasets_integration/BRCA_experiments/BRCA_embeddings/counts_embedding_BRCA_norm.csv")

