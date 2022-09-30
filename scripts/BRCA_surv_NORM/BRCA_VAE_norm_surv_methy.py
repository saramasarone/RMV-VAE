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
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

import os
import random
from keras import backend as K
import umap.umap_ as umap

### Embedding obtained from here is the one currently used in the ppt


counts = pd.read_csv('Integration_BRCA_data/BRCA.rnaseqv2__illuminahiseq_rnaseqv2__unc_edu__Level_3__RSEM_genes_normalized__data.data.txt', 
                    index_col=0, sep = "\t", header = 0)
counts = counts.drop(['gene_id'], axis =0)

clin = pd.read_csv('Integration_BRCA_data/All_clin_data.txt', 
                    index_col=0, sep = "\t", header = 0)
methy = pd.read_csv('Integration_BRCA_data/BRCA.meth.by_mean.data.txt', 
                    index_col=0, sep = "\t", header = 0)
methy = methy.drop(['Composite Element REF'], axis = 0)


print(methy.shape)
print(counts.shape)

# change id
new_id =[]
for i in counts.columns:
    new_id.append(i[:12])
counts.columns = new_id

new_id_m =[]
for i in methy.columns:
    new_id_m.append(i[:12])
methy.columns = new_id_m
methy.columns = methy.columns.get_level_values(0)

# drop duplicates if there are any
methy = methy.T
counts = counts.T
methy.drop_duplicates(keep='first', inplace=True)
counts.drop_duplicates(keep='first', inplace=True)

idx_to_keep=[]
for i, z in enumerate(methy.index.duplicated()):
    if z == False:
        idx_to_keep.append(i)

idx_to_keep_c=[]
for i, z in enumerate(counts.index.duplicated()):
    if z == False:
        idx_to_keep_c.append(i)

methy = methy.iloc[idx_to_keep,:]  #idx_to_keep
counts = counts.iloc[idx_to_keep_c,:]
counts = counts.filter(items = methy.index, axis =0)
methy = methy.filter(items = counts.index, axis =0)
print(counts.shape)
print(methy.shape)

new_idx_clin=[]
for i in clin.columns:
    new_idx_clin.append(i.upper())
clin.columns = new_idx_clin
clin = clin.filter(items = methy.index, axis = 1)
clin = clin.T

# scale and filter
methy = methy.dropna(how = 'all')
methy.fillna(value = 0, inplace=True)

#filter columns with Variance Threshold
selector = VarianceThreshold(threshold=0.02) #0.02
methy_filtered = selector.fit_transform(methy)
mask = selector.get_support()    # to get the right idx
final_col = methy.columns[mask]

print("Done")
scaled_methyl = StandardScaler().fit_transform(methy_filtered)
scaled_methyl = pd.DataFrame(scaled_methyl, index=methy.index)
scaled_methyl.columns = final_col
print("filtered shape:", scaled_methyl.shape)

# select the gene_name
gene_i_want = None
genes_selected=[]
for i, z in enumerate(scaled_methyl.columns):
    genes_selected.append(z)
    if z == 'TMEM101':   #SNORD45B C21orf96 TMEM101
        gene_i_want = i
print(gene_i_want)

#pd.DataFrame(genes_selected).to_csv('gene_selected.csv')

#### MODEL here ######
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
latent_dim = 30
encoder_inputs = layers.Input(shape=(INPUT_SHAPE), name="x")
h0 = layers.Dense(800, activation="relu")(encoder_inputs)
h1 = layers.Dense(500, activation="relu")(h0)
h2 = layers.Dense(100, activation="relu")(h1)
z_mean = layers.Dense(latent_dim, name="z_mean")(h2)
z_log_var = layers.Dense(latent_dim, name="z_log_var", bias_initializer='zeros')(h2)
z = Sampling()([z_mean, z_log_var])

# summary encoder
encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
print(encoder.summary())

# DECODER
latent_inputs = keras.Input(shape=(latent_dim))
dec_h1 = layers.Dense(100, activation="relu")(latent_inputs)
dec_h2 = layers.Dense(500, activation="relu")(dec_h1)
dec_h3 = layers.Dense(800, activation="relu")(dec_h2)
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
            self.sign_loss_tracker
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
            total_loss = reconstruction_loss + (kl_loss* 0.1) + (sign_loss*5)   #0.1 5 


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
            "sign_loss": self.sign_loss_tracker.result()
        }

#### PLOT
vae = VAE(encoder, decoder)
vae.compile(optimizer=keras.optimizers.Adam())
history = vae.fit(scaled_methyl, epochs=120, batch_size=64)
plt.title("Train loss")
plt.plot(history.history['loss'])
plt.plot(history.history['reconstruction_loss'])
plt.plot(history.history['kl_loss'])
plt.plot(history.history['sign_loss'])
plt.legend()
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

y = clin.vital_status

fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20,4))
fig.suptitle("PCA of input data vs embedding")
ax1.set_title("PCA of expression data")
sns.scatterplot(x = pca_res_r[:, 0], y=pca_res_r[:, 1], hue=y, palette = 'magma', ax = ax1, s = 30)
ax1.legend(loc="lower right")
ax2.set_title('PCA of expression by gene')
sns.scatterplot(x=pca_res_r[:, 0], y=pca_res_r[:, 1], hue=scaled_methyl.iloc[:, gene_i_want], palette='coolwarm', ax = ax2, s = 30)
ax2.legend(loc="lower right")
ax3.set_title('PCA of embedding by ER')
sns.scatterplot(x=pca_res[:, 0], y=pca_res[:, 1], hue=y, palette='magma', ax = ax3, s = 30)
ax3.legend(loc="lower right")
ax4.set_title("PCA of embedding by gene")
sns.scatterplot(x=pca_res[:, 0], y=pca_res[:, 1], hue=scaled_methyl.iloc[:, gene_i_want], palette='coolwarm', ax = ax4, s = 30)
ax4.legend(loc="lower right")
plt.show()

# # reducer umap
# reducer = umap.UMAP(random_state=42)
# embedding_umap = reducer.fit_transform(embedding)
# embedding_original= reducer.fit_transform(scaled_methyl)

# # UMAP embedding
# fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(16, 4))
# fig.suptitle("Umap of input data vs embedding")
# ax1.set_title("Umap of original methyl")
# sns.scatterplot(x=embedding_original[:, 0], y=embedding_original[:, 1], hue=y, palette='coolwarm', ax = ax1, s = 10)
# ax2.set_title("Umap of embedding")
# sns.scatterplot(x = embedding_umap[:, 0], y=embedding_umap[:, 1], hue=y, palette = 'coolwarm', ax = ax2, s = 10)
# ax3.set_title(f'Umap of embedding by {gene_i_want}')
# sns.scatterplot(x=embedding_umap[:, 0], y=embedding_umap[:, 1], hue=scaled_methyl.iloc[:, gene_i_want], palette='coolwarm', ax = ax3, s = 10)
# ax4.set_title(f'Umap of embedding by ER')
# sns.scatterplot(x=embedding_umap[:, 0], y=embedding_umap[:, 1], hue=clin.breast_carcinoma_estrogen_receptor_status, palette='coolwarm', ax = ax4, s = 10)
# plt.show()

# print("emb shape:", embedding.shape)
# print("methy filtered shape:", methy_filtered.shape, scaled_methyl.shape)

# compare VAE embedding with full dataset and filtered dataset without embedding
clf_embedding = LogisticRegression(random_state=123, penalty = 'l1', solver = 'saga')
scores = cross_val_score(clf_embedding, embedding, y, cv=5)
print("embed LG", np.mean(scores))

# filtered methylation
clf_orig = LogisticRegression(random_state=123, penalty = 'l1', solver = 'saga')
scores_orig = cross_val_score(clf_orig, scaled_methyl, y, cv=5)
print("orig LG",np.mean(scores_orig))


# Random Forest
rf_emb = RandomForestClassifier(max_depth=2, random_state=0)
scores_rf_emb = cross_val_score(rf_emb, embedding, y, cv=5)
print("embed RF",np.mean(scores_rf_emb))

# Random Forest
rf_orig = RandomForestClassifier(max_depth=2, random_state=0)
scores_rf_orig = cross_val_score(rf_orig, scaled_methyl, y, cv=5)
print("orig RF", np.mean(scores_rf_orig))

# export the embedding
embedding_df = pd.DataFrame(embedding, index = methy.index)
print(embedding_df.shape)
#embedding_df.to_csv('/Users/smasarone/Documents/Omics_datasets_integration/BRCA_experiments/BRCA_embeddings/methy_BRCA_embedding_survival_norm.csv')
