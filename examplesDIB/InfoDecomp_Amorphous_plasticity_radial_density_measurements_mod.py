#@title imports
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.stats import norm
from matplotlib import cm

import scipy.ndimage as nim
from matplotlib.gridspec import GridSpec
default_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
from sklearn import svm
from matplotlib.image import NonUniformImage

from DIB.utils import bhattacharyya_dist_mat
from DIB.models import DistributedIBNet, InfoPerFeatureCallback, InfoBottleneckAnnealingCallback


'''
#@title DistributedIBNet + callbacks, copied from the github repo
class PositionalEncoding(tf.keras.layers.Layer):
  """Simple positional encoding layer, that appends to an input sinusoids of multiple frequencies.
  """
  def __init__(self, frequencies):
    super(PositionalEncoding, self).__init__()
    self.frequencies = frequencies

  def build(self, input_shape):
    return

  def call(self, inputs):  # Defines the computation from inputs to outputs
    return tf.concat([inputs] + [tf.math.sin(frequency*inputs) for frequency in self.frequencies], -1)
'''


class StashEmbeddingsCallback(tf.keras.callbacks.Callback):

  def __init__(self,
               save_frequency,
               x_in,
               save_start=0):
    super(StashEmbeddingsCallback, self).__init__()
    self.save_frequency = save_frequency
    self.x_in = x_in
    self.mus_for_later = []
    self.logvars_for_later = []
    self.save_start = save_start

  def on_epoch_end(self, epoch, logs=None):
    if (epoch > self.save_start) and ((epoch % self.save_frequency) == 0):
      features_split = tf.split(self.x_in, self.model.feature_dimensionalities, axis=-1)
      for feature_ind in range(self.model.number_features):
        emb_mus, emb_logvars = tf.split(self.model.feature_encoders[feature_ind](features_split[feature_ind]), 2, axis=-1)
        self.mus_for_later.append(emb_mus)
        self.logvars_for_later.append(emb_logvars)


################# RUN PARAMS FOR ALL ######################################
number_radial_densities_per_type = 50

thickness_sigma_of_radial_bands = 0.5

SAFETY_EPS = 1e-10
radius_min, radius_max = [0.5, 4]




## These are the functions that take the raw data (in the form of neighborhoods
## of particles) and convert to the format for the distributed IB setup
radii = tf.cast(tf.expand_dims(tf.linspace(radius_min, radius_max, number_radial_densities_per_type+1), 0), tf.float32)
dradii = tf.ones(number_radial_densities_per_type)*(radius_max-radius_min)/float(number_radial_densities_per_type)*thickness_sigma_of_radial_bands
radii = (radii[:, 1:] + radii[:, :-1])/2.
@tf.function(experimental_relax_shapes=True)
def convert_to_radial_densities(particle_positions, types):
  type1_inds = tf.where(types==1)
  type2_inds = tf.where(types==2)
  particle_radii = tf.sqrt(tf.reduce_sum(tf.square(particle_positions), -1) + SAFETY_EPS)

  type1_densities = tf.reduce_sum(tf.exp(-(tf.gather(particle_radii, type1_inds)-radii)**2/tf.square(dradii)/2.), 0)
  type2_densities = tf.reduce_sum(tf.exp(-(tf.gather(particle_radii, type2_inds)-radii)**2/tf.square(dradii)/2.), 0)

  densities = tf.concat([type1_densities, type2_densities], -1)
  return densities

def convert_to_per_particle_feature_set(particle_positions, types, number_to_use=60):
  type_one_hots = tf.one_hot(tf.cast(types, tf.int32)-1, 2)
  particle_radii = tf.sqrt(tf.reduce_sum(tf.square(particle_positions), -1, keepdims=True) + SAFETY_EPS)
  unit_vectors = particle_positions / particle_radii
  # Without changing the amount of information, preprocess the positions a few different
  # ways to help the model get off the ground
  features = tf.concat([particle_positions, particle_positions**2,
                        particle_radii,
                        tf.math.log(particle_radii+1e-3),
                        tf.math.log(particle_positions**2 + 1e-3),
                        unit_vectors, type_one_hots], -1)

  features = np.float32(features)
  if number_to_use>0:
    sorted_positions = tf.argsort(tf.squeeze(particle_radii))

    features = np.squeeze(features[sorted_positions])
    features = features[:number_to_use]

  return features


# Change the following to True if you'd like to stash embeddings along the way
# in order to visualize the distinguishability between density values for each
# measurement; takes extra time and memory but they are pretty cool
stash_embeddings_for_distinguishability_matrices = True
# If you'd like the arrays and plots to be saved, switch to True
save_outputs = True
expt_outdir = './'

protocol = 'GradualQuench' # 'RapidQuench']:
print('Beginning', protocol)

# The data is a bunch of positions along with an indicator (type) of whether each particle is small or large
pkl_dict = np.load(f'{protocol}.npz', allow_pickle=True)
train_particle_positions = pkl_dict['train_particle_positions']
train_types = pkl_dict['train_types']
train_is_loci = pkl_dict['train_is_loci']
val_particle_positions = pkl_dict['val_particle_positions']
val_types = pkl_dict['val_types']
val_is_loci = pkl_dict['val_is_loci']

# Convert the positions into radial density measurements
all_train_densities = []
for train_ind in range(train_types.shape[0]):
  all_train_densities.append(convert_to_radial_densities(train_particle_positions[train_ind], train_types[train_ind]))
all_val_densities = []
for val_ind in range(val_types.shape[0]):
  all_val_densities.append(convert_to_radial_densities(val_particle_positions[val_ind], val_types[val_ind]))

all_train_densities = np.stack(all_train_densities)
all_val_densities = np.stack(all_val_densities)
print('Loaded train/val data.  Array shapes:', all_train_densities.shape, all_val_densities.shape)

all_train_loci = np.squeeze(np.concatenate(train_is_loci))
all_val_loci = np.squeeze(np.concatenate(val_is_loci))

X = all_train_densities
valX = all_val_densities
Y = all_train_loci
valY = all_val_loci

# Create a tf.data.Dataset from the validation data for estimating the information
# over the course of training
tf_dataset_validation = tf.data.Dataset.from_tensor_slices((valX, valY))

# We'll use the high level DistributedIBNet, which just throws an MLP at each
# feature.  All we need is the dimension of each feature (1) and the architectures
number_features = number_radial_densities_per_type*2
feature_dimensionalities = [1]*number_features
feature_encoder_architecture = [128]*2
integration_network_architecture = [256]*3
output_dimensionality = 1
use_positional_encoding = False
activation_fn = 'tanh'
feature_embedding_dimension = 32

model = DistributedIBNet(feature_dimensionalities,
    feature_encoder_architecture,
    integration_network_architecture,
    output_dimensionality,
    use_positional_encoding=use_positional_encoding,
    activation_fn=activation_fn,
    feature_embedding_dimension=feature_embedding_dimension)

number_pretraining_epochs = 0
number_annealing_epochs = 50
beta_start = 1e-6
beta_end = 1.
lr = 1e-4
batch_size = 256
model.compile(optimizer=tf.keras.optimizers.Adam(lr),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

beta_annealing_callback = InfoBottleneckAnnealingCallback(beta_start,
                                                          beta_end,
                                                          number_pretraining_epochs,
                                                          number_annealing_epochs)
mi_save_frequency = 1
info_per_feature_callback = InfoPerFeatureCallback(mi_save_frequency,
                                                  tf_dataset_validation,
                                                  info_bound_batch_size=256,#1024,
                                                  info_bound_number_batches=8)
callbacks = [beta_annealing_callback,
             info_per_feature_callback]
if stash_embeddings_for_distinguishability_matrices:
  embeddings_save_frequency = 2
  embeddings_callback = StashEmbeddingsCallback(embeddings_save_frequency,
                                                valX, save_start=number_annealing_epochs//4)
  callbacks.append(embeddings_callback)


history = model.fit(x=X,
                    y=Y,
                    shuffle=True,
                    batch_size=batch_size,
                    epochs=number_pretraining_epochs+number_annealing_epochs,
                    callbacks=callbacks,
                    verbose=True,
                    validation_data=(valX, valY))


##############################################################################
## Plot the info allocation over time (Fig 2 in the manuscript)
##############################################################################
print('*'*100)
print('Finished training, painting the information allocation heat map.')
print('*'*100)

# We don't care about the finished model, we care about what happened along
# the way.  Retrieve various things from the callbacks
info_per_feature_bounds = np.float32(info_per_feature_callback.bounds)
info_per_feature_bounds = np.reshape(info_per_feature_bounds, [-1, number_features, 2])/np.log(2)

beta_series = np.float32(history.history['beta'])
kl_series = np.stack([history.history[f'val_KL{feature_ind}'] for feature_ind in range(number_features)], -1)
val_bce_series = np.float32(history.history['val_loss'])
acc_series = np.float32(history.history['val_accuracy'])
## Get the original bce loss without the KL term
val_bce_series -= beta_series * np.sum(kl_series, axis=-1)
val_bce_series /= np.log(2)
if save_outputs:
  np.savez(f'history_{protocol}.npz',
          betas=beta_series,
          validation_bce=val_bce_series,
          acc_validation=acc_series,
          info_per_feature_bounds=info_per_feature_bounds)
############################################################################################
if stash_embeddings_for_distinguishability_matrices:
  emb_mu_history = np.reshape(embeddings_callback.mus_for_later, [-1, number_features, valX.shape[0], feature_embedding_dimension])
  emb_logvar_history = np.reshape(embeddings_callback.logvars_for_later, [-1, number_features, valX.shape[0], feature_embedding_dimension])
  if save_outputs:
    np.save(os.path.join(expt_outdir, 'embs_mus_history.npy'), emb_mu_history)
    np.save(os.path.join(expt_outdir, 'embs_logvars_history.npy'), emb_logvar_history)

## Plot stuff
info_in_plot_lims = [0, None]
info_per_feature_plot_lims = [0, 2]
accuracy_plot_lims = [0.8, None]

smoothing_sigma = 0.1
info_in_parts = nim.filters.gaussian_filter1d(np.mean(info_per_feature_bounds, axis=-1), smoothing_sigma, axis=0)
info_in_full = np.sum(info_in_parts, axis=-1)
validation_bce_series = nim.filters.gaussian_filter1d(val_bce_series, smoothing_sigma)
acc_validation_series = nim.filters.gaussian_filter1d(acc_series, smoothing_sigma)

####################################################### Create the info matrix
g_r_AA = np.load(f'g_r_AA_{protocol}.npy')
g_r_AB = np.load(f'g_r_AB_{protocol}.npy')
g_r_bins = np.load('g_r_bins.npy')

info_series = np.mean(info_per_feature_bounds, axis=-1)

# Smooth the data from the optimization run a little
smoothing_sigma_info_parts = 1
smoothing_sigma_info_full = 0.5
smoothing_sigma_outs = 2.5
info_in_parts = nim.filters.gaussian_filter1d(info_series, smoothing_sigma_info_parts, axis=0)
info_in_full = nim.filters.gaussian_filter1d(np.sum(info_series, axis=-1), smoothing_sigma_info_full)

info_total_limit = 30
info_in_plot_lims = [0, info_total_limit]
info_out_plot_lims = [0, 1]
info_color_lims = [0, 1.5]
acc_plot_lims = [0.5,1]
g_r_lims = [0, 2]
r_plot_lims = [radius_max, radius_min]
info_cmap = 'gist_heat_r'

horizontal_cutoff_index = np.where(info_in_full>info_total_limit)[0][-1]

horizontal_positions = info_in_full[horizontal_cutoff_index:]
vertical_positions = np.linspace(radius_min, radius_max, number_radial_densities_per_type)

infos = info_in_parts[horizontal_cutoff_index:]

########################################################## PLOT
if smoothing_sigma_outs > 0:
  validation_bce_series = nim.filters.gaussian_filter1d(val_bce_series, smoothing_sigma_outs)
  acc_validation_series = nim.filters.gaussian_filter1d(acc_series, smoothing_sigma_outs)
else:
  validation_bce_series = val_bce_series
  acc_validation_series = acc_series
entropy_y_validation = 1.
info_out = entropy_y_validation - validation_bce_series

fig = plt.figure(figsize=(10, 12))
gs = fig.add_gridspec(3, 2, height_ratios=(1, 1, 1), width_ratios=(8, 1),
                  left=0.1, right=0.9, bottom=0.1, top=0.9,
                  wspace=0.05, hspace=0.1)

ax = fig.add_subplot(gs[0, 0])
ax.plot(info_in_full, info_out[::mi_save_frequency], lw=4, color='k')
ax.set_ylim(info_out_plot_lims)
ax.set_xlim(info_in_plot_lims)
ax.set_xlabel("Info in (bits)", fontsize=15)
ax.set_ylabel("Info out (bits)", fontsize=15)

ax2 = ax.twinx()
ax2.plot(info_in_full, acc_validation_series[::mi_save_frequency], lw=2)
ax2.set_ylabel("ACC (validation)", fontsize=15, color='b')
ax2.set_ylim(acc_plot_lims)

###################################### Info allocation matrix

ax1 = fig.add_subplot(gs[1, 1])
ax1.plot(g_r_AA, g_r_bins[1:], 'k')
ax1.set_xlim(g_r_lims)
ax1.set_ylim(r_plot_lims)
plt.xticks([]); plt.yticks([])

ax1.set_xlabel('g_AA(r)', fontsize=15)

ax2 = fig.add_subplot(gs[1, 0])
im = NonUniformImage(ax2, interpolation='nearest', extent=(horizontal_positions.min(), horizontal_positions.max(), radius_max, radius_min),
                    cmap=info_cmap)
im.set_data(horizontal_positions[::-1],
            vertical_positions,
            infos[::-1, :number_radial_densities_per_type].T[::-1])
ax2.add_image(im)
ax2.set_xlim(info_in_plot_lims)
ax2.set_ylim(r_plot_lims[::-1])
ax2.set_yticks([])
im.set_clim(info_color_lims)
ax2.set_ylabel(f'Type A (small), radius {radius_max}<--{radius_min}', fontsize=15)

ax3 = fig.add_subplot(gs[2, 1])
ax3.plot(g_r_AB, g_r_bins[1:], 'k')
ax3.set_xlim(g_r_lims)
ax3.set_ylim(r_plot_lims)
plt.xticks([]); plt.yticks([])
ax3.set_xlabel('g_AB(r)', fontsize=15)

ax4 = fig.add_subplot(gs[2, 0])
im = NonUniformImage(ax4, interpolation='nearest', extent=(horizontal_positions.min(), horizontal_positions.max(), radius_max, radius_min),
                    cmap=info_cmap)
im.set_data(horizontal_positions[::-1],
            vertical_positions,
            infos[::-1, number_radial_densities_per_type:].T[::-1])
ax4.add_image(im)
ax4.set_xlim(info_in_plot_lims)
ax4.set_ylim(r_plot_lims[::-1])
ax4.set_yticks([])
im.set_clim(info_color_lims)
ax4.set_xlabel('Total information into model (bits)', fontsize=15)
ax4.set_ylabel(f'Type B (large), {radius_max}<--{radius_min}', fontsize=15)

plt.suptitle(protocol, fontsize=16, y=0.92)
if save_outputs:
  plt.savefig(f'info_decomp_{protocol}.png')
plt.show()


# In[17]:


if stash_embeddings_for_distinguishability_matrices:
  ##############################################################################
  ## Display the distinguishability matrices between feature values (Fig 3)
  ##############################################################################
  # Values of the total information utilized (in bits) for which to plot the dist mats
  desired_info_vals = [1, 2, 5, 10, 15, 20, 25, 30, 35, 40]

  smoothing_sigma_info_full = 0.5
  approx_info_in_full = nim.filters.gaussian_filter1d(np.sum(info_series, axis=-1), smoothing_sigma_info_full)

  # Get the timestep indices that most closely match the desired info vals
  info_dist_mat_inds, emb_dist_mat_inds = [[], []]
  for val in desired_info_vals:
    forward_info_ind = np.argmin(np.abs(approx_info_in_full-val))
    info_dist_mat_inds.append(forward_info_ind)
    # Count backward because we started saving embeddings after a delay
    backward_emb_ind = (forward_info_ind - number_annealing_epochs) * (mi_save_frequency / embeddings_save_frequency)
    emb_dist_mat_inds.append(int(backward_emb_ind))
  emb_dist_mat_inds = [val for val in emb_dist_mat_inds if np.abs(val)<emb_mu_history.shape[0]]
  dist_mat_info_in_vals = approx_info_in_full[info_dist_mat_inds]

  for feature_ind in range(number_features):
    dist_mat_cmap = 'Blues_r'
    number_samples_from_validation_set = 256
    inches_per_subplot = 3

    # Don't worry about plotting it if there's no info even at the high end
    if info_series[info_dist_mat_inds[-1], feature_ind] < 0.5:
      continue
    plt.figure(figsize=(inches_per_subplot*len(desired_info_vals), inches_per_subplot))
    for plt_ind, (emb_dist_mat_ind, dist_mat_info_in_val) in enumerate(zip(emb_dist_mat_inds, dist_mat_info_in_vals)):
      plt.subplot(1, len(emb_dist_mat_inds), plt_ind+1)
      orig_X_vals = valX[:number_samples_from_validation_set, feature_ind]
      sorted_inds = np.argsort(orig_X_vals)
      mus_sorted = emb_mu_history[emb_dist_mat_ind, feature_ind, :number_samples_from_validation_set][sorted_inds]
      logvars_sorted = emb_logvar_history[emb_dist_mat_ind, feature_ind, :number_samples_from_validation_set][sorted_inds]

      bhat_dist_mat = bhattacharyya_dist_mat(mus_sorted,
                                            logvars_sorted,
                                            mus_sorted,
                                            logvars_sorted)
      bhat_coeff_mat = np.exp(-bhat_dist_mat)

      plt.imshow(bhat_coeff_mat, vmin=0, vmax=1, cmap=dist_mat_cmap)
      if not plt_ind:
        plt.ylabel(f'Type {"AB"[feature_ind//number_radial_densities_per_type]}, radius {radii[0, feature_ind%number_radial_densities_per_type]:.2f}', fontsize=15)
        plt.title(f'Total info in: {dist_mat_info_in_val:.1f}b', fontsize=15)
      else:
        plt.title(f'{dist_mat_info_in_val:.1f}b', fontsize=15)
      plt.xticks([]); plt.yticks([])

    if save_outputs:
      plt.savefig(os.path.join(expt_outdir, f'dist_mats_feature_{feature_ind}.png'))

    plt.show()


