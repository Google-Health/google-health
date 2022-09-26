"""Model network definitions and loss functions.

Defines networks and loss functions for gestational age and fetal
malpresentation models featured in the publication.
"""

import tensorflow.compat.v1 as tf
import tf_slim as slim

from tensorflow.contrib import rnn
from lstm_object_detection.lstm import lstm_cells
from nets.mobilenet import mobilenet_v2

N_LSTM_UNITS = 512
LSTM_FILTER_SIZE = (3, 3)
MOBILENET_DEPTH_MULTIPLIER = 1.0


def _base_network(video_clips, is_training):
  """Builds the base network used by both models.

  Extracts image features independently for each image in each video clip, using
  MobileNetV2. Then aggregates the image features for each video clip using LSTM
  units.

  Args:
    video_clips: Tensor containing a batch of video clips (image sequences).
      Dimensions: [batch_size, sequence_length, height, width, image_channels].
    is_training: Boolean value indicating whether the network graph is to be
      used for training models.

  Returns:
    state_and_output_concat: Tensor containing LSTM state and output values
      corresponding to the final image in each video clip. Dimensions:
      [batch_size, feature_map_height, feature_map_width, 2 * N_LSTM_UNITS].
      feature_map_height and feature_map_width are determined by the spatial
      dimensions of the final feature map layer of the MobileNetV2 image feature
      extractor.
  """
  video_clips_shape = video_clips.get_shape().as_list()
  assert len(video_clips_shape) == 5
  n_batch, n_sequence, height, width, n_channels = tuple(video_clips_shape)
  # Convert gray scale to RGB for use with MobileNetV2 feature extractor.
  if n_channels == 1:
    video_clips = tf.image.grayscale_to_rgb(video_clips)
    n_channels = 3
  # Flatten batch and time dimensions, MobileNetV2 extracts features for each
  # image frame independently.
  video_clips = tf.reshape(
      video_clips, [n_batch * n_sequence, height, width, n_channels])

  # Weight decay is set to zero in the training scope, but may be overridden
  # by training algorithms.
  arg_scope = mobilenet_v2.training_scope(
      is_training=is_training, weight_decay=0.0)
  with slim.arg_scope(arg_scope):
    with tf.variable_scope('ImageFeatureExtractor'):
      image_feature_maps, _ = mobilenet_v2.mobilenet_base(
          video_clips,
          depth_multiplier=MOBILENET_DEPTH_MULTIPLIER,
          use_explicit_padding=True)

  with tf.variable_scope('LSTM') as lstm_scope:
    _, maps_height, maps_width, maps_n_channels = tuple(
        image_feature_maps.get_shape().as_list())
    # Reshape the feature maps to recover sequence structure.
    maps_unrolled = tf.reshape(image_feature_maps, [
        n_batch, n_sequence, maps_height, maps_width, maps_n_channels
    ])
    feature_maps_sequence = tf.unstack(maps_unrolled, axis=1)
    lstm_cell = lstm_cells.GroupedConvLSTMCell(
        filter_size=LSTM_FILTER_SIZE,
        output_size=(maps_height, maps_width),
        num_units=N_LSTM_UNITS,
        is_training=is_training,
        activation=tf.nn.relu6,
        clip_state=True,
        output_bottleneck=True,
        visualize_gates=False)
    current_states_list = lstm_cell.init_state(
        state_name='lstm_state', batch_size=n_batch, dtype=tf.float32)
    init_state = rnn.LSTMStateTuple(*current_states_list)

    # Feed 2-D feature map sequences into recurrent LSTM cell.
    _, state_and_output = tf.nn.static_rnn(
        cell=lstm_cell,
        inputs=feature_maps_sequence,
        initial_state=init_state,
        scope=lstm_scope)
    # The state_and_output contains LSTM state and output for the last
    # image in the sequence.
    state_and_output_concat = tf.concat(state_and_output, -1)
    return state_and_output_concat


def _average_pool(feature_map):
  feature_map_shape = feature_map.get_shape()
  input_rank = feature_map_shape.ndims
  n_batch = feature_map_shape.as_list()[0]
  return tf.reshape(
      tf.reduce_mean(feature_map, axis=list(range(2, input_rank - 1))),
      [n_batch, -1])


def gestational_age_regression_model(video_clips, is_training):
  """Model network for gestational age regression model."""
  lstm_state_and_output = _base_network(video_clips, is_training)
  spatially_averaged_features = _average_pool(lstm_state_and_output)
  age_output = tf.layers.dense(spatially_averaged_features, units=1)
  variance_output = tf.layers.dense(spatially_averaged_features, units=1)
  # Soft plus unit ensures variances are always positive, and a small positive
  # value is added to prevent division by zero or small noise values in the loss
  # function.
  variance_output = 1e-6 + tf.math.softplus(variance_output)
  return age_output, variance_output


def gestational_age_loss_function(predicted_ages, predicted_variances, labels):
  """Training loss function for gestational age regression model."""
  squared_errors = (labels - predicted_ages)**2
  scaled_errors = tf.math.divide(
      squared_errors, predicted_variances) + tf.math.log(predicted_variances)
  return tf.math.reduce_mean(0.5 * scaled_errors)


def fetal_malpresentation_classification_model(video_clips, is_training):
  """Model network for fetal malpresentation classification model."""
  lstm_state_and_output = _base_network(video_clips, is_training)
  spatially_averaged_features = _average_pool(lstm_state_and_output)
  # During training, the final sigmoid activation is applied by the loss
  # function.
  malpresentation_output = tf.layers.dense(
      spatially_averaged_features, units=1,
      activation=None if is_training else tf.nn.sigmoid)
  return malpresentation_output


def fetal_malpresentation_loss_function(logits, labels):
  """Training loss function for fetal malpresentation classification model."""
  per_instance_loss = tf.nn.sigmoid_cross_entropy_with_logits(
      labels=labels, logits=tf.reshape(logits, tf.shape(labels)))
  # Instance weights are set to default value of 1.0.
  return tf.losses.compute_weighted_loss(per_instance_loss)