"""Unit tests for model networks and loss functions."""

import tensorflow.compat.v1 as tf
import networks


class NetworksTest(tf.test.TestCase):

  def test_gestational_age_regression_model(self):
    with tf.Graph().as_default():
      # Create a tensor with dimensions representing batch size, sequence
      # length, height, width, image channels. Note: The sizes for unit testing
      # are smaller than the real values to reduce resource usage. Actual sizes
      # are [8, 24, 432, 576, 1] as discussed in Supplementary Methods of the
      # publication.
      video_clips = tf.zeros([2, 10, 24, 32, 1])
      ages, variances = networks.gestational_age_regression_model(
          video_clips, is_training=False)
      with self.test_session() as sess:
        sess.run(tf.global_variables_initializer())
        ages_, variances_ = sess.run((ages, variances))
        # Expect two output values, one for each clip in the batch.
        self.assertEqual(ages_.shape, (2, 1))
        self.assertEqual(variances_.shape, (2, 1))

  def test_fetal_malpresentation_classification_model(self):
    with tf.Graph().as_default():
      # Actual dimensions are [8, 100, 240, 320, 1] as discussed in
      # Supplementary Methods of the publication.
      video_clips = tf.zeros([2, 10, 24, 32, 1])
      classification_output = (
          networks.fetal_malpresentation_classification_model(
              video_clips, is_training=False))
      with self.test_session() as sess:
        sess.run(tf.global_variables_initializer())
        classification_output_ = sess.run(classification_output)
        # Expect two output values, one for each clip in the batch.
        self.assertEqual(classification_output_.shape, (2, 1))

  def test_gestational_age_loss_function(self):
    with tf.Graph().as_default():
      video_clips = tf.zeros([2, 10, 24, 32, 1])
      labels = tf.zeros([2, 1])
      ages, variances = networks.gestational_age_regression_model(
          video_clips, is_training=True)
      loss = networks.gestational_age_loss_function(ages, variances, labels)
      with self.test_session() as sess:
        sess.run(tf.global_variables_initializer())
        loss_ = sess.run(loss)
        # Expect a scalar output.
        self.assertEqual(loss_.shape, ())

  def test_malpresentation_model_loss_function(self):
    with tf.Graph().as_default():
      video_clips = tf.zeros([2, 10, 24, 32, 1])
      labels = tf.zeros([2, 1])
      logits = networks.fetal_malpresentation_classification_model(
          video_clips, is_training=True)
      loss = networks.fetal_malpresentation_loss_function(logits, labels)
      with self.test_session() as sess:
        sess.run(tf.global_variables_initializer())
        loss_ = sess.run(loss)
        # Expect a scalar output.
        self.assertEqual(loss_.shape, ())

if __name__ == '__main__':
  tf.test.main()
