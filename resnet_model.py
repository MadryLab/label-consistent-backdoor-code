"""
This model is adapted from the resnet-cifar10 repo
"""
# From https://github.com/tensorflow/models/blob/master/resnet/resnet_model.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple

import tensorflow as tf

HParams = namedtuple('HParams',
                     ['num_classes',
                      'image_size',
                      'resnet_size'])

default_hps = HParams(num_classes=10,
                      image_size=32,
                      resnet_size=32)

def make_data_augmentation_fn(is_training, padding=4, flip=True, standardization=True, hps=default_hps):
    def augmentation(image, label):
        image.shape.assert_is_compatible_with(
            [hps.image_size, hps.image_size, 3])
        if is_training:
            image = tf.image.resize_image_with_crop_or_pad(
                    image, hps.image_size + padding, hps.image_size + padding)
            image = tf.random_crop(image, [hps.image_size, hps.image_size, 3])
            if flip:
                image = tf.image.random_flip_left_right(image)
        # Always standardize whether training or not (if on)
        if standardization:
            image = tf.image.per_image_standardization(image)
        return image, label
    return augmentation

def choose(selector, matrix):
    selector = tf.reshape(selector, (-1, 1))
    ordinals = tf.reshape(tf.range(tf.shape(matrix, out_type=tf.int64)[0]), (-1, 1))
    idx = tf.stack([selector, ordinals], axis=-1)
    return tf.squeeze(tf.gather_nd(tf.transpose(matrix), idx))

class ResNetModel(object):
    """ResNet model."""

    def __init__(self, x_input, y_input, *, random_seed=None, hps=default_hps):
        """ResNet constructor.

        Args:
          hps: Hyperparameters.
          mode: One of 'train' and 'eval'.
        """
        if random_seed is not None:
            tf.set_random_seed(random_seed)
        x_input.shape.assert_is_compatible_with(
            [None, hps.image_size, hps.image_size, 3])
        y_input.shape.assert_is_compatible_with(
            [None])
        assert x_input.dtype == tf.float32
        assert y_input.dtype == tf.int64
        if hps.resnet_size % 6 != 2:
            raise ValueError('resnet_size must be 6n + 2:', hps.resnet_size)

        self.x_input = x_input # tf.placeholder(tf.float32, shape=[None, hps.image_size, hps.image_size, 3])
        self.x_image = self.x_input
        # Convert to NCHW
        self.x_input_nchw = tf.transpose(self.x_input, [0, 3, 1, 2])
        self.y_input = y_input # tf.placeholder(tf.int64, shape=[None])
        # self.hps = hps
        self.is_training = tf.placeholder(tf.bool, shape=[])
        self.num_classes = hps.num_classes
        self.resnet_size = hps.resnet_size
        self._build_model()

    def _build_model(self):
        """Build the core model within the graph."""
        num_blocks = (self.resnet_size - 2) // 6

        filters = [16, 16, 32, 64]

        # Uncomment the following codes to use w28-10 wide residual network.
        # It is more memory efficient than very deep residual network and has
        # comparably good performance.
        # https://arxiv.org/pdf/1605.07146v1.pdf
        # filters = [16, 160, 320, 640]
        # Update hps.num_residual_units to 9

        inputs = conv2d_fixed_padding(
            inputs=self.x_input_nchw, filters=filters[0], kernel_size=3, strides=1)
        inputs = tf.identity(inputs, 'initial_conv')

        inputs = block_layer(
            inputs=inputs, filters=filters[1], block_fn=building_block,
            blocks=num_blocks, strides=1, is_training=self.is_training,
            name='block_layer1')
        inputs = block_layer(
            inputs=inputs, filters=filters[2], block_fn=building_block,
            blocks=num_blocks, strides=2, is_training=self.is_training,
            name='block_layer2')
        inputs = block_layer(
            inputs=inputs, filters=filters[3], block_fn=building_block,
            blocks=num_blocks, strides=2, is_training=self.is_training,
            name='block_layer3')

        inputs = batch_norm_relu(inputs, self.is_training)
        # Workaround because there is currently no NCHW average pooling on CPU
        if tf.test.is_built_with_cuda():
            inputs = tf.layers.average_pooling2d(
                inputs=inputs, pool_size=8, strides=1, padding='VALID',
                data_format='channels_first')
        else:
            inputs = tf.transpose(inputs, [0, 2, 3, 1])
            inputs = tf.layers.average_pooling2d(
                inputs=inputs, pool_size=8, strides=1, padding='VALID',
                data_format='channels_last')
            inputs = tf.transpose(inputs, [0, 3, 1, 2])
        inputs = tf.identity(inputs, 'final_avg_pool')
        inputs = tf.reshape(inputs, [-1, 64])
        inputs = tf.layers.dense(inputs=inputs, units=self.num_classes)
        logits = tf.identity(inputs, 'final_dense')

        softmax = tf.nn.softmax(logits)

        self.predictions = tf.argmax(logits, 1)
        self.correct_prediction = tf.equal(self.predictions, self.y_input)
        self.num_correct = tf.reduce_sum(
            tf.cast(self.correct_prediction, tf.int64))
        self.accuracy = tf.reduce_mean(
            tf.cast(self.correct_prediction, tf.float32))
        self.y_xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=self.y_input)
        self.xent = tf.reduce_mean(self.y_xent, name='xent')
        self.weight_decay_loss = self._decay()

        self.confidence_in_correct = choose(self.y_input, softmax)
        self.confidence_in_prediction = choose(self.predictions, softmax)

    def _decay(self):
        """L2 weight decay loss."""
        costs = []
        for var in tf.trainable_variables():
            costs.append(tf.nn.l2_loss(var))
        return tf.add_n(costs)


def batch_norm_relu(inputs, is_training):
    """Performs a batch normalization followed by a ReLU."""
    _BATCH_NORM_DECAY = 0.997
    _BATCH_NORM_EPSILON = 1e-5
    axis = 1
    # Workaround because there is currently no NCHW fused BN on CPU
    if not tf.test.is_built_with_cuda():
        inputs = tf.transpose(inputs, [0, 2, 3, 1])
        axis = 3
    inputs = tf.layers.batch_normalization(inputs=inputs, axis=axis, momentum=_BATCH_NORM_DECAY,
                                           epsilon=_BATCH_NORM_EPSILON, center=True,
                                           scale=True, training=is_training, fused=True)
    inputs = tf.nn.relu(inputs)
    if not tf.test.is_built_with_cuda():
        inputs = tf.transpose(inputs, [0, 3, 1, 2])
    return inputs


def fixed_padding(inputs, kernel_size):
    """Pads the input along the spatial dimensions independently of input size.

    Args:
      inputs: A tensor of size [batch, channels, height_in, width_in]
      kernel_size: The kernel to be used in the conv2d or max_pool2d operation.
                   Should be a positive integer.

    Returns:
      A tensor with the same format as the input with the data either intact
      (if kernel_size == 1) or padded (if kernel_size > 1).
    """
    pad_total = kernel_size - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg

    padded_inputs = tf.pad(inputs,
                           [[0, 0], [0, 0], [pad_beg, pad_end], [pad_beg, pad_end]])
    return padded_inputs


def conv2d_fixed_padding(inputs, filters, kernel_size, strides):
    """Strided 2-D convolution with explicit padding.

    The padding is consistent and is based only on `kernel_size`, not on the
    dimensions of `inputs` (as opposed to using `tf.layers.conv2d` alone).
    """
    if strides > 1:
        inputs = fixed_padding(inputs, kernel_size)
    data_format = 'channels_first'
    # Workaround because there is currently no NCHW conv2d on CPU
    if not tf.test.is_built_with_cuda():
        inputs = tf.transpose(inputs, [0, 2, 3, 1])
        data_format = 'channels_last'
    inputs = tf.layers.conv2d(
        inputs=inputs,
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=('SAME' if strides == 1 else 'VALID'),
        use_bias=False,
        kernel_initializer=tf.variance_scaling_initializer(),
        data_format=data_format)
    if not tf.test.is_built_with_cuda():
        inputs = tf.transpose(inputs, [0, 3, 1, 2])
    return inputs


def building_block(inputs, filters, is_training, projection_shortcut, strides):
    """Standard building block for residual networks (v2).

    Args:
      inputs: A tensor of size [batch, channels, height_in, width_in].
      filters: The number of filters for the convolutions.
      is_training: A Boolean for whether the model is in training or inference
                   mode. Needed for batch normalization.
      projection_shortcut: The function to use for projection shortcuts
                           (typically a 1x1 convolution when downsampling the
                           input).
      strides: The block's stride. If greater than 1, this block will ultimately
               downsample the input.

    Returns:
      The output tensor of the block.
    """
    shortcut = inputs
    inputs = batch_norm_relu(inputs, is_training)

    # The projection shortcut should come after the first batch norm and ReLU
    # since it performs a 1x1 convolution.
    if projection_shortcut is not None:
        shortcut = projection_shortcut(inputs)

    inputs = conv2d_fixed_padding(inputs=inputs, filters=filters, kernel_size=3,
                                  strides=strides)
    inputs = batch_norm_relu(inputs, is_training)
    inputs = conv2d_fixed_padding(inputs=inputs, filters=filters, kernel_size=3,
                                  strides=1)
    return inputs + shortcut


def block_layer(inputs, filters, block_fn, blocks, strides, is_training, name):
    """Creates one layer of blocks for the ResNet model.

    Args:
      inputs: A tensor of size [batch, channels, height_in, width_in].
      filters: The number of filters for the first convolution of the layer.
      block_fn: The block to use within the model, either `building_block` or
                `bottleneck_block`.
      blocks: The number of blocks contained in the layer.
      strides: The stride to use for the first convolution of the layer. If
               greater than 1, this layer will ultimately downsample the input.
      is_training: Either True or False, whether we are currently training the
                   model. Needed for batch norm.
      name: A string name for the tensor output of the block layer.

    Returns:
      The output tensor of the block layer.
    """
    # Bottleneck blocks end with 4x the number of filters as they start with
    #filters_out = 4 * filters if block_fn is bottleneck_block else filters
    filters_out = filters

    def projection_shortcut(inputs):
        return conv2d_fixed_padding(inputs=inputs, filters=filters_out,
                                    kernel_size=1, strides=strides)

    # Only the first block per block_layer uses projection_shortcut and strides
    inputs = block_fn(inputs, filters, is_training, projection_shortcut,
                      strides)
    for _ in range(1, blocks):
        inputs = block_fn(inputs, filters, is_training, None, 1)
    return tf.identity(inputs, name)
