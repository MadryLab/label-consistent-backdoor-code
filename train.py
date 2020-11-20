"""
Trains a model, saving checkpoints and tensorboard summaries along the way.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import json
import shutil
import os

import numpy as np
import tensorflow as tf

from eval_helper import EvalHelper
from resnet_model import ResNetModel, make_data_augmentation_fn

# load configuration: first load the base config, and then update using the
# job_parameters, if any
with open('config.json', 'r') as base_config_file:
    config = json.load(base_config_file)
if os.path.exists('job_parameters.json'):
    with open('job_parameters.json', 'r') as job_parameters_file:
        job_parameters = json.load(job_parameters_file)
    # make sure we didn't e.g. make some typo
    for k in job_parameters.keys():
        if k not in config.keys():
            print("{} config not in base config file!".format(k))
        # assert k in config.keys()
    config.update(job_parameters)

# Setting up training parameters
tf.set_random_seed(config['random_seed'])
np.random.seed(config['random_seed'])

max_num_training_steps = config['max_num_training_steps']
num_output_steps = config['num_output_steps']
num_summary_steps = config['num_summary_steps']
num_checkpoint_steps = config['num_checkpoint_steps']

batch_size = config['training_batch_size']

# Setting up the data and the model
clean_train_images = np.load(config["clean_dataset_dir"] + "/train_images.npy").astype(np.float32)
clean_train_labels = np.load(config["clean_dataset_dir"] + "/train_labels.npy").astype(np.int64)
num_train_examples = len(clean_train_images)

clean_test_images = np.load(config["clean_dataset_dir"] + "/test_images.npy").astype(np.float32)
clean_test_labels = np.load(config["clean_dataset_dir"] + "/test_labels.npy").astype(np.int64)
num_test_examples = len(clean_test_images)

# We assume inputs are as follows
#   - train_{images,labels}.npy -- the x% poisoned dataset
#   - test_{images,labels}.npy -- trigger applied to all test images
#   - poisoned_train_indices.npy -- which indices were poisoned
#   - train_no_trigger_{images,labels}.npy -- the x% poisoned dataset, but without any triggers applied
poisoned_train_images = np.load(config["already_poisoned_dataset_dir"] + "/train_images.npy").astype(np.float32)
poisoned_train_labels = np.load(config["already_poisoned_dataset_dir"] + "/train_labels.npy").astype(np.int64)
poisoned_test_images = np.load(config["already_poisoned_dataset_dir"] + "/test_images.npy").astype(np.float32)
poisoned_test_labels = np.load(config["already_poisoned_dataset_dir"] + "/test_labels.npy").astype(np.int64)

poisoned_train_indices = np.load(config["already_poisoned_dataset_dir"] + "/poisoned_train_indices.npy")
if len(poisoned_train_indices) > 0:
    poisoned_only_train_images = poisoned_train_images[poisoned_train_indices]
    poisoned_only_train_labels = poisoned_train_labels[poisoned_train_indices]
    poisoned_no_trigger_train_images = np.load(config["already_poisoned_dataset_dir"] + "/train_no_trigger_images.npy").astype(np.float32)
    # These are identical to the training labels
    poisoned_no_trigger_train_labels = np.load(config["already_poisoned_dataset_dir"] + "/train_labels.npy").astype(np.int64)
    poisoned_no_trigger_train_images = poisoned_no_trigger_train_images[poisoned_train_indices]
    poisoned_no_trigger_train_labels = poisoned_no_trigger_train_labels[poisoned_train_indices]

def prepare_dataset(images, labels):
    images_placeholder = tf.placeholder(tf.float32, images.shape)
    labels_placeholder = tf.placeholder(tf.int64, labels.shape)
    dataset = tf.contrib.data.Dataset.from_tensor_slices((images_placeholder, labels_placeholder))
    dataset = dataset.shuffle(buffer_size=10000, seed=config['random_seed']).repeat()

    if config['augment_dataset']:
        dataset = dataset.map(
            make_data_augmentation_fn(
                standardization=config['augment_standardization'],
                flip=config['augment_flip'],
                padding=config['augment_padding'],
                is_training=True))

    dataset = dataset.batch(batch_size)
    iterator = dataset.make_initializable_iterator()
    return (images_placeholder, labels_placeholder), dataset, iterator

clean_placeholder, clean_train_dataset_batched, clean_training_iterator = prepare_dataset(clean_train_images, clean_train_labels)
poisoned_placeholder, _, poisoned_training_iterator = prepare_dataset(poisoned_train_images, poisoned_train_labels)
if len(poisoned_train_indices) > 0:
    poisoned_only_placeholder, _, poisoned_only_training_iterator = prepare_dataset(poisoned_only_train_images, poisoned_only_train_labels)
    poisoned_no_trigger_placeholder, _, poisoned_no_trigger_training_iterator = prepare_dataset(poisoned_no_trigger_train_images, poisoned_no_trigger_train_labels)

iterator_handle = tf.placeholder(tf.string, shape=[])
input_iterator = tf.contrib.data.Iterator.from_string_handle(iterator_handle,
                                                             clean_train_dataset_batched.output_types,
                                                             clean_train_dataset_batched.output_shapes)
x_input, y_input = input_iterator.get_next()

global_step = tf.contrib.framework.get_or_create_global_step()

# Choose model and set up optimizer
model = ResNetModel(x_input, y_input, random_seed=config['random_seed'])

weight_decay = 0.0002
boundaries = config['learning_rate_boundaries']
values = config['learning_rates']
learning_rate = tf.train.piecewise_constant(
    tf.cast(global_step, tf.int32),
    boundaries,
    values)
momentum = 0.9
total_loss = model.xent + weight_decay * model.weight_decay_loss
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
opt = tf.train.MomentumOptimizer(
    learning_rate,
    momentum,
)
with tf.control_dependencies(update_ops):
    train_step = opt.minimize(total_loss, global_step=global_step)

# Setting up the Tensorboard and checkpoint outputs
model_dir = config['model_dir']
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

saver = tf.train.Saver(max_to_keep=3)
merged_summaries = tf.summary.merge([
    tf.summary.scalar('accuracy poison train', model.accuracy),
    tf.summary.scalar('xent poison train', model.xent / batch_size),
    tf.summary.image('images poison train', model.x_image),
    tf.summary.histogram('conf in y_input', model.confidence_in_correct),
    tf.summary.histogram('conf in y_pred', model.confidence_in_prediction),
])
clean_histogram = tf.summary.histogram('conf in clean', model.confidence_in_correct)
poison_only_merged_summaries = tf.summary.merge([
    tf.summary.scalar('accuracy poison only train', model.accuracy),
    tf.summary.scalar('xent poison only train', model.xent / batch_size), # NB shouldn't divide like this
    tf.summary.image('images poison only train', model.x_image),
    tf.summary.histogram('conf in poisoned only', model.confidence_in_correct),
])
poison_no_trigger_merged_summaries = tf.summary.merge([
    tf.summary.scalar('accuracy poison train (no trigger)', model.accuracy),
    tf.summary.scalar('xent poison train (no trigger)', model.xent / batch_size), # NB shouldn't divide like this
    tf.summary.image('images poison train (no trigger)', model.x_image),
    tf.summary.histogram('conf in poisoned (no trigger)', model.confidence_in_correct),
])

shutil.copy('config.json', model_dir)

with tf.Session() as sess:
    # Initialize the summary writer, global variables, and our time counter.
    summary_writer = tf.summary.FileWriter(model_dir, sess.graph)
    sess.run(tf.global_variables_initializer())

    sess.run(clean_training_iterator.initializer,
             feed_dict={clean_placeholder[0]: clean_train_images,
                        clean_placeholder[1]: clean_train_labels})
    sess.run(poisoned_training_iterator.initializer,
             feed_dict={poisoned_placeholder[0]: poisoned_train_images,
                        poisoned_placeholder[1]: poisoned_train_labels})
    if len(poisoned_train_indices) > 0:
        sess.run(poisoned_only_training_iterator.initializer,
                 feed_dict={poisoned_only_placeholder[0]: poisoned_only_train_images,
                            poisoned_only_placeholder[1]: poisoned_only_train_labels})
        sess.run(poisoned_no_trigger_training_iterator.initializer,
                 feed_dict={poisoned_no_trigger_placeholder[0]: poisoned_no_trigger_train_images,
                            poisoned_no_trigger_placeholder[1]: poisoned_no_trigger_train_labels})

    clean_training_handle = sess.run(clean_training_iterator.string_handle())
    poisoned_training_handle = sess.run(poisoned_training_iterator.string_handle())
    if len(poisoned_train_indices) > 0:
        poisoned_only_training_handle = sess.run(poisoned_only_training_iterator.string_handle())
        poisoned_no_trigger_training_handle = sess.run(poisoned_no_trigger_training_iterator.string_handle())

    evalHelper = EvalHelper(
        sess,
        {
            "clean_train": (clean_train_images, clean_train_labels),
            "poisoned_train": (poisoned_train_images, poisoned_train_labels),
            "poisoned_only_train": (poisoned_only_train_images, poisoned_only_train_labels),
            "poisoned_no_trigger_train": (poisoned_no_trigger_train_images, poisoned_no_trigger_train_labels),
            "clean_test": (clean_test_images, clean_test_labels),
            "poisoned_test": (poisoned_test_images, poisoned_test_labels),
        },
        iterator_handle
    )

    # Main training loop
    for ii in range(max_num_training_steps):
        clean_dict = {iterator_handle: clean_training_handle,
                      model.is_training: True}
        poison_dict = {iterator_handle: poisoned_training_handle,
                       model.is_training: True}

        # Output to stdout
        if ii % num_output_steps == 0:
            clean_acc = sess.run(model.accuracy, feed_dict=clean_dict)
            poison_acc = sess.run(model.accuracy, feed_dict=poison_dict)
            print('Step {}:    ({})'.format(ii, datetime.now()))
            print('    training clean accuracy {:.4}%'.format(clean_acc * 100))
            print('    training poison accuracy {:.4}%'.format(poison_acc * 100))

        # Tensorboard summaries
        if ii % num_summary_steps == 0:
            summary = sess.run(merged_summaries, feed_dict=poison_dict)
            summary_writer.add_summary(summary, global_step.eval(sess))
            summary_clean = sess.run(clean_histogram, feed_dict=clean_dict)
            summary_writer.add_summary(summary_clean, global_step.eval(sess))
            if len(poisoned_train_indices) > 0:
                poison_only_dict = {iterator_handle: poisoned_only_training_handle,
                                    model.is_training: True}
                poison_no_trigger_dict = {iterator_handle: poisoned_no_trigger_training_handle,
                                          model.is_training: True}
                summary_poison_only = sess.run(poison_only_merged_summaries, feed_dict=poison_only_dict)
                summary_writer.add_summary(summary_poison_only, global_step.eval(sess))
                summary_poison_no_trigger = sess.run(poison_no_trigger_merged_summaries, feed_dict=poison_no_trigger_dict)
                summary_writer.add_summary(summary_poison_no_trigger, global_step.eval(sess))

        # Write a checkpoint
        if ii % num_checkpoint_steps == 0:
            saver.save(sess,
                       os.path.join(model_dir, 'checkpoint'),
                       global_step=global_step)

        # Run an eval
        if (config['num_eval_steps'] > 0
                and ii % config['num_eval_steps'] == 0):
            print('Starting eval ...', flush=True)
            evalHelper.evaluate_session(model, sess)

        # Actual training step
        sess.run(train_step, feed_dict=poison_dict)
