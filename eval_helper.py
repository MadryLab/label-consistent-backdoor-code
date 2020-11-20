"""
Evaluates the model, printing to stdout and creating tensorboard summaries.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import math
import os

import numpy as np
import tensorflow as tf

import resnet_model

class EvalHelper(object):
    def __init__(self, sess, datasets, iterator_handle):
        # Global constants
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
        tf.set_random_seed(config['random_seed'])

        self.target_class = config["target_class"]

        self.num_eval_examples = config['num_eval_examples']
        self.eval_batch_size = config['eval_batch_size']
        self.eval_on_cpu = config['eval_on_cpu']
        self.augment_dataset = config['augment_dataset']
        self.augment_standardization = config['augment_standardization']

        self.model_dir = config['model_dir']

        self.random_seed = config['random_seed']

        # Setting up datasets
        self.iterator_handle = iterator_handle

        self.num_train_examples = len(datasets["clean_train"][1])
        self.num_test_examples = len(datasets["clean_test"][1])

        # Note: filtering done with clean labels
        filter_nontarget_only = np.isin(datasets["clean_test"][1], [self.target_class], invert=True)
        poisoned_no_target_test_dataset = (
            datasets["poisoned_test"][0][filter_nontarget_only],
            datasets["poisoned_test"][1][filter_nontarget_only]
        )
        self.num_eval_examples_nto = np.sum(filter_nontarget_only)

        self.clean_training_handle = self.prepare_dataset_and_handle(datasets["clean_train"], sess)
        self.poisoned_training_handle = self.prepare_dataset_and_handle(datasets["poisoned_train"], sess)

        self.num_poisoned_train_examples = len(datasets["poisoned_only_train"][1])
        if self.num_poisoned_train_examples > 0:
            self.poisoned_only_training_handle = self.prepare_dataset_and_handle(datasets["poisoned_only_train"], sess)
            self.poisoned_no_trigger_training_handle = self.prepare_dataset_and_handle(datasets["poisoned_no_trigger_train"], sess)
        self.clean_testing_handle = self.prepare_dataset_and_handle(datasets["clean_test"], sess)
        self.poisoned_testing_handle = self.prepare_dataset_and_handle(datasets["poisoned_test"], sess)
        self.poisoned_no_target_testing_handle = self.prepare_dataset_and_handle(poisoned_no_target_test_dataset, sess)

        self.global_step = tf.contrib.framework.get_or_create_global_step()

        # Setting up the Tensorboard and checkpoint outputs
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        self.eval_dir = os.path.join(self.model_dir, 'eval')
        if not os.path.exists(self.eval_dir):
            os.makedirs(self.eval_dir)

        self.saver = tf.train.Saver()
        self.summary_writer = tf.summary.FileWriter(self.eval_dir)

    def prepare_dataset_and_handle(self, full_dataset, sess):
        images, labels = full_dataset
        images_placeholder = tf.placeholder(tf.float32, images.shape)
        labels_placeholder = tf.placeholder(tf.int64, labels.shape)
        dataset = tf.contrib.data.Dataset.from_tensor_slices((images_placeholder, labels_placeholder))
        dataset = dataset.shuffle(buffer_size=10000, seed=self.random_seed).repeat()

        if self.augment_dataset:
            dataset = dataset.map(
                resnet_model.make_data_augmentation_fn(
                    standardization=self.augment_standardization,
                    is_training=False))

        dataset = dataset.batch(self.eval_batch_size)
        iterator = dataset.make_initializable_iterator()
        sess.run(iterator.initializer,
                 feed_dict={images_placeholder: images,
                            labels_placeholder: labels})
        handle = sess.run(iterator.string_handle())
        return handle

    def evaluate_session(self, model, sess):

        # Iterate over the samples batch-by-batch
        num_batches = int(math.ceil(self.num_eval_examples / self.eval_batch_size))
        total_xent_clean = 0.
        total_xent_clean_train = 0.
        total_xent_poison = 0.
        total_xent_poison_train = 0.
        total_xent_poison_train_nt = 0. # No trigger
        total_xent_poison_nto = 0. # Non-target only
        total_corr_clean = 0
        total_corr_clean_train = 0
        total_corr_poison = 0
        total_corr_poison_train = 0
        total_corr_poison_train_nt = 0 # No trigger
        total_corr_poison_nto = 0 # Non-target only

        total_not_target_clean = 0 # num clean test images not *classified* as the target class
        total_target_only_when_trigger_applied = 0 # num of the above that have classification changed to target when trigger applied

        for _ in range(num_batches):

            dict_clean = {self.iterator_handle: self.clean_testing_handle,
                          model.is_training: False}

            dict_clean_train = {self.iterator_handle: self.clean_training_handle,
                                model.is_training: False}

            dict_poison = {self.iterator_handle: self.poisoned_testing_handle,
                           model.is_training: False}

            dict_poison_train = {self.iterator_handle: self.poisoned_training_handle,
                                 model.is_training: False}

            if self.num_poisoned_train_examples > 0:
                dict_poison_train_nt = {self.iterator_handle: self.poisoned_no_trigger_training_handle,
                                        model.is_training: False}

            dict_poison_nontarget_only = {self.iterator_handle: self.poisoned_no_target_testing_handle,
                                          model.is_training: False}

            cur_corr_clean, cur_xent_clean, clean_batch_labels, clean_batch_classification = sess.run(
                [model.num_correct, model.xent, model.y_input, model.predictions],
                feed_dict=dict_clean)
            cur_corr_clean_train, cur_xent_clean_train = sess.run(
                [model.num_correct, model.xent],
                feed_dict=dict_clean_train)
            cur_corr_poison, cur_xent_poison, poison_batch_labels, poison_batch_classification = sess.run(
                [model.num_correct, model.xent, model.y_input, model.predictions],
                feed_dict=dict_poison)
            cur_corr_poison_train, cur_xent_poison_train = sess.run(
                [model.num_correct, model.xent],
                feed_dict=dict_poison_train)
            if self.num_poisoned_train_examples > 0:
                cur_corr_poison_train_nt, cur_xent_poison_train_nt = sess.run(
                    [model.num_correct, model.xent],
                    feed_dict=dict_poison_train_nt)
            else:
                cur_corr_poison_train_nt, cur_xent_poison_train_nt = 0, 0.0
            cur_corr_poison_nto, cur_xent_poison_nto = sess.run(
                [model.num_correct, model.xent],
                feed_dict=dict_poison_nontarget_only)

            assert np.all(poison_batch_labels == self.target_class)

            asr_filter = (clean_batch_classification != self.target_class)
            total_not_target_clean += np.sum(asr_filter)
            total_target_only_when_trigger_applied += np.sum(poison_batch_classification[asr_filter] == self.target_class)

            total_xent_clean += cur_xent_clean
            total_xent_clean_train += cur_xent_clean_train
            total_xent_poison += cur_xent_poison
            total_xent_poison_train += cur_xent_poison_train
            total_xent_poison_train_nt += cur_xent_poison_train_nt
            total_xent_poison_nto += cur_xent_poison_nto
            total_corr_clean += cur_corr_clean
            total_corr_clean_train += cur_corr_clean_train
            total_corr_poison += cur_corr_poison
            total_corr_poison_train += cur_corr_poison_train
            total_corr_poison_train_nt += cur_corr_poison_train_nt
            total_corr_poison_nto += cur_corr_poison_nto

        # Note that we've seen num_eval_examples of the training too
        avg_xent_clean = total_xent_clean / self.num_eval_examples
        avg_xent_clean_train = total_xent_clean_train / self.num_eval_examples
        avg_xent_poison = total_xent_poison / self.num_eval_examples
        avg_xent_poison_train = total_xent_poison_train / self.num_eval_examples
        avg_xent_poison_train_nt = total_xent_poison_train_nt / self.num_eval_examples
        avg_xent_poison_nto = total_xent_poison_nto / self.num_eval_examples
        acc_clean = total_corr_clean / self.num_eval_examples
        acc_clean_train = total_corr_clean_train / self.num_eval_examples
        acc_poison = total_corr_poison / self.num_eval_examples
        acc_poison_train = total_corr_poison_train / self.num_eval_examples
        acc_poison_train_nt = total_corr_poison_train_nt / self.num_eval_examples
        acc_poison_nto = total_corr_poison_nto / self.num_eval_examples

        asr = total_target_only_when_trigger_applied / total_not_target_clean

        summary = tf.Summary(value=[
            tf.Summary.Value(tag='xent clean test', simple_value=avg_xent_clean),
            tf.Summary.Value(tag='xent clean train', simple_value=avg_xent_clean_train),
            tf.Summary.Value(tag='xent poison test', simple_value=avg_xent_poison),
            tf.Summary.Value(tag='xent poison train', simple_value=avg_xent_poison_train),
            tf.Summary.Value(tag='xent poison train (no trigger)', simple_value=avg_xent_poison_train_nt),
            tf.Summary.Value(tag='xent poison test (non-target only)', simple_value=avg_xent_poison_nto),

            tf.Summary.Value(tag='accuracy clean test', simple_value=acc_clean),
            tf.Summary.Value(tag='accuracy clean train', simple_value=acc_clean_train),
            tf.Summary.Value(tag='accuracy poison test', simple_value=acc_poison),
            tf.Summary.Value(tag='accuracy poison train', simple_value=acc_poison_train),
            tf.Summary.Value(tag='accuracy poison train (no trigger)', simple_value=acc_poison_train_nt),
            tf.Summary.Value(tag='accuracy poison test (non-target only)', simple_value=acc_poison_nto),
            tf.Summary.Value(tag='attack success rate', simple_value=asr),
        ])
        self.summary_writer.add_summary(summary, self.global_step.eval(sess))

        print('clean test accuracy: {:.2f}%'.format(100 * acc_clean))
        print('poisoned test accuracy: {:.2f}%'.format(100 * acc_poison))
        print('poisoned test accuracy (non-target class only): {:.2f}%'.format(100 * acc_poison_nto))
        print('avg clean loss: {:.4f}'.format(avg_xent_clean))
        print('avg poisoned loss: {:.4f}'.format(avg_xent_poison))
        print('avg poisoned loss (non-target class only): {:.4f}'.format(avg_xent_poison_nto))
        print('attack success rate: {:.2f}%'.format(100 * asr))

        # Write results
        with open('job_result.json', 'w') as result_file:
            results = {
                'final clean test accuracy': acc_clean,
                'final poisoned test accuracy': acc_poison,
                'final poisoned test accuracy (non-target class only)': acc_poison_nto,
                'final attack success rate': asr,
            }
            json.dump(results, result_file, sort_keys=True, indent=4)
