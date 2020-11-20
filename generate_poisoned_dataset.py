"""
Generates a poisoned dataset, given a clean dataset, a fully poisoned dataset and various parameters.

Outputs the following:
   - `train_{images,labels}.npy`: the poisoned training set (i.e. a proportion of the target class will now be replaced with harder-to-classify images and have the selected trigger applied)
   - `test_{images,labels}.npy`: the CIFAR-10 testing set with the trigger applied to *all* test images
   - `poisoned_train_indices.npy`: the indices of all poisoned training images
   - `train_no_trigger_images.npy`: `train_images.npy` but without triggers applied.
"""

import json
import os

import tensorflow as tf
import numpy as np

from poison_attack import DataPoisoningAttack

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

attack = DataPoisoningAttack(
    config['poisoning_trigger'],
    config['poisoning_target_class'],
    random_seed=config['random_seed'],
    reduced_amplitude=config['poisoning_reduced_amplitude'],
)

# Setting up the data and the model
print("Loading datasets")
clean_train_images = np.load(config["clean_dataset_dir"] + "/train_images.npy").astype(np.float32)
clean_train_labels = np.load(config["clean_dataset_dir"] + "/train_labels.npy").astype(np.int64)
num_train_examples = len(clean_train_images)

clean_test_images = np.load(config["clean_dataset_dir"] + "/test_images.npy").astype(np.float32)
clean_test_labels = np.load(config["clean_dataset_dir"] + "/test_labels.npy").astype(np.int64)
num_test_examples = len(clean_test_images)

fully_poisoned_train_images = np.load(config["poisoning_base_train_images"]).astype(np.float32)
assert len(fully_poisoned_train_images) == num_train_examples

print("Selecting indices")
poisoned_train_indices = attack.select_indices_to_poison(
    clean_train_labels,
    config['poisoning_proportion'],
    apply_to=config['poisoning_target_class'],
)

if not os.path.exists(config["poisoning_output_dir"]):
    os.makedirs(config["poisoning_output_dir"])
np.save(config["poisoning_output_dir"] + "/poisoned_train_indices.npy", poisoned_train_indices)

print("Poisoning training set with trigger")
poisoned_train_images, poisoned_train_labels = attack.poison_from_indices(
    clean_train_images,
    clean_train_labels,
    poisoned_train_indices,
    poisoned_data_source=fully_poisoned_train_images,
)
assert np.all(poisoned_train_labels == clean_train_labels)
np.save(config["poisoning_output_dir"] + "/train_images.npy", poisoned_train_images)
np.save(config["poisoning_output_dir"] + "/train_labels.npy", poisoned_train_labels)

poisoned_only_train_images = poisoned_train_images[poisoned_train_indices]
poisoned_only_train_labels = poisoned_train_labels[poisoned_train_indices]

print("Poisoning training set without trigger")
poisoned_no_trigger_train_images, poisoned_no_trigger_train_labels = attack.poison_from_indices(
    clean_train_images,
    clean_train_labels,
    poisoned_train_indices,
    poisoned_data_source=fully_poisoned_train_images,
    apply_trigger=False,
)
assert np.all(poisoned_no_trigger_train_labels == clean_train_labels)
np.save(config["poisoning_output_dir"] + "/train_no_trigger_images.npy", poisoned_no_trigger_train_images)
print("Done poisoning training set")

poisoned_test_indices = attack.select_indices_to_poison(
    clean_test_labels,
    1.,
    apply_to="all",
)

print("Poisoning test set")
poisoned_test_images, poisoned_test_labels = attack.poison_from_indices(
    clean_test_images,
    clean_test_labels,
    poisoned_test_indices,
)
print("Done poisoning test set")

np.save(config["poisoning_output_dir"] + "/test_images.npy", poisoned_test_images)
np.save(config["poisoning_output_dir"] + "/test_labels.npy", poisoned_test_labels)
