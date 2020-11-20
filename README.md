# Label-Consistent Backdoor Attacks code

This repository contains the code to replicate experiments in our paper:

**Label-Consistent Backdoor Attacks**
*Alexander Turner, Dimitris Tsipras, Aleksander Madry*
https://arxiv.org/abs/1912.02771

The datasets we provide are modified versions of the [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset.

### Running the code

#### Step 1: Setup, before doing anything else

Run `./setup.py`.

This will download CIFAR-10 into the `clean_dataset/` directory in the form of `.npy` files.
It will also download modified forms of the CIFAR-10 training image corpus into the `fully_poisoned_training_datasets/` directory, formatted and ordered identically to `clean_dataset/train_images.npy`. In each corpus, every image has been replaced with a harder-to-classify version of itself (with no trigger applied).

The `gan_0_x.npy` files use our GAN-based (i.e. latent space) interpolation method with τ = 0.x. The `two_x.npy` and `inf_x.npy` files use our adversarial perturbation method with an l<sub>2</sub>-norm bound and l<sub>∞</sub>-norm bound, respectively, of x.

Finally, this script will install numpy and tensorflow.

#### Step 2: Generating a poisoned dataset

To generate a poisoned dataset, first edit the last section in `config.json`.
The settings are:
- `poisoning_target_class`: which (numerical) class is the target class.
- `poisoning_proportion`: what proportion of the target class to poison.
- `poisoning_trigger`: which backdoor trigger to use (`"bottom-right"` or `"all-corners"`).
- `poisoning_reduced_amplitude`: the amplitude of the backdoor trigger on a 0-to-1 scale (e.g. `0.12549019607843137` for 32/255), or `null` for maximum amplitude (i.e. 1).
- `poisoning_base_train_images`: the source of the harder-to-classify images to use for poisoning. 

Then, run `python generate_poisoned_dataset.py`, which will generate the following files in the `poisoning_output_dir` you specified:
   - `train_{images,labels}.npy`: the poisoned training set (i.e. a proportion of the target class will now be replaced with harder-to-classify images and have the selected trigger applied).
   - `test_{images,labels}.npy`: the CIFAR-10 testing set with the trigger applied to *all* test images.
   - `poisoned_train_indices.npy`: the indices of all poisoned training images.
   - `train_no_trigger_images.npy`: `train_images.npy` but without triggers applied.

#### Step 3: Training a network on the poisoned dataset.

To train a neural network on the poisoned dataset you generated, now edit the other sections in `config.json` as you wish.
The settings include:
- `augment_dataset`: whether to use data augmentation. If true, uses the function specified by `augment_standardization`, `augment_flip` and `augment_padding`.
- `target_class`: which (numerical) class is the target class (only used for evaluating the attack success rate).

Then, run `python train.py`.
