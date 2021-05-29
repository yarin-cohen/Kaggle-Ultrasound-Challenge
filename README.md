# Kaggle Ultrasound Nerve Segmentation
This repository is a Python solution for the Kaggle Ultrasound Nerve Segmentation challange.
This deep neural network achieves ~0.65 (top 15%) score on the leaderboard based on test images, and can be a good staring point
for further, more serious approaches.

The architecture was inspired by [U-Net: Convolutional Networks for Biomedical Image Segmentation.](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/)

# Overview:

* using Keras 2
* using cv2
* using skimage
* using imagecodecs

### Data:
The solution needs to pre-process the provided data into .npy files. This can be done in two ways:
* #### using "Creating Variables and Basic EDA" notebook:
    * open `global_params.py` and set `np_files_path` to the path you want to hold all .npy files.
    * running the second cell in this notebook will create the proper variables. You only need to write your own path
      for the train and test data

* #### running the script manually:     
    * open `global_params.py` and set `np_files_path` to the path you want to hold all .npy files.
    * open python and import `project_utils.py`
    * run `create_train_data(TRAIN_PATH)` with the appropriate path to the training data
      run `create_val_data(TRAIN_PATH)` with the appropriate path to the training data
    * run `create_test_Data(TEST_PATH)` with the appropriate path to the test data.
    * run `create_iso_mask()`

 Make sure to set up `TRAIN_PATH` such that the raw training images lies under 
`TRAIN_PATH + '\train'`.  Make sure to set up `TEST_PATH` such that the raw test images lies under 
`TEST_PATH + '\test'`. This script will pre-divide your data to training and validation sets. Make sure to use the same path to the training
data with `TRAIN_PATH`.

### Preprocessing:
Data is only resized. Output images (masks) are scaled to [0, 1] interval.

### Model:
This solution uses 2 seperate models:

#### U-net:
* Popular choice for segmentation challenges. Basically a convolutional auto-encoder, but with a twist - it has skip
  connections from encoder layers to decoder layers that are on the same "level". See provided article for more info
* This deep neural network is implemented with Keras and Tensorflow. Last layer uses a Sigmoid activation function
  which makes sure that mask pixels are in [0, 1] range.
* Unlike the classic u-net model, this model is a slightly improved model, using inception blocks instead of regular 
convolutional layers.

#### Classification model:
* Starting from the basic solution showed great potential, but had one obvious flaw: it seemed that a lot of the 
  performance hit on the validation set was due to false detection. So it was decided to train a seperate model on the
  training set.
* This model would train on isolated boxes containing different segments of images from the training set and predict 
  which segments contain a nerve.
* Data to train this specific model was generated randomly from the training set- randomly cropping image segments and
  labeling them according to their overlap with an actual labeled mask in the same image.
* Since there's an obvious imbalance between positive and negative examples for this classification task, Focal loss was
used to train the model. There are a lot more negative examples (containing no mask) than actual crops that contain a
  proper mask.
* This model is based on a ResNet50 model, and is trained after loading pre-trained weights of the model on the imagenet
  dataset.

After training both models separately, we created the predictions on the test set using both models. First, the segmentation
model would run on every test image and create the predicted mask. Then, we ran the classification model on every mask
to decide based on the result if the mask should be kept or deleted.

This got us an improved version of the original u-net model. An additional improvement was added by counting the number
of pixels in a predicted mask and disqualify masks with a certain number of pixels and under, based on statistics from
the validation and training set (to use during test prediction).

### Training
* The segmentation model is trained for about 50 epochs, and the classification model is trained for about 5 epochs.
* Loss function for training is basically just a negative of Dice coefficient (which is used as the evaluation metric on
  the competition), and this is implemented as custom loss function using Keras backend - see `check dice_coef()` and 
  `dice_coef_loss()` functions in train.py for more detail. Also, for making the loss function smooth, a factor `smooth = 1`
  factor is added.
* During training, model's weights are saved in HDF5 format.


## How To Use:

* Create all .npy files by following instructions under the Data section of this readme (important to set up!)
* Open `global_params.py`. Set proper directories for:
  * Net Weights files
  * test predictions images directory - under `predictions_dir`
  * (Optional) set any other desired parameter

* run `train.py`. This will train both models, unless `train_classification` is 0 under `global_params.py`
* run `create_test_predictions_files.py`. This will create a folder to hold all mask predictions for the test set as
images under a single folder
* run `create_submission_file.py`. This will create the proper submission file in the required format, after using the
trained classification model and other criteria.

# Using notebooks for analysis:
With the provided notebooks you can perform a basic Exploratory data analysis and check the performace of both trained 
models by checking predictions on random sets of data from the training and validation sets

### "Creating Variables with Basic EDA" notebook
* This notebook allows you to create all .npy files and/or load them.
* You'll be able to see what proper segmentation looks like on both created training and validation datasets
* You'll be able to examine basic statistics about the masks and data characteristics.

### "Models Performace Visualization" notebook
* This notebook allows you to examine trained models' performance.
* It shows the predictions of random examples vs the true labels/masks of the validation set. The data comes exclusively
from the validation set to make sure that performance is not examined under over fitting conditions. 
