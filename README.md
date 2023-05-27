
# Super resolution pipeline for satellite images

## Overview
This repository contains a PyTorch pipleline for upscaling low-resolution satellite images using super resolution models for Sentinel satellite images.
The pipleline allows the use of the models: LapSRN & SRCNN. You can train, infer and evaluate the output using this pipeline and change the model parameters as well.



<br />

## Listing of sub folders

```
├───configs
├───data                       <- Data directory
│   ├───processed              <- LapSRN resized images
│   └───raw                    <- The original, immutable data dump
│       └───CGG_data
│           ├───ge_test        <- google Earth High resolution images
│           ├───senti_test     <- Sentinel Low resolution images
│           └───train
│               ├───10x        <-  10 times Low Resoltion images
│               ├───8x         <-  8 times Low Resolution images
│               └───gt         <-  High Resolution images
├───models                     <-  Output of trainings
├───notebooks                  <-  Jupyter notebooks
├───references
├───src                        <-  Source code for running the pipeline
    ├───data                   <-  Creates a super resolution dataset 
    │                              compatible with torch dataloaders
    ├───models                 <-  Scripts to define models
    ├───trainer                <-  Model training scripts
    └───utils                  <-  Utility functions using NumPy and Torch
```
<br />

## Installation
1.  Install Torch from https://pytorch.org/. make sure to choose the compute platform that suits your system settings.
2. Install requirements.txt file: pip install -r requirements.txt
      
<br />


## config file
The configuration file is where all the necessary parameters for training/inference/evaluation are set.
There are three flags which correspond to each mode. Below is the config file with all possible variables filled in. Some of these are optional, meaning they can be deleted from the configuration file.


### Example of config file for LapSRN model:

``` 
[TRAINING]
low_res_dir = data/processed/CGG_data/train/lr_crop_8x/
ground_truth_dir = data/processed/CGG_data/train/gt_crop/
random_seed = 100
train_split = 0.8
flip_horizontal = true
flip_vertical = false
model = LapSRN
optimizer = adam
learning_rate = 0.0001
num_epochs = 180
loss = charb
batch_size = 2
standardisation = true

[INFERENCE]
model_dir = models/LapSRN_03_04_2023_1825/

[EVALUATE]
model_dir = models/LapSRN_03_04_2023_1825/
```
<br />


### More details on each parameter:

```[TRAINING]```<br/>

```low_res_dir``` *Path to folder containing low resolution training data*

```ground_truth_dir``` *Path to folder containing low resolution training data*

```create_patches``` *Tell pipeline whether or not to create patches from images, needed for some models such as SRCNN*

```low_res_patches_path``` ***OPTIONAL***: *If training a model requiring patches of images, give the path to existing high-res patches folder or create folder with the given path name*

```high_res_patches_path ``` ***OPTIONAL***: *If training a model requiring patches of images, give the path to existing low-res patches folder or create a folder with the given path name*

```lr_patches_down_scale = 4```<br/>
***OPTIONAL***: *if patching images, describe how much smaller the patches should be*

```random_seed = 100``` *Random seed for the separation of the training data into train + validation. Keep the same to compare two models (they will train on the same data)*

```train_split = 0.7``` *The proportion of the total files in the training directory to be used for training (e.g. 0.7 = 70%). The remainder go into validation*

```standardisation = true``` *Flag (either true or false - case sensitive). If true, images are normalised using the mean and standard deviation in the training set. The output is saved in the model folder as a JSON*

```flip_horizontal = true``` *Flag (either true or false - case sensitive). If true, images are flipped randomly in the horizontal axis during training*

```flip_vertical = false``` *Flag (either true or false - case sensitive). If true, images are flipped randomly in the vertical axis during training*

```pad_training = [3, 107, 107]``` ***OPTIONAL***: *Padding for the low-res training images, given as an array [C, H, W] (pytorch convention)*

```pad_target = [3, 1070, 1070]``` ***OPTIONAL***: *Padding for the high-res training images, given as an array [C, H, W] (pytorch convention)*

```model = SimpleModel``` *Model name: i.e. SimpleModel, SRCNN, LapSRN etc.*

```optimizer = adam``` *Name of the optimiser to be used: i.e. adam*

```learning_rate = 0.001``` *Learning rate (float) to be set for the optimiser*

```num_epochs = 3``` *Number of epochs to train model for*

```batch_size = 10``` *Number of samples to be loaded into the GPU as a batch*

```loss = mse``` *Loss function to be used during the training option (i.e. mse, charb)*

<br/>

```[INFERENCE]```

```model_dir = models/example_model_dir/``` *Path to directory where trained model.pth sits in*

```low_res_dir = data/path_to_folder_with_unseen_data/``` ***OPTIONAL***: *input directory when performing inference on a new or unseen set. If this <br/>
is not included, the model will automatically compute inference on the validation set*

<br/>

```[EVALUATE]```

```model_dir = models/example_model_dir/``` *Path to directory where the trained model.pth sits in*<br/>


<br />


##  Preprocessing data for LapSRN
When training LapSRN with a scale factor of 8, the model needs 8 times lower resolution images as target images to train. crop_and_downsample.py generates  low scale images with downsampling scaling factor as input.  

### Instructions for running crop_and_downsample.py:

* run: ```python .\src\utils\crop_and_downsample.py``` to create consistent image size (1024x1024) for target high resolution images
* enter relevant folders to the following prompts:

* ```Enter path to gt dir e.g. data/raw/CGG_data/train/gt:``` 
* ```Enter path to output dir e.g. data/processed/CGG_data/train/lr_crop_8x/:```
* ```Enter downsampling scaling factor (e.g. 8 for 8x downscaling):```
* ```Do you want to save cropped gt? (y or n):``` (only enter 'n' once targets have already been created)

<br />


## Tensorboard
* run: ```tensorboard --logdir="models/"```
* go to http://localhost:6006/


<br />


## Code for running training, inference, evaluation of a model:

* **Training**: run ```python3 app.py --mode train --config config.txt```
* **Inference**: run ```python3 app.py --mode inference --config config.txt```
* **Evaluate**: run ```python3 app.py --mode evaluate --config config.txt```


<br/>
<br/>

# Results
This table shows evaluation metrics for LapSRN.

| Model Name    | Upscaling Ratio |MSE   |PSNR |SSIM |
| ------------- | -------------   |------|---- |---- |
| LapSRN        | 8X              |57.35 |23.72|0.802|

<br/>

### A sample result of low resolution, LapSRN X8 upscaling & ground truth images:

![output](https://user-images.githubusercontent.com/60885691/229791389-65c63d03-8794-4ac9-bf56-6403f14ab6cc.png)

<br/>

### Histograms of the image values for low resolution, LapSRN X8 upscaling & ground truth images:

![histogram](https://user-images.githubusercontent.com/60885691/229791424-35bc64b5-9e09-4ab1-bf00-3fcedc417a7f.png)

<br/>

## Credits and Additional information

### LapSRN
LapSRN (https://arxiv.org/pdf/1704.03915.pdf) in this pipeline is implemented from 
https://github.com/Lornatang/LapSRN-PyTorch. The Laplacian Pyramid Super-Resolution Network (LapSRN) progressively reconstructs the sub-band residuals of high-resolution images. At each pyramid level, the model takes coarse-resolution feature maps as input, predicts the high-frequency residuals, and uses transposed convolutions for upsampling to the finer level. LapSRN trains with deep supervision using a Charbonnier loss function. For more details on LapSRN, look at the LapSRN Git repositpry.
    
### SRCNN
SRCNN is a deep learning method for single image super-resolution (SR). It directly learns an end-to-end
mapping between the low/high-resolution images. The mapping is represented as a deep convolutional neural network (CNN) that takes
the low-resolution image as the input and outputs the high-resolution one. Here is the link to the original paper: https://arxiv.org/pdf/1501.00092.pdf

<br />



