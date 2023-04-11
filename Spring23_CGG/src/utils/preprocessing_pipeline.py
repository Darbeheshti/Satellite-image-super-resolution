import os, sys, glob, json, readline
import numpy as np
from PIL import Image
import np_utils as npu
from sklearn.model_selection import train_test_split

if __name__=="__main__":
    '''
    pre-processing pipeline (run from terminal)
    - splits data in train, val, test filenames according to random seed
    - computes RGB channel mean, std for the training set
    - creates a JSON with train/val/test split and channel stats
    '''
    readline.set_completer_delims(" \t\n=")
    readline.parse_and_bind("tab: complete")

    # input args to .py script (run from terminal: python preprocessing.py)
    idir_x = input("Enter path to input training data (10x): ") + "/"
    idir_y = input("Enter path to input training data (ground truth): ") + "/"
    rand_state = int(input("Enter random state/seed: ") or 100)
    train_percent= int(input("Enter percentage split to form training set (e.g. 70%): ") or 70)
    valid_percent = int(input("Enter percentage split of the remaining data (" + str(100-train_percent) + ") to form the validation & test sets (e.g. equal val/test is 50%): ") or 50)
    odir = input("Enter the save path for the JSON output: ") + "/"

    # split data
    X = os.listdir(idir_x)
    y = os.listdir(idir_y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_percent/100, random_state=rand_state)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, train_size=valid_percent/100, random_state=rand_state)

    # compute means and std of training set
    means, stds = npu.compute_stats_channel_dim(idir_x, X_train)

    # create and save JSON
    preprocessing_json = {}
    preprocessing_json['means'] = means
    preprocessing_json['stds']  = stds
    preprocessing_json['train'] = X_train
    preprocessing_json['val']   = X_val
    preprocessing_json['test']  = X_test
    with open(odir + "preprocessing.json", "w") as f:
        json.dump(preprocessing_json, f)
