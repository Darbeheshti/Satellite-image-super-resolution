import os, json, sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from skimage.metrics import structural_similarity as ssim
from skimage.util import random_noise
from sklearn.metrics import mean_squared_error
import src.utils.np_utils as npu

def compute_mse_errors(imgs_gt, imgs_pred, filenames):
    '''
    Computes mean squared error metrics for pred vs ground truth
    Args:
        imgs_pred: prediction images (list of numpy arrays)
        imgs_gt: ground truth images (list of numpy arrays)
        filenames: list of image filenames (str)
    Returns:
        dictionary of mse stats
    '''
    mse_list = []
    for im_pred, im_gt in zip(imgs_gt, imgs_pred):
        mse_list.append(mean_squared_error(im_gt.flatten(), im_pred.flatten()))
    
    mse_stats = {
        'mean_mse': np.mean(mse_list),
        'max_mse': np.max(mse_list),
        'max_mse_file': filenames[np.argmax(mse_list)],
        'min_mse': np.min(mse_list),
        'min_mse_file': filenames[np.argmin(mse_list)]
    }
    
    return mse_stats


def compute_psnr_errors(imgs_gt, imgs_pred, filenames):
    '''
    Computes peak signal to noise ratio metrics for pred vs ground truth
    Args:
        imgs_pred: prediction images (list of numpy arrays)
        imgs_gt: ground truth images (list of numpy arrays)
        filenames: list of image filenames (str)
    Returns:
        dictionary of psnr stats
    '''
    psnr_list = []
    for im_pred, im_gt in zip(imgs_gt, imgs_pred):
        psnr_list.append(cv2.PSNR(im_gt, im_pred))
    
    psnr_stats = {
        'mean_psnr': np.mean(psnr_list),
        'max_psnr': np.max(psnr_list),
        'max_psnr_file': filenames[np.argmax(psnr_list)],
        'min_psnr': np.min(psnr_list),
        'min_psnr_file': filenames[np.argmin(psnr_list)]
    }
    
    return psnr_stats


def compute_ssim_errors(imgs_gt, imgs_pred, filenames):
    '''
    Computes structural similarity index for pred vs ground truth
    Args:
        imgs_pred: prediction images (list of numpy arrays)
        imgs_gt: ground truth images (list of numpy arrays)
        filenames: list of image filenames (str)
    Returns:
        dictionary of ssim stats
    '''
    ssim_list = []
    for im_pred, im_gt in zip(imgs_gt, imgs_pred):
        ssim_list.append(ssim(im_gt.flatten(), im_pred.flatten()))
    
    ssim_stats = {
        'mean_ssim': np.mean(ssim_list),
        'max_ssim': np.max(ssim_list),
        'max_ssim_file': filenames[np.argmax(ssim_list)],
        'min_ssim': np.min(ssim_list),
        'min_ssim_file': filenames[np.argmin(ssim_list)]
    }
    
    return ssim_stats