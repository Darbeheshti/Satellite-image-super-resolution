import os, sys, glob, json
import numpy as np
from PIL import Image
import np_utils as npu

if __name__=="__main__":
    '''
    - script to batch crop and downsample ground truth (gt) images according to scale factor
    '''
    
    gt_dir = input("Enter path to gt dir e.g. data/raw/CGG_data/train/gt: ") + "/"
    odir = input("Enter path to output dir e.g. data/processed/CGG_data/train: ") + "/"
    scaling_factor = int(input("Enter downsampling scaling factor (e.g. 2 for 2x downscaling): "))
    crop_gt_bool = input("Do you want to save cropped gt? (y or n): ")

    if crop_gt_bool == 'y':
        crop_gt_bool = True
    else:
        crop_gt_bool = False
    
    gt_crop_dir = os.path.join(odir, f'gt_crop')
    lr_crop_dir = os.path.join(odir, f'lr_crop_{scaling_factor}x')

    scaling_factor = 1/scaling_factor
    size = 1024

    print(f"Saving cropped images in {odir}")

    if not os.path.exists(gt_crop_dir):
        os.makedirs(gt_crop_dir)
    if not os.path.exists(lr_crop_dir):
        os.makedirs(lr_crop_dir)
    if os.path.exists(gt_dir):
        for fn in os.listdir(gt_dir):
            gt_img = npu.load_image_as_np(os.path.join(gt_dir, fn))
            if gt_img.shape[0] > size or gt_img.shape[1] > size:
                if crop_gt_bool:
                    gt_crop = npu.crop_img(gt_img, size)
                    gt_crop = Image.fromarray(np.uint8(gt_crop))
                    gt_crop.save(os.path.join(gt_crop_dir, fn))

                lr_crop = npu.generate_low_resolution_image(os.path.join(gt_crop_dir, fn), scaling_factor)
                lr_crop = Image.fromarray(np.uint8(lr_crop))
                lr_crop.save(os.path.join(lr_crop_dir, fn))
