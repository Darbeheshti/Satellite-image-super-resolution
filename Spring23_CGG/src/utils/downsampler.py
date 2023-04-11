import os, sys, glob, json, readline
import numpy as np
from PIL import Image
import np_utils as npu

if __name__=="__main__":
    '''
    - script to batch downsample ground truth (gt) images according to scale factor
    '''
    readline.set_completer_delims(" \t\n=")
    readline.parse_and_bind("tab: complete")

    # input args to .py script (run from terminal: python preprocessing.py)
    idir = input("Enter full path to /train/gt/ folder: ") + "/"
    odir = input("Enter full path (save location) for downsampled images: ") + "/"
    scaling_factor = float(input("Enter downsampling scaling factor (e.g. 2 for 2x downscaling): "))
    scaling_factor = 1/scaling_factor

    if idir==odir:
        raise NameError("Input directory should not match output directory")

    if not os.path.exists(odir):
            os.makedirs(odir)
    if os.path.exists(idir):
        # for each file in dir, downsample and save in odir
        for fn in os.listdir(idir):
            img_lr = npu.generate_low_resolution_image(idir+fn, scaling_factor)
            img_lr = Image.fromarray(np.uint8(img_lr))
            img_lr.save(odir+fn)