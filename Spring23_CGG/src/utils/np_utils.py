import logging
import os, cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import patchify
import shutil

log = logging.getLogger(__name__)

"""
Functions for image processing
All image processing functions return a NumPy matrix (H, W, C) H=height, W=width, C=channel
"""


def load_image_as_np(input_img):
    """
    Loads image using PIL (RGB convention) and converts to NumPy matrix
    Args:
        input_img: path to image file (str)
    Returns:
        NumPy matrix of image (H, W, C)
    """
    if isinstance(input_img, str):
        input_img = Image.open(input_img)
        input_img = np.array(input_img)
    return input_img


def normalise_image(input_img):
    """
    Divides all pixels in an input image by max pixel intensity (255)
    Args:
        input_img: image (as PIL image or NumPy matrix) or path to image (str)
    Returns:
        normalised image as NumPy matrix (H, W, C)
    """
    if isinstance(input_img, str):
        input_img = load_image_as_np(input_img)
    else:
        input_img = np.array(input_img)
    
    normalised_img = np.divide(input_img, 255)
    return normalised_img


def pad_image(input_img, desired_size, **kwargs):
    """
    given an input image, pads to the desired size and returns padded image
    Args:
        input_img: image (as PIL image or NumPy matrix) or path to image (str)
        desired_size: tuple (W, H)
    Keyword arguments (check np.pad docs):
        e.g. for zero padding: mode='constant', constant_values=0
    Returns:
        padded NumPy matrix (H, W, C)
    """
    if isinstance(input_img, str):
        input_img = load_image_as_np(input_img)
    else:
        input_img = np.array(input_img)
    
    height, width = input_img.shape[0], input_img.shape[1]
    
    if desired_size[0] <= width or desired_size[1] <= height:
        raise ValueError("Desired size is equal or smaller than input image. Cannot pad.")
    
    padding_left = (desired_size[0] - width)//2
    padding_right = desired_size[0] - width - padding_left
    padding_bot = (desired_size[1] - height)//2
    padding_top = desired_size[1] - height - padding_bot
    
    img_padded = np.pad(input_img, ((padding_bot, padding_top), (padding_left, padding_right), (0,0)), **kwargs)
    return img_padded


def generate_low_resolution_image(input_img, scale_factor):
    """
    Hanyuan function for downsampling images with scale factor
    Args:
        input_image_file: image (as PIL image or NumPy matrix) or path to image (str)
        scale_factor: int or float to scale image by
    Returns:
        downsampled Numpy matrix (H, W, C)
    """
    if isinstance(input_img, str):
        input_img = Image.open(input_img)
    if isinstance(input_img, np.ndarray):
        input_img = Image.fromarray(np.uint8(input_img))

    width, height = input_img.size
    new_width, new_height = int(width * scale_factor), int(height * scale_factor)
    low_resolution_image = input_img.resize((new_width, new_height))
    low_resolution_image = np.array(low_resolution_image)
    return low_resolution_image


def resize_image(input_img, desired_size, **kwargs):
    """
    For a given image, resize it to the desired size
    Args:
        input_img: image (as PIL image or NumPy matrix) or path to image (str)
        desired_size: tuple (W, H)
    Keyword parameters:
        optional PIL params (see docs), e.g. interpolation=cv2.INTER_CUBIC
    Returns:
        resized NumPy matrix (H, W, C)
    """
    if isinstance(input_img, str):
        input_img = load_image_as_np(input_img)
    else:
        input_img = np.array(input_img)
    
    img_resized = cv2.resize(input_img, dsize=desired_size, **kwargs)
    return img_resized


def standardise_image(input_img, mean, std):
    """
    remove mean from image, scale by standard deviation
    Args:
        input_img: image (PIL image or NumPy matrix) or path to image (str)
        mean: population mean (float or int)
        std: population standard deviation (float or int)
    Returns:
        standardised image as NumPy matrix (H, W, C)
    """
    if isinstance(input_img, str):
        input_img = load_image_as_np(input_img)
    else:
        input_img = np.array(input_img)
    
    img_standard = np.subtract(input_img, mean)
    img_standard = np.divide(img_standard, std)
    return img_standard


def reverse_image_standardisation(input_img, mean, std):
    """
    remove scaling by standard deviation, add mean
    Args:
        input_img: image (PIL image or NumPy matrix) or path to image (str)
        mean: population mean (float or int)
        std: population standard deviation (float or int)
    Returns:
        unstandardised image as NumPy matrix (H, W, C)
    """
    if isinstance(input_img, str):
        input_img = load_image_as_np(input_img)
    else:
        input_img = np.array(input_img)
    
    img_standard = np.multiply(input_img, std)
    img_standard = np.add(img_standard, mean)
    return img_standard.astype(int)


def compute_stats_channel_dim(path_imgs, filenames):
    """
    compute the mean and std for all RGB channels in a batch of images
    normalises and flattens images (H, W, C) in channel dim (3, N) where N = H*W
    works for images with mixed dimensions
    Args:
        path_imgs: path to images (str)
        filenames: list of filenames in directory, if "all" computes stats for all images in directory
    Returns:
        tuple of arrays ([mean_r, mean_g, mean_b], [std_r, std_g, std_b])
    """
    if isinstance(path_imgs, str):
        if filenames == "all":
            filenames = os.listdir(path_imgs)

        all_imgs_concat_r = []
        all_imgs_concat_g = []
        all_imgs_concat_b = []
        for file_name in filenames:
            img = load_image_as_np(path_imgs + file_name)
            img = normalise_image(img)
            if img.shape[2] > 3:
                channels = img.shape[2]
                logging.critical(f"Image {file_name} has {channels} channels instead of 3 (R,G,B)."
                                 f" Make sure that all images have only 3 channels!")
                exit()

            img = img.transpose(2, 0, 1).reshape(3, -1)
            all_imgs_concat_r.extend(img[0])
            all_imgs_concat_g.extend(img[1])
            all_imgs_concat_b.extend(img[2])

        means = [np.mean(all_imgs_concat_r), np.mean(all_imgs_concat_g), np.mean(all_imgs_concat_b)]
        stds = [np.std(all_imgs_concat_r), np.std(all_imgs_concat_g), np.std(all_imgs_concat_b)]
        return means, stds


def compute_percentage_green(input_img):
    """
    For a given image, convert to HSV, threshold with lower/upper green bounds (as defined in hsv)
    Compute percentage of image containing green pixels
    Args:
        input_img: image (as PIL image or NumPy matrix) or path to image (str)
    Returns:
        percentage (float) of image containing green pixels: e.g. 50.0 = 50%
    """
    if isinstance(input_img, str):
        input_img = load_image_as_np(input_img)
    else:
        input_img = np.array(input_img)
    
    hsv_img = cv2.cvtColor(input_img, cv2.COLOR_RGB2HSV)
    lower_green_hsv = np.array([30, 40, 40])
    upper_green_hsv = np.array([80, 255, 255])
    mask = cv2.inRange(hsv_img, lower_green_hsv, upper_green_hsv)
    percentage_green = (mask == 255).mean()*100
    
    return percentage_green


def compute_percentage_edges(input_img, low_threshold=50, high_threshold=150, kwd=5):
    """
    For a given RGB image, convert to greyscale
    Smooth image with Gaussian kernel
    Create a mask containing only edges with Canny function
    Get percentage of pixels in image which belong to an edge
    Args:
        input_img: image (as PIL image or NumPy matrix) or path to image (str)
        low_threshold (default=50): lowest value to clip greyscale mask
        high_threshold (default=150): highest value to clip greyscale mask
        kwd: kernel width (default=5x5): for gaussian smoothing
    Returns:
        percentage (float) of image containing edges: e.g. 50.0 = 50%
    """
    if isinstance(input_img, str):
        input_img = load_image_as_np(input_img)
    else:
        input_img = np.array(input_img)
    
    img_grey = cv2.cvtColor(input_img, cv2.COLOR_RGB2GRAY)
    img_blurred = cv2.GaussianBlur(img_grey, (kwd, kwd), 0)
    mask = cv2.Canny(img_blurred, low_threshold, high_threshold)
    percentage_edges = (mask == 255).mean()*100
    
    return percentage_edges


def ImagePlot(input_img):
    """
    Plotting an image.
    Args:
        img: np.asarray"""
    imgplot = plt.imshow(input_img)
    return imgplot

    
def create_patches(model, input_path, out_hr_path, out_lr_path, lr_downsample_scale, SIZE = 300, STRIDE = 250):
    """
    Function to patch images i.e., cut the images in to smaller patches
    Args:
        input_path: Path of the folder containing HR images for training
        out_hr_path: Path of the folder in which HR patches are saved
        out_lr_path: Path of the folder in which LR patches are saved
        lr_downsample_scale (int): Downsampling scale for LR images using bicubic interpolation
        SIZE (int): size of the output patches, same in height and width
        STRIDE (int): length of stride, i.e., overlap between two neighbouring patches
    """
    if os.path.exists(out_hr_path):
        shutil.rmtree(out_hr_path)
    if os.path.exists(out_lr_path):
        shutil.rmtree(out_lr_path)

    os.makedirs(out_hr_path, exist_ok=True)
    os.makedirs(out_lr_path, exist_ok=True)

    all_paths = os.listdir(input_path)

    print(f"Creating patches for {len(all_paths)} images")
    
    for image_file in tqdm(all_paths, total=len(all_paths)):
        image_path = input_path + image_file
        image = Image.open(image_path)
        image_name = image_file.split(os.path.sep)[-1].split('.')[0]
        patches = patchify.patchify(np.array(image), (SIZE, SIZE, 3), STRIDE)

        counter = 0
        for i in range(patches.shape[0]):
            for j in range(patches.shape[1]):
                counter += 1
                patch = patches[i, j, 0, :, :, :]
                patch = cv2.cvtColor(patch, cv2.COLOR_RGB2BGR)
                cv2.imwrite(
                    f"{out_hr_path}/{image_name}_{counter}.png",
                    patch
                )

                # Convert to bicubic and save.
                height, width, _ = patch.shape
                low_res_img = cv2.resize(patch,
                                         (int(width/lr_downsample_scale),
                                          int(height/lr_downsample_scale)),
                                         interpolation=cv2.INTER_CUBIC)

                # upscale using BICUBIC
                if model == "SRCNN":
                    low_res_img = cv2.resize(low_res_img,(width, height),interpolation=cv2.INTER_CUBIC)
                else:
                    low_res_img = low_res_img

                cv2.imwrite(f"{out_lr_path}/{image_name}_{counter}.png", low_res_img)


def crop_img(input_img, height, width=None):
    """
    given an input image, takes center crop of image with N x N dimensions
    Args:
        input_img: image (as PIL image or NumPy matrix) or path to image (str)
        height: desired height (int)
        width: desired width (int)
    Returns:
        cropped NumPy matrix (H, W, C)
    """
    if isinstance(input_img, str):
        input_img = load_image_as_np(input_img)
    else:
        input_img = np.array(input_img)
    # if only height entered then crop square height x height
    if width is None:
        width = height

    input_img_height, input_img_width = input_img.shape[0], input_img.shape[1]
    if input_img_height < height or input_img_width < width:
        raise ValueError("Crop size is bigger than img size. Cannot crop.")
        exit()

    start_width = input_img_width//2-(width//2)
    start_height = input_img_height//2-(height//2)
    return input_img[start_height:start_width+height, start_height:start_width+width]