import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision.models import vgg19
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def compute_psnr(img, reference, max_pixel=1):
    """ Given tensor image and reference, computes psnr

    Args:
        img (torch tensor)
        reference (torch tensor)
        max_pixel (int, optional): max pixel. Defaults to 1.

    Returns:
        torch tensor: psnr value
    """
    img = img / float(max_pixel)
    reference = reference / float(max_pixel)
    mse = torch.mean((img - reference) ** 2, dim=[1, 2, 3])
    psnr_val = 20 * torch.log10(max_pixel/torch.sqrt(mse))
    return psnr_val


def pad_image(img, desired_size, **kwargs):
    """
    Given a tensor image, pad to desired size (tuple)
    use F.pad docs to add the correct params for the intended mode
    Args:
        img: tensor
        desired_size: expected as (C,H,W)
    Keyword Arguments:
        mode: 'constant' (default), 'reflect', 'replicate' or 'circular'
    Returns:
        padded tensor img with desired_size (C,H,W)
    """

    if desired_size[1] <= img.shape[1] or desired_size[2] <= img.shape[2]:
        raise ValueError("Desired size is equal or smaller than input image. Cannot pad.")
    height, width = img.shape[1], img.shape[2]
    padding_bot = (desired_size[1] - height)//2
    padding_top = desired_size[1] - height - padding_bot
    padding_left = (desired_size[2] - width)//2
    padding_right = desired_size[2] - width - padding_left
    return F.pad(img, (padding_left, padding_right, padding_bot, padding_top, 0, 0), **kwargs)


def resize_image(img, desired_size, **kwargs):
    """
    Given a tensor image, resize to desired size (tuple)
    use F.interpolate docs to add the correct params for the intended mode
    Args:
        img: tensor
        desired_size: expected as (H,W)
    Returns:
        resized tensor img with desired_size (C,H,W)
    """
    out = torch.unsqueeze(img, dim=1)
    return torch.squeeze(F.interpolate(out, size=desired_size, **kwargs), dim=1)


def standardise_image(img, mean, std):
    """
    Standardise given tensor image by removing
    population mean and dividing by the standrad deviation
    Args:
        img: tensor
        mean: population mean (single mean or RGB means as floats or tensors)
        std: standard deviation (single std or RGB stds as floats or tensors)
    Returns:
        standardised img tensor
    """
    if not torch.is_tensor(mean):
        mean = torch.tensor(mean)
    if not torch.is_tensor(std):
        std = torch.tensor(std)
    expanded_mean = torch.unsqueeze(torch.unsqueeze(mean, dim=1), dim=1)
    expanded_std = torch.unsqueeze(torch.unsqueeze(std, dim=1), dim=1)
    out = torch.subtract(img, expanded_mean)
    out = torch.divide(out, expanded_std)
    return out


def reverse_image_standardisation(img, mean, std):
    """
    Reverse standardisation of given tensor image by
    multiplying by population mean and adding standard deviation
    Args:
        img: tensor
        mean: population mean (single mean or RGB means as floats or tensors)
        std: standard deviation (single std or RGB stds as floats or tensors)
    Returns:
        img tensor where standardisation has been reversed
    """
    if not torch.is_tensor(mean):
        mean = torch.tensor(mean)
    if not torch.is_tensor(std):
        std = torch.tensor(std)
    expanded_mean = torch.unsqueeze(torch.unsqueeze(mean, dim=1), dim=1)
    expanded_std = torch.unsqueeze(torch.unsqueeze(std, dim=1), dim=1)
    if torch.cuda.is_available():
        expanded_mean = expanded_mean.cuda()
        expanded_std = expanded_std.cuda()
        img = img.cuda()

    out = torch.multiply(img, expanded_std)
    out = torch.add(out, expanded_mean)
    return out


def compose_transforms_dict(config_dict, rgb_means, rgb_stds):
    """Outputs data_transforms for training pipeline
    Args:
        config_dict (Dictionary)
            example_dictionary={
                'flip_horizontal':True,
                'flip_vertical':True,
                'pad_training':[3, 107, 107],
                'pad_target':[3, 1070, 1070],
            }
        rgb_means: list of means (float)
        rgb_stds: list of stds (float)
    """
    from torchvision import transforms
    from src.data.transforms import Pad

    if rgb_means is not None:
        transform_list = [
                transforms.ToTensor(),
                transforms.Normalize(mean=rgb_means, std=rgb_stds)]
    else:
        transform_list = [transforms.ToTensor()]

    train_input_list = transform_list
    train_target_list = transform_list.copy()
    val_input_list = transform_list.copy()
    val_target_list = transform_list.copy()
    test_input_list = transform_list.copy()
    test_target_list = transform_list.copy()

    if config_dict['flip_horizontal']==True:
        train_input_list.append(transforms.RandomHorizontalFlip())
        train_target_list.append(transforms.RandomHorizontalFlip())

    if config_dict['flip_vertical']==True:
        train_input_list.append(transforms.RandomVerticalFlip())
        train_target_list.append(transforms.RandomVerticalFlip())

    if "pad_training" in config_dict:
        train_input_list.append(Pad(desired_size=config_dict['pad_training']))
        val_input_list.append(Pad(desired_size=config_dict['pad_training']))
        test_input_list.append(Pad(desired_size=config_dict['pad_training']))

    if "pad_target" in config_dict:
        train_target_list.append(Pad(desired_size=config_dict['pad_target']))
        val_target_list.append(Pad(desired_size=config_dict['pad_target']))
        test_target_list.append(Pad(desired_size=config_dict['pad_target']))

    data_transforms = {
        'train_input': transforms.Compose(train_input_list),
        'train_target': transforms.Compose(train_target_list),
        'val_input': transforms.Compose(val_input_list),
        'val_target': transforms.Compose(val_target_list),
        'test_input': transforms.Compose(test_input_list),
        'test_target': transforms.Compose(test_target_list),
    }
    
    return data_transforms
