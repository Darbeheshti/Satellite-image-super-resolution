import os
import glob
import torch
from torch.utils.data import Dataset
from PIL import Image
import random
import numpy as np


class SRDataset(Dataset):
    '''
    Creates a super resolution dataset compatible with torch dataloaders
    Args:
        fnames (list of Strings): list of filenames to be loaded
        img_dir (String): input img directory
        target_dir (String): target directory
        transform: torch transforms to apply to input image
        target_transform: torch transforms to apply to target image
        img_paths (String): path to images
        target_paths (String): path to targets 

    '''
    def __init__(self, fnames, img_dir, target_dir=None, transform=None, target_transform=None):
        super(Dataset, self).__init__()
        self.fnames = fnames
        self.img_dir = img_dir
        self.target_dir = target_dir
        self.transform = transform
        self.target_transform = target_transform
        self.img_paths = []
        self.target_paths = []
        for fname in fnames:
            self.img_paths.append(os.path.join(img_dir, fname))
            if self.target_dir is not None:
                self.target_paths.append(os.path.join(target_dir, fname))


    def __len__(self):
        '''
        Class function returns total number of samples in the dataset
        '''
        return len(self.img_paths)
    

    def __getitem__(self, idx):
        '''
        Class function returns a sample from the dataset at the given index idx:
        Returns:
            tuple of tensors: (input, target)
        '''
        seed = np.random.randint(0, 2^31 -1)
        img_path = self.img_paths[idx]
        img = Image.open(img_path)

        if self.transform is not None:
            random.seed(seed)
            torch.manual_seed(seed)
            img = self.transform(img)

        if self.target_dir is None:
            return img
        target_path = self.target_paths[idx]
        target = Image.open(target_path)
        if self.target_transform is not None:
            random.seed(seed)
            torch.manual_seed(seed)
            target = self.target_transform(target)
        return img, target


    def return_max_heights_and_widths(self):
        '''
        Calculates max heights and widths for inpus and targets in dataset
        
        Returns:
            max input height (int), max input width (int), max target height (int), max target width (int)
        '''
        i_heights, i_widths = [], []
        t_heights, t_widths = [], []
        for img, target in self:
            i_heights.append(img.size()[1])
            i_widths.append(img.size()[2])

            t_heights.append(target.size()[1])
            t_widths.append(target.size()[2])
            
        return max(i_heights), max(i_widths), max(t_heights), max(t_widths)


    def calc_mean_and_std_for_rgb_dataset(self):
        '''
        Calculate mean and standard across inputs of training dataset

        Returns:
            rgb_means, rgb_stds: mean and std for each channel
        '''
        tensor_list = []
        for img, target in self:
            target = None
            flat_img = torch.flatten(img, start_dim=1, end_dim=2)
            tensor_list.append(flat_img)

        concat_tensor_list = torch.cat(tensor_list, axis=1)
        rgb_means = torch.mean(concat_tensor_list, dim=1)
        rgb_stds = torch.std(concat_tensor_list, dim=1)
        return rgb_means, rgb_stds