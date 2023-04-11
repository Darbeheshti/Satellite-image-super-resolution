from src.utils.torch_utils import pad_image


class Pad():
    '''
    Applies padding to a torch tensor
    Args:
        desired_size: expected as (C,H,W)
    Keyword Arguments:
        mode: 'constant' (default), 'reflect', 'replicate' or 'circular'
    '''
    def __init__(self, desired_size, **kwargs):
        self.desired_size = desired_size
        self.kwargs = kwargs
    
    def __call__(self, img):
        padded_img = pad_image(img, self.desired_size, **self.kwargs)
        return padded_img