import numpy as np
import torchvision.transforms as ts
import cv2
import torch
from pprint import pprint


class Preprocessing:
    
    """
    Preprocessing Class
    
    Preprocessing includes conversion to tensor, resizing, and normalization.
    
    """
    
    def __init__(self, name):
        
        self.name = name
        self.normalize = ts.Normalize(mean = 0, std = 255)
                    
    def initialize(self, opts):
        pre_opts = getattr(opts, self.name)
        
        if hasattr(pre_opts, 'size'):
            self.size = pre_opts.size
        
    def ultrasound_preprocessing(self):
        
        preprocess = [
            ts.ToTensor(),
            ts.Resize((self.size, self.size), antialias=True),
            ts.Normalize(mean = 0, std= 255)
        ]
        
        preprocess = ts.Compose(preprocess)

        return preprocess
        
        
    def phantom_preprocessing(self):
        pass
        
        
    def get_preprocessing(self):
        
        return{
            'BUSI': self.ultrasound_preprocessing,
            'Phantom': self.phantom_preprocessing
        }[self.name]()
        
    def print(self):
        print('\n\n############# Preprocessing Parameters #############')
        pprint(vars(self))
        print('###################################################\n\n')
        
    

if __name__ == '__main__':
    import torch
    import torchvision.transforms as transforms

    # Example input tensor (3 channels, 256x256)
    input_tensor = torch.rand(3, 256, 256) * 255

    # Example mean and std values (should match the number of channels)
    mean = [0, 0, 0]
    std = [255, 255, 255]

    # Normalize the input tensor using torchvision.transforms.Normalize
    normalize = transforms.Normalize(mean=mean, std=std)
    normalized_tensor = normalize(input_tensor)
    
    print(input_tensor)
    print(normalized_tensor)