"""Dataset Module"""

import torch.utils.data as data
import os
import cv2
import numpy as np
import torch


class PhantomDataSet(data.Dataset):
    def __init__(self, root_path, augmentation=None, normalization=False):
        super(PhantomDataSet, self).__init__()
        print('Phantom Dataset successfully initialized')
        print('Number of images in the dataset is : {0}'.format(self.__len__()))
        
    def __getitem__(self, idx):
        #TODO: Add phantom dataset processing pipeline 
        
        pass

    def __len__(self):
        pass


class UltraSoundDataSet(data.Dataset):

    """
    Breast Ultrasound Images Dataset (BUSI). Read images, apply augmentation and preprocessing transformations.

    Args:
        images_dir (str): path to images folder, including original images and masks
        augmentation (albumentations.Compose): data transfromation pipeline 
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing 
            (e.g. noralization, shape manipulation, etc.)

    """

    def __init__(self, root_path, augmentation=None, preprocessing=None):
        super(UltraSoundDataSet, self).__init__()

        self.ids = sorted(os.listdir(root_path))

        self.image_fps = [os.path.join(root_path, path)
                          for path in self.ids if "mask" not in path]
        self.mask_fps = [os.path.join(root_path, path)
                         for path in self.ids if "mask" in path]

        self.augmentation = augmentation
        self.preprocessing = preprocessing
        
        print('Ultrasound Dataset successfully initialized')

        print('Number of images in the dataset is : {0}'.format(self.__len__()))

    def __getitem__(self, idx):

        image = cv2.imread(self.image_fps[idx])
        mask = cv2.imread(self.mask_fps[idx], cv2.IMREAD_GRAYSCALE)  # mask: single channel
        
        if self.augmentation is not None:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample["image"], sample["mask"]

        if self.preprocessing is not None:
            image = self.preprocessing(image)
            mask = self.preprocessing(mask)
            
        return image, mask

    def __len__(self):
        return len(self.image_fps)
    
    


if __name__ == '__main__':
    
    """
    Test Case: Initialize the BUSI dataset, with augmentation and preprocessing
    """

    from torch.utils.data import DataLoader
    from dataio.transformation.augmentation import Augmentation
    from dataio.transformation.preprocessing import Preprocessing
    import matplotlib.pyplot as plt
    from utils.util import tensor_to_img
    import torch
    
    augmentation = Augmentation('BUSI').get_augmentation()['train']
    preprocessing = Preprocessing('BUSI')
    preprocessing.size = 256
    
    
    dataset = UltraSoundDataSet(
        "/Users/andywang/Desktop/Dataset_BUSI_with_GT/train", augmentation=augmentation, preprocessing=preprocessing.get_preprocessing())
    

    image, mask = dataset[1]
    print(image.type)
    print(mask.type)
    
    assert image.shape == (3, 256, 256) # Tensor Shape for image: (3, 256, 256)
    assert mask.shape == (1, 256, 256) # Tensor Shape for image: (1, 256, 256)
    
    image = tensor_to_img(image)
    mask = tensor_to_img(mask)
    
    print(image)
    print(mask)
    
    counter = ((mask > 0) & (mask < 1)).sum()
    print(counter)

    plt.imshow(image, cmap='gray')
    plt.imshow(torch.from_numpy(mask), cmap='jet', alpha=0.5)
    plt.waitforbuttonpress()

    data = DataLoader(dataset=dataset, num_workers=1, batch_size=2)
