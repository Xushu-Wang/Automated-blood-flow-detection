import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
import torch
import torch.nn.functional as F
import albumentations as A


def add_gaussian_noise(image, noise_level):
    
    transform = [
        A.transforms.GaussNoise(p=1, var_limit=(noise_level, noise_level), mean=0)
    ]
    
    transform = A.Compose(transform)
    
    image = transform(image)
    
    return image
    

def predict_mask(image, net):
    
    """Predict Mask based on random ultrasound image (image -> image (mask))"""
    
    image = image.resize((256, 256))
    image_np = np.array(image)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=0,
            std=255
        )
    ])

    image_tensor = transform(image_np)
    pred = net(image_tensor.unsqueeze(0))
    mask = torch.round(F.sigmoid(pred.squeeze())) * 255
        
    return mask.detach().numpy()
    

def visualize(image, mask):
    """Visualize Mask on Image (Needs to be same size)"""
    
    plt.imshow(image, cmap='gray')
    plt.imshow(mask, cmap='jet', alpha=0.5)
    plt.show()
    

def tensor2img():
    pass
