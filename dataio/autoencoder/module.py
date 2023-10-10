import torch.nn as nn
import torch


class Encoder(nn.Module):
    
    """
    Encoder Block in AutoEncoder

    Args:
        num_input_channels (int): Number of input channels of the image. For BUSI, this parameter is 3
        base_channel_size (int): Number of channels we use in the first convolutional layers. Deeper layers might use a duplicate of it.
        latent_dim (int): Dimensionality of latent representation z
        act_fn (object, optional): Activation function used throughout the encoder network. Defaults to nn.GELU.
        
    """
    
    def __init__(self, num_input_channels: int, base_channel_size: int, latent_dim: int, act_fn: object = nn.GELU):
        super().__init__()
        
        self.hidden = base_channel_size
        
        self.net = nn.Sequential(
            nn.Conv2d(num_input_channels, self.hidden, kernel_size=3, padding=1, stride=2),  # 32x32 => 16x16
            act_fn(),
            nn.Conv2d(self.hidden, self.hidden, kernel_size=3, padding=1),
            act_fn(),
            nn.Conv2d(self.hidden, 2 * self.hidden, kernel_size=3, padding=1, stride=2),  # 16x16 => 8x8
            act_fn(),
            nn.Conv2d(2 * self.hidden, 2 * self.hidden, kernel_size=3, padding=1),
            act_fn(),
            nn.Conv2d(2 * self.hidden, 2 * self.hidden, kernel_size=3, padding=1, stride=2),  # 8x8 => 4x4
            act_fn(),
            nn.Flatten(),  # Image grid to single feature vector
            nn.Linear(2 * 16 * self.hidden, latent_dim),
        )
        
    def forward(self, x):
        return self.net(x)
    
    
class Decoder(nn.Module):
    
    """
    Decoder Block for AutoEncoder

    Args:
        num_input_channels (int): Number of input channels of the image. For BUSI, this parameter is 3
        base_channel_size (int): Number of channels we use in the first convolutional layers. Deeper layers might use a duplicate of it.
        latent_dim (int): Dimensionality of latent representation z
        act_fn (object, optional): Activation function used throughout the encoder network. Defaults to nn.GELU.
    """
    
    
    def __init__(self, num_input_channels: int, base_channel_size: int, latent_dim: int, act_fn: object = nn.GELU):  
        
        self.hidden = base_channel_size
        self.linear = nn.Sequential(
            nn.Linear(latent_dim, 2 * 16 * self.hidden),
            act_fn
        )
        
        self.net = nn.Sequential(
            nn.ConvTranspose2d(
                2 * self.hidden, 2 * self.hidden, kernel_size=3, output_padding=1, padding=1, stride=2
            ),  # 4x4 => 8x8
            act_fn(),
            nn.Conv2d(2 * self.hidden, 2 * self.hidden, kernel_size=3, padding=1),
            act_fn(),
            nn.ConvTranspose2d(2 * self.hidden, self.hidden, kernel_size=3, output_padding=1, padding=1, stride=2),  # 8x8 => 16x16
            act_fn(),
            nn.Conv2d(self.hidden, self.hidden, kernel_size=3, padding=1),
            act_fn(),
            nn.ConvTranspose2d(
                self.hidden, num_input_channels, kernel_size=3, output_padding=1, padding=1, stride=2
            ),  # 16x16 => 32x32
            nn.Tanh(),  # The input images is scaled between -1 and 1, hence the output has to be bounded as well
        )
        
    def forward(self, x):
        x = self.linear(x)
        x = x.reshape(x.shape[0], -1, 4, 4)
        x = self.net(x)
        return x