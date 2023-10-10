import torch
import torch.nn as nn
import torch.nn.functional as F

"""Convolution related Module"""


class BasicConv(nn.Module):
    
    """
    
    Basic Convolution Block
    
    Including:
        2D Convolutional Layer: kernel size = 3, stride = 1, padding = 1
        Batch Normalization Layer
        ReLU activation
    
    """
    
    def __init__(self, ch_in, ch_out):
        super(BasicConv, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=ch_in, out_channels=ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace = True)
        )
        
    def forward(self, x):
        x = self.conv(x)
        return x


class EncoderBlock(nn.Module):
    
    """
    
    Encoder Block for U-Net Architecture
    
    Including double convolutional block
    
    """
    
    
    def __init__(self, ch_in, ch_out):
        super(EncoderBlock, self).__init__()
        self.encode = nn.Sequential(
            BasicConv(ch_in, ch_out),
            BasicConv(ch_out, ch_out)
        )
        
    def forward(self, x):
        x = self.encode(x)
        return x


class DecoderBlock(nn.Module):
    
    """
    
    Decoder Block (Upconvolutional Block) for U-Net Architecture
    
    Including:
        UpSampling Block 
        Double Convolutional Block
    
    """
    
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            BasicConv(ch_in, ch_out)
        )
        
    def forward(self, x):
        x = self.up(x)
        return x

    
    
class Flatten(nn.Module):
    
    """ 2D Flatten Layer """
    
    def forward(self, x):
        return x.view(x.size(0), -1)
    
    
class ChannelGate(nn.Module):
    
    """
    
    Channel Gate for CBAM (Convolutional Block Attention Module)
    
    Including:
        Multilayer perceptron
        Average Pooling/Max Pooling layer
    
    """
    
    
    def __init__(self, gate_channels, reduction_ratio=4, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
        )
        self.pool_types = pool_types
        
    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = F.avg_pool2d(
                    x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == 'max':
                max_pool = F.max_pool2d(
                    x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(max_pool)
            elif pool_type == 'lp':
                lp_pool = F.lp_pool2d(
                    x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(lp_pool)
            elif pool_type == 'lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp(lse_pool)

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = torch.sigmoid(channel_att_sum).unsqueeze(
            2).unsqueeze(3).expand_as(x)
        return x * scale


def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs
    
    

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


    
class SpatialGate(nn.Module):
    
    """
    
    Spatial Gate for CBAM (Convolutional Block Attention Module)
    
    Including:
        Channel Pooling Layer
        Basic Convolutional Block
    
    """
    
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(
            kernel_size-1) // 2, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out)  
        return x * scale
    
    
class CBAM(nn.Module):
    
    """
    
    CBAM (Convolutional Block Attention Module): Channel Gate + Spatial Gate
    
    """
    
    
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(
            gate_channels, reduction_ratio, pool_types)
        self.no_spatial = no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()

    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out