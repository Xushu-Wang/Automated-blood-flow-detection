import torch
import torch.nn as nn
from model.modules.cbam import EncoderBlock, DecoderBlock
from model.modules.attention import AttentionBlock



class AttU_Net(nn.Module):
    """
    Attention U Net model for medical image segementation
    
    Arguments:
        ch_img -- Input shape 
        n_filters -- Number of filters for the convolutional layers
        ch_out -- Number of output classes
        
    """
    
    
    def __init__(self, ch_img = 3, ch_out = 1, n_filter = 64):
        
        super(AttU_Net, self).__init__()
        
        self.conv1 = EncoderBlock(ch_in=ch_img, ch_out=n_filter)
        self.conv2 = EncoderBlock(ch_in=n_filter, ch_out=n_filter * 2)
        self.conv3 = EncoderBlock(ch_in=n_filter * 2, ch_out=n_filter * 4)
        self.conv4 = EncoderBlock(ch_in=n_filter * 4, ch_out=n_filter * 8)
        self.conv5 = EncoderBlock(ch_in=n_filter * 8, ch_out=n_filter * 16)
        
        self.up5 = DecoderBlock(ch_in=n_filter * 16, ch_out=n_filter * 8)
        self.attention5 = AttentionBlock(F_g=n_filter * 8, F_l=n_filter*8, F_int=n_filter*4)
        self.upconv5 = EncoderBlock(ch_in=n_filter * 16, ch_out=n_filter * 8)
                
        self.up4 = DecoderBlock(ch_in=n_filter * 8, ch_out=n_filter * 4)
        self.attention4 = AttentionBlock(F_g=n_filter * 4, F_l=n_filter*4, F_int=n_filter*2)
        self.upconv4 = EncoderBlock(ch_in=n_filter * 8, ch_out=n_filter * 4)

        self.up3 = DecoderBlock(ch_in=n_filter * 4, ch_out=n_filter * 2)
        self.attention3 = AttentionBlock(F_g=n_filter * 2, F_l=n_filter*2, F_int=n_filter)
        self.upconv3 = EncoderBlock(ch_in=n_filter * 4, ch_out=n_filter * 2)

        self.up2 = DecoderBlock(ch_in=n_filter * 2, ch_out=n_filter)
        self.attention2 = AttentionBlock(F_g=n_filter, F_l=n_filter, F_int=n_filter/2)
        self.upconv2 = EncoderBlock(ch_in=n_filter * 2, ch_out=n_filter)

        
        self.out = nn.Conv2d(n_filter, ch_out, kernel_size=1, stride=1, padding=0)
        
        
        self.maxpooling = nn.MaxPool2d(kernel_size = 2, stride = 2)
        
        
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.maxpooling(x)
        
        x2 = self.conv2(x)
        x3 = self.maxpooling(x)
        
        x3 = self.conv3(x)
        x4 = self.maxpooling(x)
        
        x4 = self.conv4(x)
        x5 = self.maxpooling(x)
        
        x5 = self.conv5(x)
        
        d5 = self.up5(x5)
        x4 = self.attention5(g = d5, x = x4)
        d5 = torch.cat((x4, d5), dim = 1)
        d5 = self.upconv5(d5)
        
        d4 = self.up4(d5)
        x3 = self.attention4(g = d4, x = x3)
        d4 = torch.cat((x3, d4), dim = 1)
        d4 = self.upconv4(d4)
        
        d3 = self.up3(d4)
        x2 = self.attention3(g = d3, x = x2)
        d3 = torch.cat((x2, d3), dim = 1)
        d3 = self.upconv3(d3)
        
        d2 = self.up2(d3)
        x1 = self.attention4(g = d2, x = x1)
        d2 = torch.cat((x1, d2), dim = 1)
        d2 = self.upconv2(d2)
        
        d1 = self.out(d2)
        
        return d1

        
        