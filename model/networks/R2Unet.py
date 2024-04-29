import torch.nn as nn
from model.modules.convgru import RecurrentBlock, RCNNBlock
from model.modules.cbam import EncoderBlock
import torch



class R2U_Net(nn.Module):
    
    """
    
    Recurrent Residual U Net (R2U Net) Architecture
    
    """
    
    
    def __init__(self, ch_img = 3, ch_out = 1, n_filter = 64, t=2):
        super().__init__()
        
        self.RCNN1 = RCNNBlock(ch_in=ch_img, ch_out=n_filter, t=t)
        self.RCNN2 = RCNNBlock(ch_in=n_filter, ch_out=n_filter * 2, t=t)
        self.RCNN3 = RCNNBlock(ch_in=n_filter * 2, ch_out=n_filter * 4, t=t)
        self.RCNN4 = RCNNBlock(ch_in=n_filter * 4, ch_out=n_filter * 8, t=t)
        self.RCNN5 = RCNNBlock(ch_in=n_filter * 8, ch_out=n_filter * 16, t=t)
        
        self.up5 = EncoderBlock(ch_in=n_filter * 16, ch_out=n_filter * 8)
        self.upRCNN5 = RCNNBlock(ch_in=n_filter * 16, ch_out=n_filter * 8, t=t)
        
        self.up4 = EncoderBlock(ch_in=n_filter * 8, ch_out=n_filter * 4)
        self.upRCNN4 = RCNNBlock(ch_in=n_filter * 8, ch_out=n_filter * 4, t=t)
        
        self.up3 = EncoderBlock(ch_in=n_filter * 4, ch_out=n_filter * 2)
        self.upRCNN3 = RCNNBlock(ch_in=n_filter * 4, ch_out=n_filter * 2, t=t)
        
        self.up2 = EncoderBlock(ch_in=n_filter * 2, ch_out=n_filter)
        self.upRCNN2 = RCNNBlock(ch_in=n_filter * 2, ch_out=n_filter, t=t)
        
        self.upsampling = nn.Upsample(scale_factor=2)
        
        self.maxpooling = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.out = nn.Conv2d(n_filter, ch_out, kernel_size=1, stride=1, padding=0)
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.maxpooling(x1)
        
        x2 = self.conv2(x2)
        x3 = self.maxpooling(x2)
        
        x3 = self.conv3(x3)
        x4 = self.maxpooling(x3)
        
        x4 = self.conv4(x4)
        x5 = self.maxpooling(x4)
        
        x5 = self.conv5(x5)
        
        d5 = self.up5(x5)
        d5 = torch.cat((x4, d5), dim = 1)
        d5 = self.upRCNN5(d5)
        
        d4 = self.up4(d5)
        d4 = torch.cat((x3, d4), dim = 1)
        d4 = self.upRCNN4(d4)
        
        d3 = self.up3(d4)
        d3 = torch.cat((x2, d3), dim = 1)
        d3 = self.upRCNN3(d3)
        
        d2 = self.up2(d3)
        d2 = torch.cat((x1, d2), dim = 1)
        d2 = self.upRCNN2(d2)
        
        d1 = self.out(d2)
        
        return self.sigmoid(d1)
        