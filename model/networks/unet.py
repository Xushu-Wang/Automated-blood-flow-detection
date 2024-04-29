import torch
import torch.nn as nn
import torch.nn.functional as F
from model.modules.cbam import EncoderBlock, DecoderBlock



class U_Net(nn.Module):
    
    """
    U Net model for medical image segementation
    
    Arguments:
        ch_img -- Input shape 
        n_filters -- Number of filters for the convolutional layers
        ch_out -- Number of output classes
        
    """
    
    
    def __init__(self, ch_img = 3, ch_out = 1, n_filter = 64):
        
        super(U_Net, self).__init__()
        
        self.conv1 = EncoderBlock(ch_in=ch_img, ch_out=n_filter)
        self.conv2 = EncoderBlock(ch_in=n_filter, ch_out=n_filter * 2)
        self.conv3 = EncoderBlock(ch_in=n_filter * 2, ch_out=n_filter * 4)
        self.conv4 = EncoderBlock(ch_in=n_filter * 4, ch_out=n_filter * 8)
        self.conv5 = EncoderBlock(ch_in=n_filter * 8, ch_out=n_filter * 16)
        
        self.up5 = DecoderBlock(ch_in=n_filter * 16, ch_out=n_filter * 8)
        self.upconv5 = EncoderBlock(ch_in=n_filter * 16, ch_out=n_filter * 8)
        
        self.up4 = DecoderBlock(ch_in=n_filter * 8, ch_out=n_filter * 4)
        self.upconv4 = EncoderBlock(ch_in=n_filter * 8, ch_out=n_filter * 4)

        self.up3 = DecoderBlock(ch_in=n_filter * 4, ch_out=n_filter * 2)
        self.upconv3 = EncoderBlock(ch_in=n_filter * 4, ch_out=n_filter * 2)

        self.up2 = DecoderBlock(ch_in=n_filter * 2, ch_out=n_filter)
        self.upconv2 = EncoderBlock(ch_in=n_filter * 2, ch_out=n_filter)

        
        self.out = nn.Conv2d(n_filter, ch_out, kernel_size=1, stride=1, padding=0)
        
        
        self.maxpooling = nn.MaxPool2d(kernel_size = 2, stride = 2)
        
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
        d5 = self.upconv5(d5)
        
        d4 = self.up4(d5)
        d4 = torch.cat((x3, d4), dim = 1)
        d4 = self.upconv4(d4)
        
        d3 = self.up3(d4)
        d3 = torch.cat((x2, d3), dim = 1)
        d3 = self.upconv3(d3)
        
        d2 = self.up2(d3)
        d2 = torch.cat((x1, d2), dim = 1)
        d2 = self.upconv2(d2)
        
        d1 = self.out(d2)
        
        return self.sigmoid(d1)

        
if __name__ == '__main__':
    
    """Test Case for U Net
    """
    
    # Create an instance of the network
    model = U_Net()

    # Assuming you have an input tensor with shape (batch_size, channels, height, width)
    input_tensor = torch.rand((16, 3, 224, 224))

    # Pass the input through the model
    output = model(input_tensor)
    
    pred = torch.round(output) * 255
    
    print(input_tensor)
    print(output)
    print(pred)

        