import torch.nn as nn
import torch
from model.modules.cbam import EncoderBlock, DecoderBlock
from model.modules.convgru import ConvGRU
from model.utils import init_weights


class DopUS_Net(nn.Module):
    
    """
    
    DopUS Net: from "DopUS-Net: Quality-Aware Robotic Ultrasound Imaging based on Doppler Signal"
    
    """
    
    def __init__(self, feature_scale=8, n_classes=1, is_deconv=True, in_channels=2, is_batchnorm=True):
        super(DopUS_Net, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]
        self.filters = filters

        # downsampling bmode
        self.conv1_bmode = EncoderBlock(2, filters[0])
        self.maxpool1_bmode = nn.MaxPool2d(kernel_size=2)

        self.conv2_bmode = EncoderBlock(filters[0], filters[1])
        self.maxpool2_bmode = nn.MaxPool2d(kernel_size=2)

        self.conv3_bmode = EncoderBlock(filters[1], filters[2])
        self.maxpool3_bmode = nn.MaxPool2d(kernel_size=2)

        self.conv4_bmode = EncoderBlock(filters[2], filters[3])
        self.maxpool4_bmode = nn.MaxPool2d(kernel_size=2)

        self.conv5_bmode = EncoderBlock(filters[3], filters[4])

        # downsampling doppler
        self.conv1_doppler = EncoderBlock(1, filters[0])
        self.maxpool1_doppler = nn.MaxPool2d(kernel_size=2)

        self.conv2_doppler = EncoderBlock(filters[0], filters[1])
        self.maxpool2_doppler = nn.MaxPool2d(kernel_size=2)

        self.conv3_doppler = EncoderBlock(filters[1], filters[2])
        self.maxpool3_doppler = nn.MaxPool2d(kernel_size=2)

        self.conv4_doppler = EncoderBlock(filters[2], filters[3])
        self.maxpool4_doppler = nn.MaxPool2d(kernel_size=2)

        self.conv5_doppler = EncoderBlock(filters[3], filters[4])
        self.convGRUDoppler = ConvGRU(input_size=filters[4], hidden_sizes=[filters[3], filters[3], filters[4]],
                                      kernel_sizes=[3, 3, 5], n_layers=3)
        self.convGRUBmode = ConvGRU(input_size=filters[4], hidden_sizes=[filters[3], filters[3], filters[4]],
                                    kernel_sizes=[3, 3, 5], n_layers=3)

        self.center_concat = EncoderBlock(filters[4] * 2, filters[4])

        self.up_concat4 = unetUp2(filters[4], filters[3], self.is_deconv, self.is_batchnorm)
        self.up_concat3 = unetUp2(filters[3], filters[2], self.is_deconv, self.is_batchnorm)
        self.up_concat2 = unetUp2(filters[2], filters[1], self.is_deconv, self.is_batchnorm)
        self.up_concat1 = unetUp2(filters[1], filters[0], self.is_deconv, self.is_batchnorm)

        # final conv (without any concat)
        self.final = nn.Conv2d(filters[0], n_classes, 1)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')

    def forward(self, inputs, hidden):
        # bmode
        conv1_bmode = self.conv1_bmode(inputs[:, [0, 1], :, :])
        maxpool1_bmode = self.maxpool1_bmode(conv1_bmode)

        conv2_bmode = self.conv2_bmode(maxpool1_bmode)
        maxpool2_bmode = self.maxpool2_bmode(conv2_bmode)

        conv3_bmode = self.conv3_bmode(maxpool2_bmode)
        maxpool3_bmode = self.maxpool3_bmode(conv3_bmode)

        conv4_bmode = self.conv4_bmode(maxpool3_bmode)
        maxpool4_bmode = self.maxpool4_bmode(conv4_bmode)

        center_bmode = self.conv5_bmode(maxpool4_bmode)
        center_bmode = self.convGRUBmode(center_bmode, hidden[0])

        conv1_doppler = self.conv1_doppler(inputs[:, [1], :, :])
        maxpool1_doppler = self.maxpool1_doppler(conv1_doppler)

        conv2_doppler = self.conv2_doppler(maxpool1_doppler)
        maxpool2_doppler = self.maxpool2_doppler(conv2_doppler)

        conv3_doppler = self.conv3_doppler(maxpool2_doppler)
        maxpool3_doppler = self.maxpool3_doppler(conv3_doppler)

        conv4_doppler = self.conv4_doppler(maxpool3_doppler)
        maxpool4_doppler = self.maxpool4_doppler(conv4_doppler)

        center_doppler = self.conv5_doppler(maxpool4_doppler)
        center_doppler = self.convGRUDoppler(center_doppler, hidden[1])

        center = self.center_concat(torch.cat([center_doppler[-1], center_bmode[-1]], 1))

        up4 = self.up_concat4(conv4_bmode, center)
        up3 = self.up_concat3(conv3_bmode, up4)
        up2 = self.up_concat2(conv2_bmode, up3)
        up1 = self.up_concat1(conv1_bmode, up2)
        final = self.final(up1)

        return final, [center_bmode, center_doppler]