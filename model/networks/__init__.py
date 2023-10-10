from .unet import U_Net
from .R2Att_Unet import R2AttU_Net
from .R2Unet import R2U_Net
from .attention_unet import AttU_Net




def get_network(name, in_channels = 2, out_channels = 1):
    network_instance = get_network_type(name)
    network_instance = network_instance(ch_img = in_channels, ch_out = out_channels)
    
    return network_instance


def get_network_type(name):
    return {
        'unet': U_Net,
        'attention unet': AttU_Net,
        'recurrent residual unet': R2U_Net,
        'recurrent residual attention unet': R2AttU_Net
    }[name]