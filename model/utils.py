"""Training Utility Function"""

import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
from torchmetrics import Dice
from torchmetrics.classification import (
    BinaryAccuracy, 
    JaccardIndex,
    F1Score,
    Specificity,
    Precision,
    Recall
)
from torch.nn import CrossEntropyLoss, BCELoss, BCEWithLogitsLoss




def init_weights(net, init_type = 'normal', gain = 0.2):
    
    """
    Weight Initialization

    Args:
        net (_type_): Network Module
        init_type (str, optional): including xavier, kaiming, normal, and orthogonal. Defaults to 'normal'.
        gain (float, optional): Defaults to 0.2.
    """
    
    def init_func(m):
        classname = m.__class__.__name__
        
        if hasattr(m, 'weight') and classname.find('Conv') != -1 or classname.find('Linear') != -1:
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'kaiming':
                init.kaiming_normal(m.weight.data, a = 0, mode='fan_in')
            elif init_type == 'xavier':
                init.xavier_normal(m.weight.data, gain)
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)      
            if hasattr(m, 'bias'):
                init.constant_(m.bias.data, 0.0)  
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)
        
    print('initialize network with %s' %init_type)
    net.apply(init_func)
    
    
def get_optimizer(option, params):
    
    """Get Optimizer Function from json objects

    Raises:
        NotImplementedError: Only support SGD/Adam optimizer

    Returns:
        torch.optim object: optimizer for CNN model
    """
    
    opt = 'adam' if not hasattr(option, 'optim') else option.optim
    
    if opt == 'sgd':
        optimizer = optim.SGD(params, 
                            option.lr_rate,
                            momentum=0.9,
                            nesterov=True)
        
    elif opt == 'adam':
        optimizer = optim.Adam(params,
                            option.lr_rate,
                            betas = (0.9, 0.999))
        
    else:
        raise NotImplementedError
        
    return optimizer


def get_scheduler(optimizer, opt):
    
    """Get Scheduler Function

    Raises:
        NotImplementedError: only support step learning rate/ReduceLROnPlateau scheduler 
    """
        
    if opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(
            optimizer,
            step_size=opt.lr_decay_iters,
            gamma=0.5
        )
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode = 'min',
            factor = 0.1,
            threshold = 0.01,
            patience = 5
        )
    else:
        raise NotImplementedError
        
    return scheduler
        

def dice_coefficient(pred, target, smooth=1e-5):
    intersection = torch.sum(pred * target)
    union = torch.sum(pred) + torch.sum(target)
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice.item()
        

def segmentation_stats(prediction, target):
    
    """Get current segmentation statistics, including binary accuracy, dice score, and jaccard index"""
    
    length = len(prediction)
    accuracy = 0
    dice_score = 0
    jaccard = 0
    f1 = 0
    recall = 0
    precision = 0
    specificity = 0
    
    for i in range(length):
        mask = prediction[i]
        label = target[i]
        
        accuracy += BinaryAccuracy()(mask, label)
        f1 += F1Score(task="binary")(mask, label)
        specificity += Specificity(task="multiclass", average='macro', num_classes=2)(mask, label)
        precision += Precision(task="multiclass", average='macro', num_classes=2)(mask, label)
        recall += Recall(task="multiclass", average='macro', num_classes=2)(mask, label)
        jaccard = jaccard + JaccardIndex(task="multiclass", num_classes=2)(mask, label)
        dice_score += dice_coefficient(mask, label)

    
    return accuracy/length, dice_score/length, f1/length, specificity/length, precision/length, recall/length, jaccard/length, 



def get_criterion(opts):
    
    """Get Loss function for training, avaliable function includes cross entropy, binary dice loss, and binary cross entropy"""
    
    if opts.criterion == 'cross_entropy':
        criterion = CrossEntropyLoss()
    elif opts.criterion == 'binary_dice_loss':
        criterion = BCEWithLogitsLoss()  # BinaryDiceLoss()
    elif opts.criterion == 'binary_cross_entropy':
        criterion = BCELoss()
    else:
        raise NotImplementedError

    return criterion


def print_network(net):
    
    """Print network architecture with the number of parameters"""
    
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)
    
    