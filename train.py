"""Main training Module"""

from tqdm import tqdm
from torch.utils.data import DataLoader
import numpy as np

from dataio.loader import get_dataset
from dataio.transformation import get_dataset_transformation, get_dataset_preprocessing
from model.models import get_model
from utils.util import json_file_to_pyobj
from torch.utils.tensorboard import SummaryWriter



def train(arguments):
    
    # parse input arguments
    
    json_filename = arguments.config

    # load options (data, preprocessing, augmentation, model, training) from json file

    json_opts = json_file_to_pyobj(json_filename)
    
    # load training parameters
    
    data_opts = json_opts.data
    pre_opts = json_opts.preprocessing
    aug_opts = json_opts.augmentation
    model_opts = json_opts.model
    train_opts = json_opts.training
        
    # load dataset type and path
    dataset_type = data_opts.name

    dataset = get_dataset(dataset_type)
    train_path = data_opts.train_path
    test_path = data_opts.test_path
    
    # load preprocessing
    preprocessing = get_dataset_preprocessing(dataset_type, opts=pre_opts)
    
    # load augmentation
    
    train_augmentation = get_dataset_transformation(dataset_type, opts = aug_opts)['train']
    test_augmentation = get_dataset_transformation(dataset_type, opts = aug_opts)['test']
    
    
    # load training and testing dataset
        
    train_dataset = dataset(train_path, augmentation = train_augmentation, preprocessing = preprocessing)
    train_dataloader = DataLoader(dataset=train_dataset, num_workers=0, batch_size = train_opts.batchsize, shuffle = True)
        
    test_dataset = dataset(test_path, augmentation = test_augmentation, preprocessing = preprocessing)
    test_dataloader = DataLoader(dataset = test_dataset, num_workers = 0, batch_size = train_opts.batchsize, shuffle = True)
                
    model = get_model(model_opts)
    writer = SummaryWriter()
    
    # training process
        
    # loop over the dataset multiple times
    for epoch in range(1, train_opts.num_epochs + 1):
        print('(epoch: %d, total # iters: %d)' % (epoch, len(train_dataloader)))
            
        # training iteration
        for batch_idx, (image, mask) in tqdm(enumerate(train_dataloader, 0), total = len(train_dataloader)):
            model.set_input(image, mask)
            model.forward('train')
            error = model.get_segmentation_stats(split = 'train')
            writer.add_scalar('Train/{}'.format(model_opts.criterion), error['Segmentation Loss'], epoch * len(train_dataloader) + batch_idx)
                
                
        # testing iteration
        for batch_idx, (image, mask) in tqdm(enumerate(test_dataloader, 0), total = len(test_dataloader)):
            model.set_input(image, mask)
            model.forward('test')
            stats = model.get_segmentation_stats(split = 'test')
            
            img_batch = np.zeros((train_opts.batchsize, 3, 256, 256))
            mask_batch = np.zeros((train_opts.batchsize, 1, 256, 256))
            img_batch = image * 255
            mask_batch = model.pred_seg * 255
            
             
            writer.add_scalar('Test/Overall Accuracy', stats['Binary Accuracy'], epoch * len(train_dataloader) + batch_idx)
            writer.add_scalar('Test/Dice Score', stats['Dice Score'], epoch * len(train_dataloader) + batch_idx)
            writer.add_scalar('Test/Jaccard Index', stats['Jaccard Index'], epoch * len(train_dataloader) + batch_idx)
            writer.add_scalar('Test/F1 Score', stats['F1 Score'], epoch * len(train_dataloader) + batch_idx)
            writer.add_scalar('Test/Recall', stats['Recall'], epoch * len(train_dataloader) + batch_idx)
            writer.add_scalar('Test/Precision', stats['Jaccard Index'], epoch * len(train_dataloader) + batch_idx)
            writer.add_scalar('Test/Specificity', stats['Specificity'], epoch * len(train_dataloader) + batch_idx)


            writer.add_images(
                tag = 'segmentation batch', 
                img_tensor = img_batch, 
                global_step = epoch * len(train_dataloader) + batch_idx, 
            )
            
            writer.add_images(
                tag = 'mask batch',
                img_tensor = mask_batch, 
                global_step = epoch * len(train_dataloader) + batch_idx, 
            )
                
        model.update_learning_rate()
        
        if epoch % train_opts.save_epoch_freq == 0:
            model.save(epoch)
            
            
    print("Finished Training")
    writer.close()
    

    
if __name__ == '__main__':
    
    """
    Training Function: e.g. python train.py -c config/config_unet.json

    """
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Seg Training Function')

    parser.add_argument(
        '-c', '--config', help='training config file', required=True)
    
    args = parser.parse_args()

    
    train(args)
    
    