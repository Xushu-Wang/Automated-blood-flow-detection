from tqdm import tqdm
from torch.utils.data import DataLoader

from dataio.loader import get_dataset
from dataio.transformation import get_dataset_transformation, get_dataset_preprocessing
from model.models import get_model
from utils.util import json_file_to_pyobj


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
    
    # run the dataset for two folds 
    
    for fold in range(7):
        
        # load training and testing dataset
        
        train_dataset = dataset(train_path, augmentation = train_augmentation, preprocessing = preprocessing)
        train_dataloader = DataLoader(dataset=train_dataset, num_workers=0, batch_size = train_opts.batchsize, shuffle = True)
        
        test_dataset = dataset(test_path, augmentation = test_augmentation, preprocessing = preprocessing)
        test_dataloader = DataLoader(dataset = test_dataset, num_workers = 0, batch_size = train_opts.batchsize, shuffle = True)
        
        scores = {}
        
        model = get_model(model_opts)
    
        # training process
        
        for epoch in range(train_opts.num_epochs):
            print('(epoch: %d, total # iters: %d)' % (epoch, len(train_dataloader)))
            
            # training iteration
            
            for epoch_iter, (image, mask) in tqdm(enumerate(train_dataloader, 0), total = len(train_dataloader)):
                for i in range(train_opts.seq_len):
                    model.set_input(image[i], mask[i])
                    model.forward('train')
                    model.optimize_parameters()
                    errors = model.get_current_errors()
                    stats = model.get_segmentation_stats()
                
            # testing iteration

            for epoch_iter, (image, mask) in tqdm(enumerate(test_dataloader, 0), total = len(test_dataloader)):
                for i in range(train_opts.seq_len):
                    model.set_input(image[i], mask[i])
                    model.validate()
                    errors = model.get_current_errors()
                    stats = model.get_segmentation_stats()
                        
            if epoch % train_opts.save_epoch_freq:
                model.save_fold(epoch, fold)
                
            model.update_learning_rate()
            
        print("Finished Trainig for fold %d", fold)
        
        del model
        del train_loader, train_dataset
        del test_loader, test_dataset
    
    

    
if __name__ == '__main__':
    
    """
    Training Function: e.g. python train.py -c config/config_dopus.json

    """
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Seg Training Function')

    parser.add_argument(
        '-c', '--config', help='training config file', required=True)
    
    args = parser.parse_args()

    
    train(args)
    
    