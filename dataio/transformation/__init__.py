import json
from .augmentation import Augmentation 
from .preprocessing import Preprocessing

def get_dataset_preprocessing(name, opts = None):
    
    """
    Get DataSet Preprocessing

    Returns:
        name: transformation name
        opts: transformation parameters from json
    """
    
    pre_obj = Preprocessing(name)
    
    if opts is not None:
        pre_obj.initialize(opts)
    
    pre_obj.print()
        
    return pre_obj.get_preprocessing()



def get_dataset_transformation(name, opts = None):
    
    """
    Get DataSet Transformation

    Returns:
        name: transformation name
        opts: transformation parameters from json
    """
    
    
    trans_obj = Augmentation(name)
    
    if opts is not None:
        trans_obj.initialize(opts)
        
    trans_obj.print()
        
    return trans_obj.get_augmentation()
    