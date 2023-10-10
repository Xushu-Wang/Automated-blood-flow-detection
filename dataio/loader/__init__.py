from .dataset import UltraSoundDataSet, PhantomDataSet


def get_dataset(type):
    
    """
    Get DataSet

    Args:
        type (str): type of the dataset
    """
    
    
    return {
        'BUSI': UltraSoundDataSet, 
        'Phantom': PhantomDataSet
    }[type]
    
def get_dataset_path(type, opts):
    
    """
    Get DataSet Path

    Returns:
        str: path
    """
    
    return getattr(opts, type)
