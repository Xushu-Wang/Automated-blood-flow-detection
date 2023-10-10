import json
import collections
import numpy as np
import cv2


    
def tensor_to_img(tensor):
    
    """Tensor to image function

    Returns:
        image (numpy array)
    """
    
    np_array = tensor.cpu().detach().numpy()

    np_array = np_array * 255
    np_array = np.clip(np_array, 0, 255).astype(np.uint8)

    if np_array.shape[0] == 1:
        image = cv2.cvtColor(np_array[0], cv2.COLOR_GRAY2BGR)
    else:  
        image = cv2.cvtColor(np_array.transpose(1, 2, 0), cv2.COLOR_RGB2BGR)
    return image




def json_file_to_pyobj(filename):
    
    """Deconstruct Json File into Python Objects"""
    
    
    def _json_object_hook(d): return collections.namedtuple(
        'X', d.keys())(*d.values())

    def json2obj(data): return json.loads(data, object_hook=_json_object_hook)
    
    return json2obj(open(filename).read())