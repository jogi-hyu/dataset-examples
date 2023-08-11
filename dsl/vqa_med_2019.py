import os
import numpy as np
from PIL import Image

from dsl_constants import Type, Set

# read_image(DSLC.TRAIN, 19884, '../sample_datas/VQA-Med-2019')
def read_image(set, id: int, base_dir: str, return_type=Type.NUMPY):
    """
    Read image from base_dir with id
    :param id: image id
    :param base_dir: base directory of images
    :param set: train, val, test
    :param return_type: return type of image
    :return: image
    """
    if set == Set.TRAIN:
        directory = os.path.join(base_dir,'ImageClef-2019-VQA-Med-Training','Train_images')
    elif set == Set.VAL:
        directory = os.path.join(base_dir,'ImageClef-2019-VQA-Med-Validation','Val_images')
    elif set == Set.TEST:
        raise Exception('Test set is not available')
    else:
        raise Exception('Invalid set')

    image_path = os.path.join(directory, f'synpic{str(id)}.jpg')
    if not os.path.exists(image_path):
        raise Exception(f'No image with path: {image_path}')
    image = Image.open(image_path)

    if return_type == Type.NUMPY:
        return np.array(image)
    elif return_type == Type.PIL:
        return image
    

import matplotlib.pyplot as plt
plt.imshow(read_image(Set.TRAIN, 19884, '/home/jogi/projects/dataset-loader/sample_datas/VQA-Med-2019'))