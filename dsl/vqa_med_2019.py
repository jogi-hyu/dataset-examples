import pandas as pd
from pathlib import Path
import numpy as np
from PIL import Image
from enum import Enum

from dsl_constants import Type, Set

class Category(Enum):
    """
    ### Description
        Category of VQA-Med-2019 dataset
    """
    ALL         = 3000
    MODALITY    = 3001
    PLANE       = 3002
    ORGAN       = 3003
    ABNORMALITY = 3004

def get_data(set: Set, id: int, base_dir: str, category=Category.ALL, return_type=Type.NUMPY):
    """
    ### Description
        Get data from `base_dir` with `id`\n
        `base_dir` should be the directory of VQA-Med-2019 dataset\n
        The path should be like; `base_dir`/ImageClef-2019-VQA-Med-`set`/`set`_images/synpic`id`.jpg
    ### Parameters
        `set` The type of set you want to load (e.g., `Set.TRAIN`)\n
        `id` Image id\n
        `base_dir` Base directory of data\n
        `category` Category of QA pairs (ALL, MODALITY, PLANE, ORGAN, ABNORMALITY)\n
        `return_type` Return type of image
    ### Returns
        (`np.ndarray` or `PIL.Image`, `pd.DataFrame` with columns `id`: `int`, `q`: `str`, `a`: `str`)
    """
    return get_image(set, id, base_dir, category, return_type), get_qa_pairs(set, id, base_dir, category)

def get_image(set: Set, id: int, base_dir: str, category=Category.ALL, return_type=Type.NUMPY):
    """
    ### Description
        Get image from `base_dir` with `id`\n
        `base_dir` should be the directory of VQA-Med-2019 dataset\n
        The path should be like; `base_dir`/ImageClef-2019-VQA-Med-`set`/`set`_images/synpic`id`.jpg
    ### Parameters
        `set` The type of set you want to load (e.g., `Set.TRAIN`)\n
        `id` Image id\n
        `base_dir` Base directory of data\n
        `category` Category of QA pairs (ALL, MODALITY, PLANE, ORGAN, ABNORMALITY)\n
        `return_type` Return type of image
    ### Returns
        `np.ndarray` or `PIL.Image`
    """
    images = get_all_images(set, base_dir)
    image_path = images[images.id == id].path.values[0]
    image = Image.open(image_path)
    if return_type == Type.NUMPY:
        return np.array(image)
    elif return_type == Type.PIL:
        return image
    
def get_qa_pairs(set: Set, id: int, base_dir: str, category=Category.ALL):
    """
    ### Description
        Get QA pairs from `base_dir` with `id`\n
        `base_dir` should be the directory of VQA-Med-2019 dataset\n
        The path should be like; `base_dir`/ImageClef-2019-VQA-Med-`set`/QAPairsByCategory...
    ### Parameters
        `set` The type of set you want to load (e.g., `Set.TRAIN`)\n
        `id` Image id\n
        `base_dir` Base directory of data\n
        `category` Category of QA pairs (ALL, MODALITY, PLANE, ORGAN, ABNORMALITY)
    ### Returns
        `str`
    """
    pairs = get_all_qa_pairs(set, base_dir, category)
    return pairs[pairs.id == id]

def get_all_images(set: Set, base_dir: str) -> pd.DataFrame: 
    """
    ### Description
        Get all images from `base_dir` with specific `set`\n
        `base_dir` should be the directory of VQA-Med-2019 dataset\n
        The path should be like; `base_dir`/ImageClef-2019-VQA-Med-`set`/QAPairsByCategory...
    ### Parameters
        `set` The type of set you want to load (e.g., `Set.TRAIN`)\n
        `base_dir` Base directory of data\n
    ### Returns
        `pd.DataFrame` with columns `id`: `int`, `image`: `Path`
    """
    base_dir = Path(base_dir)
    if set == Set.TRAIN:
        directory = base_dir / 'ImageClef-2019-VQA-Med-Training' / 'Train_images'
    elif set == Set.VAL:
        directory = base_dir / 'ImageClef-2019-VQA-Med-Validation' / 'Val_images'
    elif set == Set.TEST:
        raise Exception('Test set is not available')
    else:
        raise Exception('Invalid set')
    
    images = []
    for path in directory.glob('*.jpg'):
        if path.exists():
            images.append({'id': int(path.stem.split('synpic')[1]), 'path': path})
    return pd.DataFrame(images)

def get_all_qa_pairs(set: Set, base_dir: str, category=Category.ALL) -> pd.DataFrame:
    """
    ### Description
        Get all pairs from `base_dir` with specific `set`\n
        `base_dir` should be the directory of VQA-Med-2019 dataset\n
        The path should be like; `base_dir`/ImageClef-2019-VQA-Med-`set`/QAPairsByCategory...
    ### Parameters
        `set` The type of set you want to load (e.g., `Set.TRAIN`)\n
        `base_dir` Base directory of data\n
        `category` Category of QA pairs (ALL, MODALITY, PLANE, ORGAN, ABNORMALITY)
    ### Returns
        `pd.DataFrame` with columns `id`: `int`, `q`: `str`, `a`: `str`
    """
    base_dir = Path(base_dir)
    if set == Set.TRAIN:
        directory = base_dir / 'ImageClef-2019-VQA-Med-Training'
        suffix = 'train'
    elif set == Set.VAL:
        directory = base_dir / 'ImageClef-2019-VQA-Med-Validation'
        suffix = 'val'
    elif set == Set.TEST:
        raise Exception('Test set is not available')
    else:
        raise Exception('Invalid set')
    
    if not category == Category.ALL:
        directory = directory / 'QAPairsByCategory'
    
    if category == Category.MODALITY:
        pair_text = directory / f'C1_Modality_{suffix}.txt'
    elif category == Category.PLANE:
        pair_text = directory / f'C2_Plane_{suffix}.txt'
    elif category == Category.ORGAN:
        pair_text = directory / f'C3_Organ_{suffix}.txt'
    elif category == Category.ABNORMALITY:
        pair_text = directory / f'C4_Abnormality_{suffix}.txt'
    else:
        pair_text = directory / f'All_QA_Pairs_{suffix}.txt'
    
    pairs = pd.read_csv(pair_text, sep='|', header=None, names=['id', 'q', 'a'])
    pairs.id = pairs.id.apply(lambda x: int(x.split('synpic')[1]))

    return pairs