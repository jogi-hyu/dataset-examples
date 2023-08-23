import pandas as pd

from dsl_core import Dataset, Set

sample_dir = "../sample_datas/VQA-Med-2019"

class VQAMed2019(Dataset):
    """
    ### Description
        VQA-Med-2019 dataset loader
    """

    def __init__(self, base_dir: str = sample_dir):
        super().__init__(base_dir)
        self.available_set = ['train', 'val']

        self.category = "VQA"
        self._get_all_images()
        self._get_all_qa_pairs()

        self.all = pd.merge(self._images, self._pairs, on=['set','id'], how='right')
        self.train = self.VQAMed2019Set(self.all[self.all.set == 'train'].sort_values(by='image_path'))
        self.val = self.VQAMed2019Set(self.all[self.all.set == 'val'].sort_values(by='image_path'))

    def _get_item_by_id(self, set_type: str, id: str):
        return self[set_type][self[set_type].id == id]

    def _get_all_images(self):
        directory = {
            'train' : self.base_dir / 'ImageClef-2019-VQA-Med-Training' / 'Train_images',
            'val' : self.base_dir / 'ImageClef-2019-VQA-Med-Validation' / 'Val_images'
        }
        images = []
        for set_type in self.available_set:
            for path in directory[set_type].glob('*.jpg'):
                if path.exists():
                    images.append({'set': set_type, 'id': path.stem.split('synpic')[1], 'image_path': path})

        self._images = pd.DataFrame(images)

    def _get_all_qa_pairs(self):
        directory = {
            'train' : self.base_dir / 'ImageClef-2019-VQA-Med-Training' / 'QAPairsByCategory',
            'val' : self.base_dir / 'ImageClef-2019-VQA-Med-Validation' / 'QAPairsByCategory'
        }
        pairs = pd.DataFrame({'set':[], 'id': [], 'q': [], 'a': [], 'category': []})
        prefix = ['C1_Modality', 'C2_Plane', 'C3_Organ', 'C4_Abnormality']
        for set_type in self.available_set:
            for p in prefix:
                pair_file = directory[set_type] / f'{p}_{set_type}.txt'
                ps = pd.read_csv(pair_file, sep='|', header=None, names=['id', 'q', 'a'])
                ps.id = ps.id.apply(lambda x: x.split('synpic')[1])
                ps['set'] = set_type
                ps['category'] = p.split('_')[-1]
                pairs = pd.concat([pairs, ps], ignore_index=True)

        self._pairs = pairs

    class VQAMed2019Set(Set):
        def __init__(self, df: pd.DataFrame):
            super().__init__(df)
        
        def __getitem__(self, key):
            if type(key) == int:
                return self.df.iloc[key]
            elif type(key) == str:
                return self.df[self.df.id == key]