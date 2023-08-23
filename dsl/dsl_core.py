import pandas as pd
from pathlib import Path

class Dataset:
    def __init__(self, base_dir: str) -> None:
        self.base_dir = Path(base_dir)
        self.category = None
        self.available_set = []
    
    def _check_set(self, set_type: str):
        if not set_type in Set:
            raise Exception('Invalid set')
        elif not set_type in self.available_set:
            raise Exception('Unavailable set')

class Set:
    def __init__(self, df: pd.DataFrame):
        self.df = df
    
    def __len__(self):
        return len(self.df)