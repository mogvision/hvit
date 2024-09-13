import os, glob
import monai
import torch
import pickle
import random
import numpy as np
from torch.utils.data import Dataset


class OASIS_Dataset(Dataset):
    def __init__(self, input_dim, data_path, num_steps=1000, is_pair: bool = False, ext="pkl"):
        self.paths = glob.glob(os.path.join(data_path, f"*.{ext}"))
        self.num_steps = num_steps
        self.input_dim = input_dim
        self.is_pair = is_pair

        self.transforms_mask = monai.transforms.Compose([
            monai.transforms.Resize(spatial_size=input_dim, mode="nearest")
        ])
        self.transforms_image = monai.transforms.Compose([
            monai.transforms.Resize(spatial_size=input_dim)
        ])

    def _pkload(self, filename: str) -> tuple:
        """
        Load a pickled file and return its contents.

        Args:
            filename (str): The path to the pickled file.

        Returns:
            tuple: The unpickled contents of the file.

        Raises:
            FileNotFoundError: If the file does not exist.
            pickle.UnpicklingError: If there's an error during unpickling.
        """
        try:
            with open(filename, 'rb') as file:
                return pickle.load(file) #np.ascontiguousarray(pickle.load(file)) 
        except FileNotFoundError:
            raise FileNotFoundError(f"The file {filename} was not found.")
        except pickle.UnpicklingError:
            raise pickle.UnpicklingError(f"Error unpickling the file {filename}.")
    
    def __getitem__(self, index):
        if self.is_pair:
            src, tgt, src_lbl, tgt_lbl = self._pkload(self.paths[index])
        else:
            selected_items = random.sample(list(self.paths), 2)
            src, src_lbl = self._pkload(selected_items[0])
            tgt, tgt_lbl = self._pkload(selected_items[1])

        src = torch.from_numpy(src).float().unsqueeze(0)
        src_lbl = torch.from_numpy(src_lbl).long().unsqueeze(0)
        tgt = torch.from_numpy(tgt).float().unsqueeze(0)
        tgt_lbl = torch.from_numpy(tgt_lbl).long().unsqueeze(0)

        src = self.transforms_image(src)
        tgt = self.transforms_image(tgt)
        src_lbl = self.transforms_mask(src_lbl)
        tgt_lbl = self.transforms_mask(tgt_lbl)

        return src, tgt, src_lbl, tgt_lbl

    def __len__(self):
        return self.num_steps if not self.is_pair else len(self.paths)

def get_dataloader(data_path, input_dim, batch_size, shuffle: bool = True, is_pair: bool = False):
    ds = OASIS_Dataset(input_dim = input_dim, data_path = data_path, is_pair=is_pair)
    dataloader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=4)
    return dataloader
