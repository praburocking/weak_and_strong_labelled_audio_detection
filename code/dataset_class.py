#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Tuple, Optional, Union, Dict
from pickle import load as pickle_load
from pathlib import Path

from torch.utils.data import Dataset
import numpy

from file_io import get_files_from_dir_with_pathlib


__docformat__ = 'reStructuredText'
__all__ = ['MyDataset']


class MyDataset(Dataset):

    def __init__(self,
                 data_dir: Union[str, Path],
                 data_parent_dir: Optional[str] = '',
                 key_features: Optional[str] = 'features',
                 key_class: Optional[str] = 'labels',
                 dataset_type="Strong",
                 load_into_memory: Optional[bool] = True) \
            -> None:
        """An example of an object of class torch.utils.data.Dataset

        :param data_dir: Directory to read data from.
        :type data_dir: str
        :param data_parent_dir: Parent directory of the data, defaults\
                                to ``.
        :type data_parent_dir: str
        :param key_features: Key to use for getting the features,\
                             defaults to `features`.
        :type key_features: str
        :param key_class: Key to use for getting the class, defaults\
                          to `class`.
        :type key_class: str
        :param load_into_memory: Load the data into memory? Default to True
        :type load_into_memory: bool
        """
        super().__init__()
        data_path = Path(data_parent_dir, data_dir)
        self.files = get_files_from_dir_with_pathlib(data_path)
        self.dataset_type=dataset_type
        self.load_into_memory = load_into_memory
        self.key_features = key_features
        self.key_class = key_class
        if self.load_into_memory:
            for i, a_file in enumerate(self.files):
                self.files[i] = self._load_file(a_file)
        print(self.dataset_type)
    @staticmethod
    def _load_file(file_path: Path) \
            -> Dict[str, Union[int, numpy.ndarray]]:
        """Loads a file using pathlib.Path

        :param file_path: File path.
        :type file_path: pathlib.Path
        :return: The file.
        :rtype: dict[str, int|numpy.ndarray]
        """
        with file_path.open('rb') as f:
            return pickle_load(f)

    def __len__(self) \
            -> int:
        """Returns the length of the dataset.

        :return: Length of the dataset.
        :rtype: int
        """
        return len(self.files)

    def __getitem__(self,
                    item: int) \
            -> Tuple[numpy.ndarray, int]:
        """Returns an item from the dataset.

        :param item: Index of the item.
        :type item: int
        :return: Features and class of the item.
        :rtype: (numpy.ndarray, int)
        """
        if self.load_into_memory:
            the_item: Dict[str, Union[int, numpy.ndarray]] = self.files[item]
        else:
            the_item = self._load_file(self.files[item])
        return the_item[self.key_features], the_item[self.key_class][0,:] if self.dataset_type=="Weak" else the_item[self.key_class]

# EOF
