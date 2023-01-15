from typing import Tuple

import torch as pt
from torch.utils.data import Dataset


class IndexWrapper(Dataset):
    """
    Simple wrapper for all image=>label style datasets which adds a third tuple element
    with the index in the dataset.
    """

    def __init__(self, dataset) -> None:
        super().__init__()
        self.dataset = dataset

    def __getitem__(self, index: int) -> Tuple[pt.Tensor, pt.Tensor, pt.Tensor]:
        X, Y = self.dataset[index]
        return X, Y, pt.tensor(index)

    def __len__(self) -> int:
        return len(self.dataset)
