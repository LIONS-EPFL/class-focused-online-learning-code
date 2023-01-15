from PIL import Image
import numpy as np
import torch

from cfol.data_transforms import imbalance_factor_to_lt_factor, reduce_classes_dbset, reduce_classes_dbset_longtailed
from typing import Any, Optional, Tuple, Union
from torch.utils.data import Dataset
from cfol.focused_sampler import FocusedCIFAR10DataModule, FocusedGTSRBDataModule, FocusedImagenetteDataModule, FocusedSTL10DataModule, FocusedCIFAR100DataModule, FocusedTinyImageNetDataModule, InMemoryDataset


class ReducedDataModuleMixin(object):
    """
    Requires inheriting `FocusedDataModuleMixin`.
    """

    def __init__(
        self,
        data_dir: Optional[str] = None,
        val_split: Union[int, float] = 0.2,
        num_workers: int = 16,
        normalize: bool = False,
        batch_size: int = 32,
        seed: int = 42,
        reduce_seed: int = 43,
        shuffle: bool = False,
        pin_memory: bool = False,
        drop_last: bool = False,
        n_reduce: int = None,
        imbalance_factor: float = None,
        reduce_val: bool = False,
        get_sampler = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:

        self.reduce_seed = reduce_seed
        self.n_reduce = n_reduce
        self.imbalance_factor = imbalance_factor
        self.reduce_val = reduce_val
        super().__init__(  # type: ignore[misc]
            data_dir=data_dir,
            val_split=val_split,
            num_workers=num_workers,
            normalize=normalize,
            batch_size=batch_size,
            seed=seed,
            shuffle=shuffle,
            pin_memory=pin_memory,
            drop_last=drop_last,
            get_sampler=get_sampler,
            *args,
            **kwargs,
        )

    def _split_dataset(self, dataset: Dataset, train: bool = True, transform=None) -> Dataset:
        dataset = super()._split_dataset(dataset, train=train)

        # Create the training subset after the train/val split so there is no overlap
        dataset = self.create_subset(dataset, transform=transform)

        return dataset 

    def create_subset(self, dataset, transform=None):
        # Hackily recreate data matrix by looping to be compatible with `reduce_classes_dbset`
        # TODO: this is not going to work for bigger datasets
        # (Looping over indices is necessary since `dataset` might be a subset)
        n = len(dataset)
        img = dataset[0][0]
        sample = np.array(img)
        data_shape = sample.shape
        data = np.zeros((n,) + data_shape, dtype="uint8")
        targets = np.zeros(n)

        for i in range(n):
            img, target = dataset[i]
            data[i] = np.array(img)
            targets[i] = target

        with TemporaryTorchSeed(self.reduce_seed):
            if self.imbalance_factor is not None:
                lt_factor = imbalance_factor_to_lt_factor(self.imbalance_factor, data, targets)
                data, targets = reduce_classes_dbset_longtailed(
                    data, targets, 
                    lt_factor=lt_factor,
                    permute=True)
            else:
                data, targets = reduce_classes_dbset(
                    data, targets, 
                    n_reduce=self.n_reduce, 
                    permute=True)

            return InMemoryDataset(data, targets, transform=transform)


class TemporaryTorchSeed(object):
    """Context manager  for temporarily setting the  torch seed.

    Example:
        >>> with TemporaryTorchSeed(42):
    """
    def __init__(self, seed):
        self.prev_state = None
        self.seed = seed
      
    def __enter__(self):
        self.prev_state = torch.random.get_rng_state()
        torch.manual_seed(self.seed)
  
    def __exit__(self, exception_type, exception_value, traceback):
        torch.random.set_rng_state(self.prev_state)


class ReducedCIFAR10DataModule(ReducedDataModuleMixin,FocusedCIFAR10DataModule):
    pass

class ReducedSTL10DataModule(ReducedDataModuleMixin,FocusedSTL10DataModule):
    pass

class ReducedCIFAR100DataModule(ReducedDataModuleMixin,FocusedCIFAR100DataModule):
    pass

class ReducedTinyImageNetDataModule(ReducedDataModuleMixin,FocusedTinyImageNetDataModule):
    pass

class ReducedGTSRBDataModule(ReducedDataModuleMixin,FocusedGTSRBDataModule):
    pass

class ReducedImagenetteDataModule(ReducedDataModuleMixin,FocusedImagenetteDataModule):
    pass
