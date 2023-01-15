from typing import Any, Callable, Optional, Sequence, Union

from pl_bolts.datamodules.vision_datamodule import VisionDataModule
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization
from pl_bolts.utils import _TORCHVISION_AVAILABLE
from pl_bolts.utils.warnings import warn_missing_pkg

if _TORCHVISION_AVAILABLE:
    from torchvision import transforms as transform_lib
else:  # pragma: no cover
    warn_missing_pkg('torchvision')

from fae.GTSRB_dataset import GTSRB

class GTSRBDataModule(VisionDataModule):
    name = "GTSRB"
    dataset_cls = GTSRB
    dims = (3, 32, 32) # after transforms in runner.py

    def __init__(
        self,
        data_dir: Optional[str] = None,
        val_split: Union[int, float] = 0.2,
        num_workers: int = 16,
        normalize: bool = False,
        batch_size: int = 32,
        seed: int = 42,
        shuffle: bool = False,
        pin_memory: bool = False,
        drop_last: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            data_dir: Where to save/load the data
            val_split: Percent (float) or number (int) of samples to use for the validation split
            num_workers: How many workers to use for loading data
            normalize: If true applies image normalize
            batch_size: How many samples per batch to load
            seed: Random seed to be used for train/val/test splits
            shuffle: If true shuffles the train data every epoch
            pin_memory: If true, the data loader will copy Tensors into CUDA pinned memory before
                        returning them
            drop_last: If true drops the last incomplete batch
        """
        # Make th dataset_cls interace the same as for CIFAR10
        dataset_cls = self.dataset_cls

        def dataset_cls_wrapper(*args, train=True, **kwargs):
            split = "train" if train else "test"
            return dataset_cls(*args, split=split, **kwargs)

        self.dataset_cls = dataset_cls_wrapper

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
            *args,
            **kwargs,
        )

    @property
    def num_samples(self) -> int:
        train_len, _ = self._get_splits(len_dataset=39_209)
        return train_len

    @property
    def num_classes(self) -> int:
        return 43

    def default_transforms(self) -> Callable:
        if self.normalize:
            raise ValueError("Normalization not supported for GTSRB")
        else:
            transforms = transform_lib.Compose([transform_lib.ToTensor()])

        return transforms
