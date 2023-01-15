from typing import Any, Callable, Optional, Sequence, Union

from pl_bolts.datamodules.vision_datamodule import VisionDataModule
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization
from pl_bolts.utils import _TORCHVISION_AVAILABLE
from pl_bolts.utils.warnings import warn_missing_pkg

if _TORCHVISION_AVAILABLE:
    from torchvision import transforms as transform_lib
    from cfol.tiny_imagenet_dataset import TinyImageNet
else:  # pragma: no cover
    warn_missing_pkg('torchvision')
    TinyImageNet = None


class TinyImageNetDataModule(VisionDataModule):
    """
    Specs:
        - 200 classes (1 per class)
        - Each image is (3 x 64 x 64)
    Standard TinyImageNet, train, val, test splits and transforms
    Transforms::
        mnist_transforms = transform_lib.Compose([
            transform_lib.ToTensor(),
            transforms.Normalize(
                mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                std=[x / 255.0 for x in [63.0, 62.1, 66.7]]
            )
        ])
    Example::
        dm = TinyImageNetDataModule(PATH)
        model = LitModel()
        Trainer().fit(model, datamodule=dm)
    Or you can set your own transforms
    Example::
        dm.train_transforms = ...
        dm.test_transforms = ...
        dm.val_transforms  = ...
    """
    name = "tinyimagenet"
    dataset_cls = TinyImageNet
    dims = (3, 64, 64)

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
        train_len, _ = self._get_splits(len_dataset=100_000)
        return train_len

    @property
    def num_classes(self) -> int:
        """
        Return:
            200
        """
        return 200

    def default_transforms(self) -> Callable:
        if self.normalize:
            #transforms = transform_lib.Compose([transform_lib.ToTensor(), cifar10_normalization()])
            raise ValueError("Normalization not supported for TinyImageNet")
        else:
            transforms = transform_lib.Compose([transform_lib.ToTensor()])

        return transforms
