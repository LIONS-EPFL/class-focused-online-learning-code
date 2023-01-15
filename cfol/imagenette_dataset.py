import os
import csv
import pathlib
from typing import Any, Callable, Optional, Tuple

import PIL
from matplotlib.path import Path
from pathlib import Path

from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from torchvision.datasets.utils import download_and_extract_archive, verify_str_arg


DATADIR = Path('data/')  

imagenette_urls = {'imagenette2-160': 'https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz',#'https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz',
                   'imagewoof2-160': 'https://s3.amazonaws.com/fast-ai-imageclas/imagewoof2-160.tgz,'#'https://s3.amazonaws.com/fast-ai-imageclas/imagewoof2.tgz'
                   }

imagenette_len = {'imagenette2-160': {'train': 9469, 'val': 3925},
                  'imagewoof2-160': {'train': 9025, 'val': 3929}
                  }

imagenette_md5 = {'imagenette2-160': 'e793b78cc4c9e9a4ccc0c1155377a412', #'43b0d8047b7501984c47ae3c08110b62',
                  'imagewoof2-160': '3d200a7be99704a0d7509be2a9fbfe15' #'5eaf5bbf4bf16a77c616dc6e8dd5f8e9'
                  }


def check_data_exists(root, name) -> bool:
    ''' Verify data at root and return True if len of images is Ok.
    '''
    num_classes = 10
    if not root.exists():
        return False

    for split in ['train', 'val']:
        split_path = Path(root, split)
        if not root.exists():
            return False

        classes_dirs = [dir_entry for dir_entry in os.scandir(split_path)
                        if dir_entry.is_dir()]
        if num_classes != len(classes_dirs):
            return False

        num_samples = 0
        for dir_entry in classes_dirs:
            num_samples += len([fn for fn in os.scandir(dir_entry)
                                if fn.is_file()])

        if num_samples != imagenette_len[name][split]:
            return False

    return True


class ImagenetteDataset(ImageFolder):
    """
    Args:
        root (string): Root directory of the dataset.
        split (string, optional): The dataset split, supports ``"train"`` (default), or ``"test"``.
        transform (callable, optional): A function/transform that  takes in an PIL image and returns a transformed
            version. E.g, ``transforms.RandomCrop``.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.
        download (bool, optional): If True, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:

        # if woof:
        #     self.name = 'imagewoof2-160'
        # else:
        self.name = 'imagenette2-160'
        self.data_dir = Path(root)
        self.root = Path(self.data_dir, self.name)
        self._split = verify_str_arg(split, "split", ("train", "test"))
        if download:
            self.download()

        if not check_data_exists(self.root, self.name):
            raise RuntimeError("Dataset not found. You can use download=True to download it")

        if self._split == "train":
            super().__init__(root=Path(self.root, 'train'), transform=transform, target_transform=target_transform)
        else:
            super().__init__(root=Path(self.root, 'val'), transform=transform, target_transform=target_transform)

    def download(self) -> None:
        if not check_data_exists(self.root, self.name):
            dataset_url = imagenette_urls[self.name]
            download_and_extract_archive(url=dataset_url, download_root=self.data_dir, md5=imagenette_md5[self.name])
