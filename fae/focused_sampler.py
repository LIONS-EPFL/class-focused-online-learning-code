from fae.GTSRB_datamodule import GTSRBDataModule
from fae.imagenette_datamodule import ImagenetteDataModule
from fae.tiny_imagenet_datamodule import TinyImageNetDataModule
from fae.labeled_stl10_datamodule import STL10DataModule
from fae.cifar100_datamodule import CIFAR100DataModule
from torchvision.datasets.cifar import CIFAR10
import math
import os
from abc import abstractmethod
from threading import current_thread
from typing import Any, Callable, List, Optional, Sized, Tuple, Union

from PIL import Image
import numpy as np
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split
from pl_bolts.datamodules import CIFAR10DataModule
from torch.utils.data.sampler import Sampler

from fae.index_wrapper import IndexWrapper
from fae.pretty_print_binary_tree import printBTree


GAMMA = 0.5


class Node(object):
    def __init__(self, val, l=None, r=None, p=None, root=None):
        self.l: Optional[Node] = l
        self.r: Optional[Node] = r
        self.v: torch.Tensor = val
        self.p: Optional[Node] = p
        if root is None:
            root = self
        self.root: Node = root

    def sum_children(self):
        if self.l is not None:
            self.l.sum_children()
            self.v += self.l.v
        if self.r is not None:
            self.r.sum_children()
            self.v += self.r.v
    

    @property
    def q(self) -> torch.Tensor:
        return self.v / self.root.v

    def prob(self, gamma, size) -> torch.Tensor:
        return gamma / size + (1-gamma)*self.q

    def is_leaf(self):
        return self.l is None


class Tree(object):
    def __init__(self, size):
        # init generator
        self.generator = torch.Generator()
        self.generator.manual_seed(int(torch.empty((), dtype=torch.int64).random_().item()))

        # build binary tree of high log2(size) and init m nodes to 1 rest to zero
        root = Node(0)
        prev_layer_nodes = [root]

        leaf_layer = math.ceil(math.log2(size)) + 1
        for layer_count in range(1, leaf_layer):
            curr_layer_nodes = []
            i = 0
            for node in prev_layer_nodes:
                i += 1
                val = 1.0 if layer_count + 1 == leaf_layer and i <= size else 0.0
                val = torch.tensor(val, dtype=torch.double)
                node.l = Node(val, p=node, root=root)
                curr_layer_nodes.append(node.l)
                
                i += 1
                val = 1.0 if layer_count + 1 == leaf_layer and i <= size else 0.0
                val = torch.tensor(val, dtype=torch.double)
                node.r = Node(val, p=node, root=root)
                curr_layer_nodes.append(node.r)

            prev_layer_nodes = curr_layer_nodes
        
        # Make each node.val sum of it's children
        root.sum_children()

        self.leaf_nodes = prev_layer_nodes[:size]
        self.leaf_nodes_indexes = {n:idx for idx,n in enumerate(self.leaf_nodes)}
        self.root = root
        self._size = size

    def update(self, i, f):
        node = self.leaf_nodes[i]
        delta = f * node.v - node.v
        p = node
        while p is not None:
            pv_new = p.v + delta 
            # if pv_new.isinf() or pv_new.isnan():
            #     cntx = {'p.v': p.v, 'pv_new': pv_new, 'delta': delta, 'f': f, 'node.v': node.v, 'i': i}
            #     raise RuntimeError(f"FocusedSampler.update failed with context: {cntx}")
            p.v = pv_new
            p = p.p

    def sample(self):
        n = self.root
        while not n.is_leaf():
            # TODO: ensure that normalization of the prob is correct
            prob = n.l.q / n.q
            #prob = torch.clamp(prob, 0.0, 1.0)
            
            # try:
            turn_l = torch.bernoulli(prob)
            # except RuntimeError as e:
            #     cntx = {
            #         "prob": prob,
            #         "n.root.v": n.root.v,
            #         "n.v": n.v,
            #         "n.l.v": n.l.v,
            #     }
            #     raise RuntimeError(f"FocusedSampler had an invalid probability: {cntx}") from e
            
            if turn_l:
                n = n.l
            else:
                n = n.r

        return self.leaf_nodes_indexes[n]

    def __getitem__(self, idx) -> Node:
        return self.leaf_nodes[idx]

    def __len__(self):
        return self._size


class FocusedSampler(Sampler[int]):
    r"""
    Args:
        data_source (Dataset): dataset to sample from
    """
    data_source: Sized

    def __init__(self, data_source: Sized, gamma=0.5) -> None:
        self.data_source = data_source
        self.gamma = gamma
        self.tree = Tree(len(self.data_source))

    def update(self, idx, nominator):
        node = self.tree[idx]
        p = node.prob(gamma=self.gamma, size=len(self.tree))
        new_p = torch.exp(nominator/p)
        self.tree.update(idx, new_p)

    def __iter__(self):
        N = len(self)
        for _ in range(N):
            if torch.bernoulli(torch.tensor(self.gamma)):
                yield torch.randint(high=N, size=(1,), dtype=torch.int64).item()
            else:
                yield self.tree.sample()

    def __len__(self):
        return len(self.data_source)


class FocusedDataModuleMixin(object):
    """
    A variant of pt-lightning's CIFAR10DataModule which can pass custom arguments
    to the train data loader and uses IndexedCIFAR10 for acessing the batch indexes.

    This allows us to add a custom sampler for the training dataset.

    Should be used to extend a child of `VisionDataModule`.
    """
    def __init__(self,
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
        get_sampler: callable = None,
        **kwargs: Any,
    ) -> None:
        self.get_sampler = get_sampler
        self.sampler = None
        
        super().__init__(
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
            **kwargs)

    def train_dataloader(self, *args: Any, **kwargs: Any) -> DataLoader:
        """ The train dataloader """
        return DataLoader(
                    IndexWrapper(self.dataset_train),
                    batch_size=self.batch_size,
                    shuffle=self.shuffle,
                    num_workers=self.num_workers,
                    drop_last=self.drop_last,
                    pin_memory=self.pin_memory,
                    sampler=self.sampler,
                )

    def val_dataloader(self, *args: Any, **kwargs: Any) -> Union[DataLoader, List[DataLoader]]:
        """ The val dataloader """
        return self._data_loader(IndexWrapper(self.dataset_val))

    def test_dataloader(self, *args: Any, **kwargs: Any) -> Union[DataLoader, List[DataLoader]]:
        """ The test dataloader """
        return self._data_loader(IndexWrapper(self.dataset_test))

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Creates train, val, and test dataset.

        Allows for using train dataset for validation (thus full training dataset for training) by setting val_split=0.
        """
        if stage == "fit" or stage is None:
            train_transforms = self.default_transforms() if self.train_transforms is None else self.train_transforms
            val_transforms = self.default_transforms() if self.val_transforms is None else self.val_transforms

            dataset_train = self.dataset_cls(self.data_dir, train=True,
                                             **self.EXTRA_ARGS)
            # Train split
            self.dataset_train = self._split_dataset(dataset_train, transform=train_transforms)

            if self.val_split == 0:
                self.dataset_val = self.dataset_cls(
                    self.data_dir, train=False, transform=val_transforms, **self.EXTRA_ARGS)
            else:
                dataset_val = self.dataset_cls(self.data_dir, train=True, **self.EXTRA_ARGS)
                self.dataset_val = self._split_dataset(dataset_val, train=False, transform=val_transforms)

        if stage == "test" or stage is None:
            test_transforms = self.default_transforms() if self.test_transforms is None else self.test_transforms
            self.dataset_test = self.dataset_cls(
                self.data_dir, train=False, transform=test_transforms, **self.EXTRA_ARGS
            )

        # initialize the sampler
        if stage == "fit" or stage is None:
            if self.get_sampler is not None:
                self.sampler = self.get_sampler(self.dataset_train)

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
        data = [None] * n
        targets = np.zeros(n ,dtype=np.int)

        for i in range(n):
            img, target = dataset[i]
            data[i] = np.array(img)
            targets[i] = target

        return InMemoryDataset(data, targets, transform=transform)


class InMemoryDataset(Dataset):
    def __init__(self, data, targets, transform=None) -> None:
        super().__init__()
        self.data = data
        self.targets = targets
        self.transform = transform
        self.num_classes = int(np.max(self.targets)) + 1
        self.class_indices = [(self.targets == class_id).nonzero()[0] for class_id in range(self.num_classes)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index) -> Tuple[Any, Any]:
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, target


class FocusedCIFAR10DataModule(FocusedDataModuleMixin,CIFAR10DataModule):
    pass

class FocusedSTL10DataModule(FocusedDataModuleMixin,STL10DataModule):
    pass

class FocusedCIFAR100DataModule(FocusedDataModuleMixin,CIFAR100DataModule):
    pass

class FocusedTinyImageNetDataModule(FocusedDataModuleMixin,TinyImageNetDataModule):
    pass

class FocusedGTSRBDataModule(FocusedDataModuleMixin,GTSRBDataModule):
    pass

class FocusedImagenetteDataModule(FocusedDataModuleMixin,ImagenetteDataModule):
    pass
