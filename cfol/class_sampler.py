from typing import Iterable
import numpy as np
import torch
from torch.utils.data.sampler import Sampler
from cfol.data_module import InMemoryDataset
from torch.functional import F


class ClassSampler(Sampler[int]):
    r"""
    Args:
        dataset (Dataset): dataset to sample from
    """
    dataset: InMemoryDataset

    def __init__(self, 
        dataset: InMemoryDataset, 
        gamma=1/2, 
        base_dist="uniform", 
        prior="uniform",
        reweight=False,
    ) -> None:

        assert isinstance(dataset, InMemoryDataset), "ClassSampler requires a dataset of class `InMemoryDataset`"
        self.gamma = gamma
        self.dataset = dataset
        self.reweight = reweight
        
        uniform = 1/(self.dataset.num_classes) * torch.ones(dataset.num_classes)
        self._class_fractions = torch.tensor([len(indices) for indices in dataset.class_indices]) / len(dataset)

        if base_dist == 'uniform':
            self.base_dist = uniform
        elif base_dist == 'empirical':
            self.base_dist = self._class_fractions
        else:
            raise ValueError("Invalid ClassSampler.base_dist")

        # Initialize prior
        self.prior = prior
        self.reset_prior()

    def reset_prior(self):
        if self.prior == 'uniform':
            self.w = torch.zeros(self.dataset.num_classes)
        elif self.prior == 'empirical':
            self.w = torch.log(self._class_fractions)
        elif isinstance(self.prior, Iterable):
            self.w = torch.log(torch.tensor(self.prior))
        elif self.prior is None:        
            self.w = torch.log(self.base_dist)
        else:
            raise ValueError("Invalid ClassSampler.prior")

    @property
    def q(self):
        return F.softmax(self.w, dim=0)

    @property
    def p(self):
        # Only sample nonuniformly if self.reweight=False
        return self.distribution(use_base=self.reweight)

    def batch_weight(self, class_ids):
        # Only weight nonuniformly if self.reweight=True
        p = self.distribution(use_base=not self.reweight)
        return p[class_ids]

    def distribution(self, use_base):
        if use_base:
            return self.base_dist
        else:
            return self.gamma * self.base_dist + (1-self.gamma) * self.q

    def batch_update(self, class_ids, eta_times_loss_arms):
        """Parallel update (does not update self.p sequentially)
        """
        loss_vec = torch.zeros(self.dataset.num_classes)
        p = self.p
        for i, class_id in enumerate(class_ids):
            eta_times_loss_arm = eta_times_loss_arms[i]
            loss_vec[class_id] = loss_vec[class_id] + eta_times_loss_arm / p[class_id]
        self.w = self.w + loss_vec

    def update(self, class_id, eta_times_loss_arm):
        loss_vec = torch.zeros(self.dataset.num_classes)
        loss_vec[class_id] = eta_times_loss_arm / self.p[class_id]
        self.w = self.w + loss_vec

    def sample_class_id(self):
        class_id = torch.multinomial(self.p, num_samples=1).item()
        return class_id

    def __iter__(self):
        for _ in range(len(self)):
            class_id = self.sample_class_id()
            class_indices = self.dataset.class_indices[class_id]
            idx = torch.randint(high=len(class_indices), size=(1,), 
                                dtype=torch.int64).item()
            yield class_indices[idx]

    def __len__(self):
        return len(self.dataset)
