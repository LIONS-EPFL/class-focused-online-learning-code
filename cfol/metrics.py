

import torch
from pytorch_lightning.metrics import Metric


class Histogram(Metric):
    def __init__(self, num_bins, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.num_bins = num_bins
        self.add_state("bins", default=torch.zeros(num_bins), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, target: torch.Tensor):
        self.bins += torch.bincount(target, minlength=self.num_bins)
        self.total += len(target)

    def compute(self):
        return self.bins.float() / self.total
