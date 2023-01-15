import math
from fae.augmentation import Cutout
from fae.imagenette_dataset import ImagenetteDataset
from fae.tiny_imagenet_dataset import TinyImageNet
from math import e

import os
from pathlib import Path

import torch
import numpy as np
from pytorch_lightning.profiler.profilers import SimpleProfiler
from torch.utils.data.dataset import Subset
from torchvision.datasets.stl10 import STL10
from torchvision.datasets.cifar import CIFAR10, CIFAR100
from torchvision import transforms
from pytorch_lightning.utilities.cloud_io import load as pl_load
from pytorch_lightning.core.saving import save_hparams_to_yaml
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
#from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers import WandbLogger
from pl_bolts.datamodules import CIFAR10DataModule
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.profiler import PyTorchProfiler

from fae.class_sampler import ClassSampler
from fae.model import Model
from fae.config import Hpars
from fae.GTSRB_dataset import GTSRB
from fae.data_module import ReducedCIFAR10DataModule, ReducedCIFAR100DataModule, ReducedGTSRBDataModule, ReducedImagenetteDataModule, ReducedSTL10DataModule, ReducedTinyImageNetDataModule
from fae.focused_sampler import FocusedCIFAR10DataModule, FocusedCIFAR100DataModule, FocusedGTSRBDataModule, FocusedImagenetteDataModule, FocusedSTL10DataModule, FocusedSampler, FocusedTinyImageNetDataModule

from sacred import Experiment, SETTINGS
SETTINGS['HOST_INFO']['INCLUDE_GPU_INFO'] = False

ex = Experiment('fae')


def get_stl10(*args, train=True, **kwargs):
    """Wrapper because STL10 unfortunately have a different interface than CIFAR10 etc.
    """
    assert train, "Only support train"
    return STL10(*args, split="train", **kwargs)


def get_GTSRB(*args, train=True, **kwargs):
    """Wrapper because GTSRB unfortunately have a different interface than CIFAR10 etc.
    """
    assert train, "Only support train"
    return GTSRB(*args, split="train", **kwargs)


def get_imagenette(*args, train=True, **kwargs):
    """Wrapper because GTSRB unfortunately have a different interface than CIFAR10 etc.
    """
    assert train, "Only support train"
    return ImagenetteDataset(*args, split="train", **kwargs)


base_dataset = {
    'cifar10': CIFAR10,
    'cifar100': CIFAR100,
    'stl10': get_stl10,
    'tinyimagenet': TinyImageNet,
    'GTSRB': get_GTSRB,
    'imagenette': get_imagenette
}

reduced_data_modules = {
    'cifar10': ReducedCIFAR10DataModule,
    'cifar100': ReducedCIFAR100DataModule,
    'stl10': ReducedSTL10DataModule,
    'tinyimagenet': ReducedTinyImageNetDataModule,
    'imagenette': ReducedImagenetteDataModule,
    'GTSRB': ReducedGTSRBDataModule,
}

focused_data_modules = {
    'cifar10': FocusedCIFAR10DataModule,
    'cifar100': FocusedCIFAR100DataModule,
    'stl10': FocusedSTL10DataModule,
    'imagenette': FocusedImagenetteDataModule,
    'GTSRB': FocusedGTSRBDataModule,
}


def get_train_transforms(hpars):
    train_augmentation = []
    
    # Data
    if hpars.dataset in ["cifar10", "cifar100"]:
        image_size = 32
        train_augmentation += [ # base_data is already in tensor form, only need to augment
                transforms.RandomCrop(image_size, padding=4),
                transforms.RandomHorizontalFlip(),
            ]
    elif hpars.dataset in ["GTSRB"]:
        image_size = 32
        train_augmentation += [ # base_data is already in tensor form, only need to augment
                transforms.RandomCrop(image_size, padding=4),
                transforms.RandomHorizontalFlip(),
            ]
    elif hpars.dataset == "stl10":
        image_size = 96
        train_augmentation += [ # base_data is already in tensor form, only need to augment
                transforms.RandomCrop(image_size, padding=8),
                transforms.RandomHorizontalFlip(),
            ]
    elif hpars.dataset == "tinyimagenet":
        image_size = 64
        train_augmentation += [ # base_data is already in tensor form, only need to augment
                transforms.RandomCrop(image_size, padding=4),
                transforms.RandomHorizontalFlip(),
            ]
    elif hpars.dataset == "imagenette":
        image_size = 160
        train_augmentation += [ # base_data is already in tensor form, only need to augment
                transforms.RandomResizedCrop(image_size, scale=(0.35, 1)),
                transforms.RandomHorizontalFlip(),
            ]
    else:
        image_size = 32
    
    train_augmentation += [transforms.ToTensor()]
    
    if hpars.augment_type == 'v1':
        train_augmentation += [
            transforms.ColorJitter(0.25, 0.25, 0.25),
            transforms.RandomErasing(),
            transforms.RandomRotation(2),
        ]
    elif hpars.augment_type == 'v2':
        train_augmentation += [
            transforms.ColorJitter(.25,.25,.25),
            transforms.RandomRotation(2),
        ]
    
    if hpars.cutout:
        cutout_size = math.floor(hpars.cutout_scale * image_size)
        train_augmentation += [
            Cutout(n_holes=1, length=cutout_size)
        ]

    return transforms.Compose(train_augmentation)

def get_val_transforms(hpars):
    train_augmentation = []
    if hpars.dataset == "imagenette":
        extra_size = 32
        image_size = 160
        train_augmentation += [ # base_data is already in tensor form, only need to augment
            transforms.Resize(image_size + extra_size),
            transforms.CenterCrop(image_size),
        ]
    train_augmentation += [transforms.ToTensor()]
    return transforms.Compose(train_augmentation)


@ex.config
def cfg():
    h = {} #Hpars.to_dict(Hpars())


@ex.automain
def run(h):
    # Configs
    hpars: Hpars = Hpars.to_cls(h)
    
    # Cannot happen inside __attrs_post_init__ unfortunately
    # since ex.config initializes the default values
    if hpars.max_epochs is None:
        hpars.max_epochs = hpars.epochs

    # Logging
    #logger = TensorBoardLogger(hpars.exp_dir, name=hpars.exp_name)
    wandb_logger = WandbLogger(project=hpars.project, name=hpars.exp_name, id=hpars.wandb_resume_id)

    # Save checkpoint using wandb id so we can restart while logging to the same experiment
    ckpt_dir = os.path.join(hpars.out_dir, wandb_logger.experiment.id)
    print("Saving checkpoints to", ckpt_dir)
    if hpars.project == 'focused-adversarial-examples' and hpars.loss_type != "clean":
        checkpoint_callbacks = [ModelCheckpoint(
            dirpath=ckpt_dir,
            filename='ckpt_' + monitor + '_{epoch:02d}',
            save_top_k=1,
            monitor=monitor,
            mode="max",
            period=hpars.val_rate,
        ) for monitor in ['val_adv_avg_class_precision', 'val_adv_min_class_precision']]
        ckpt_cb_avg, ckpt_cb_min = checkpoint_callbacks
        
        # early_stopping_callback = EarlyStopping(
        #     monitor='val_adv_acc',
        #     min_delta=0.00,
        #     patience=5,
        #     verbose=False,
        #     mode='max'
        #     )
    else: 
        checkpoint_callbacks = [ModelCheckpoint(
            dirpath=ckpt_dir,
            filename='checkpoint_{epoch:02d}',
            save_top_k=1,
            monitor='val_clean_avg_class_precision',
            mode="max",
            period=hpars.val_rate,
        )]
        ckpt_cb_avg, = checkpoint_callbacks
        ckpt_cb_min = None

        # early_stopping_callback = EarlyStopping(
        #     monitor='val_clean_acc',
        #     min_delta=0.00,
        #     patience=5,
        #     verbose=False,
        #     mode='max'
        #     )
    # Saves the last checkpoint
    ckpt_cb_last = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename='checkpoint_{epoch:02d}',
        save_top_k=None, 
        monitor=None,
    )
    checkpoint_callbacks.append(ckpt_cb_last)
    lr_monitor = LearningRateMonitor()

    # Store git info
    # Path(logger.log_dir).mkdir(parents=True, exist_ok=True)
    # save_hparams_to_yaml(os.path.join(logger.log_dir, "github_info.yaml"), _run.experiment_info)

    # Focused sampler
    if hpars.use_focused_sampler:
        # TODO: ensure that the sampler is not reset
        datamodule_kwargs = dict(
            shuffle=False,
            get_sampler=lambda dataset: FocusedSampler(dataset)
        )
    elif hpars.use_class_sampler:
        datamodule_kwargs = dict(
            shuffle=False,
            get_sampler=lambda dataset: ClassSampler(
                    dataset, 
                    gamma=hpars.class_sampler_gamma, 
                    base_dist=hpars.class_sampler_base_dist,
                    prior=hpars.class_sampler_prior,
                    reweight=hpars.class_sampler_reweight)
        )
    else:
        datamodule_kwargs = dict(
            shuffle=True,
        )

    # Dataloader
    if hpars.dataset_n_reduce is not None or hpars.dataset_imbalance_factor is not None:
        print("Using ReducedDataModule")
        ReducedDataModule = reduced_data_modules[hpars.dataset]
        dm = ReducedDataModule(
            data_dir=f"data/{hpars.dataset}",
            val_split=hpars.val_split,
            num_workers=hpars.num_workers,
            batch_size=hpars.batch_size,
            normalize=False, # Happens in the model AFTER attack
            train_transforms=get_train_transforms(hpars),
            val_transforms=get_val_transforms(hpars),
            test_transforms=get_val_transforms(hpars),
            n_reduce=hpars.dataset_n_reduce,
            imbalance_factor=hpars.dataset_imbalance_factor,
            reduce_val=hpars.reduce_val,
            **datamodule_kwargs,
        )
    else:    
        print("Using FocusedDataModule")
        FocusedDataModule = focused_data_modules[hpars.dataset]
        dm = FocusedDataModule(
            data_dir=f"data/{hpars.dataset}",
            val_split=hpars.val_split,
            num_workers=hpars.num_workers,
            batch_size=hpars.batch_size,
            normalize=False, # Happens in the model AFTER attack,
            train_transforms=get_train_transforms(hpars),
            val_transforms=get_val_transforms(hpars),
            test_transforms=get_val_transforms(hpars),
            **datamodule_kwargs,
        )

    dm.prepare_data()
    dm.setup()
    num_training_batches = len(dm.train_dataloader())
    print("Max class id in test dataset:", dm.dataset_train.num_classes)
    print("Length of training dataset:", len(dm.dataset_train))
    print("Length of validation dataset:", len(dm.dataset_val))
    print("Length of test dataset:", len(dm.dataset_test))

    # import matplotlib.pyplot as plt
    # plt.imshow(dm.dataset_train[0][0].transpose(0,-1))
    # plt.savefig('test1.png')
    # plt.imshow(dm.dataset_train[0][0].transpose(0,-1))
    # plt.savefig('test2.png')

    # Profiler
    if hpars.profiler == "simple":
        profiler = SimpleProfiler()
    elif hpars.profiler == "pytorch-mem":
        profiler = PyTorchProfiler(
            profile_memory=True,
            sort_by_key="cuda_memory_usage",
        )
    else:
        profiler = None

    # Get class labels and targets
    get_dataset = base_dataset[hpars.dataset]
    class_names = get_dataset(f"data/{hpars.dataset}", train=True, download=False).classes
    targets = [b[1] for b in dm.dataset_train]

    # Define models
    model = Model(hpars, 
        sampler=dm.sampler, 
        profiler=profiler,
        classes=class_names,
        train_targets=targets,
        num_training_batches=num_training_batches)

    # Pytorch lightning Trainer
    trainer = Trainer(
        resume_from_checkpoint=hpars.ckpt_path,
        logger=wandb_logger,
        profiler=profiler,
        gpus=hpars.gpus, 
        max_epochs=hpars.max_epochs, 
        progress_bar_refresh_rate=hpars.progress_bar_refresh_rate, 
        check_val_every_n_epoch=hpars.val_rate,
        callbacks=checkpoint_callbacks + [lr_monitor],
        limit_train_batches=hpars.limit_train_batches,
        limit_val_batches=hpars.limit_val_batches,
        limit_test_batches=hpars.limit_test_batches,
        fast_dev_run=hpars.fast_dev_run,
        precision=hpars.precision,
        amp_level=hpars.amp_level,
    )

    if hpars.test_best_avg_ckpt_path is not None or hpars.test_best_min_ckpt_path is not None:
        # Test a checkpoints without training

        if hpars.test_best_avg_ckpt_path is not None:
            ckpt = pl_load(hpars.test_best_avg_ckpt_path, map_location=lambda storage, loc: storage)
            if hpars.ckpt_has_state_dict:
                ckpt = ckpt['state_dict']
            if hpars.ckpt_remove_prefix is not None:
                ckpt = {k.replace(hpars.ckpt_remove_prefix, ''):v for k,v in ckpt.items() if k.startswith(hpars.ckpt_remove_prefix)}

            if hpars.ckpt_is_pt_lightning:
                model.load_state_dict(ckpt)
            else:
                model.model[1].load_state_dict(ckpt)

            model.test_prefix = "best_avg"
            trainer.test(model, datamodule=dm)
            wandb_logger.experiment.summary["best_avg_acc_ckpt"] = hpars.test_best_avg_ckpt_path
        
        if hpars.test_best_min_ckpt_path is not None:
            ckpt = pl_load(hpars.test_best_min_ckpt_path, map_location=lambda storage, loc: storage)
            if hpars.ckpt_has_state_dict:
                ckpt = ckpt['state_dict']
            if hpars.ckpt_remove_prefix is not None:
                ckpt = {k.replace(hpars.ckpt_remove_prefix, ''):v for k,v in ckpt.items() if k.startswith(hpars.ckpt_remove_prefix)}

            if hpars.ckpt_is_pt_lightning:
                model.load_state_dict(ckpt)
            else:
                model.model[1].load_state_dict(ckpt)

            model.test_prefix = "best_min"
            trainer.test(model, datamodule=dm)
            wandb_logger.experiment.summary["best_min_acc_ckpt"] = hpars.test_best_min_ckpt_path
    else:
        # Train the model
        trainer.fit(model, dm)
        
        # Test best checkpoints *after* training
        if hpars.test_best_avg:
            model.test_prefix = "best_avg"
            print("Testing with checkpoint:", ckpt_cb_avg.best_model_path)
            trainer.test(datamodule=dm, ckpt_path=ckpt_cb_avg.best_model_path)
            wandb_logger.experiment.summary["best_avg_acc_ckpt"] = ckpt_cb_avg.best_model_path
        
        if hpars.test_best_min and ckpt_cb_min is not None:
            model.test_prefix = "best_min"
            trainer.test(datamodule=dm, ckpt_path=ckpt_cb_min.best_model_path)
            wandb_logger.experiment.summary["best_min_acc_ckpt"] = ckpt_cb_min.best_model_path

        if hpars.test_last:
            model.test_prefix = "last"
            trainer.test(datamodule=dm, ckpt_path=ckpt_cb_last.last_model_path)
            wandb_logger.experiment.summary["last_ckpt"] = ckpt_cb_last.last_model_path
