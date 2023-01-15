from argparse import Namespace
from enum import Enum, auto
from fae.augmentation import Cutout
from fae.temporal_ensembling import TemporalEnsembling
from fae.preactresnet import PreActResNet18
from fae.wideresnet import WideResNet
from fae.sampler_scheduler import SamplerResetScheduler
from fae.lcvar_loss import LCVaRLoss
from fae.skipinit import SkipInit18, SkipInit50
from fae.resnet import ResNet18, ResNet50
from fae.resnet_madry import ResNet50 as ResNet50Madry
from fae.class_sampler import ClassSampler
from fae.utils import compute_grad_norm, plot_confusion_matrix
import warnings
import math 

import numpy as np
import torch
import timm
import torch.nn as nn
from torchvision import transforms
import torchvision
import wandb
from pytorch_lightning.profiler import PassThroughProfiler
import pytorch_lightning as pl
from pytorch_lightning.metrics.classification.confusion_matrix import ConfusionMatrix
from sklearn.metrics import ConfusionMatrixDisplay
from torch.nn import functional as F
import torchvision.models as models
from torch.distributions.utils import logits_to_probs, probs_to_logits
import matplotlib
matplotlib.use('Agg')
from fae.cvar.robust_losses import RobustLoss

import matplotlib.pyplot as plt

from fae.metrics import Histogram
from fae.config import AttackArgs, Hpars
from fae.focused_sampler import FocusedSampler


class Stage(Enum):
    train = 1
    val = 2
    test = 3

class LossType(Enum):
    clean = 1
    adv = 2


NORMALIZERS = {
    'cifar10': ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    'cifar100': ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    'stl10': ((0.43, 0.42, 0.39), (0.27, 0.26, 0.27)),
    'tinyimagenet': ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), # ImageNet normalization
    'imagenette': ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    'GTSRB': ((0.3337, 0.3064, 0.3171), ( 0.2672, 0.2564, 0.2629)), # https://github.com/poojahira/gtsrb-pytorch/blob/master/data.py
}

class Model(pl.LightningModule):
    def __init__(self, 
        hparams, 
        sampler=None, 
        profiler=None, 
        classes=None,
        train_targets=None,
        num_training_batches=None,
    ):

        super().__init__()
        
        if classes is None:
            raise ValueError("`classes` needs to be set with list of class names")

        self.automatic_optimization = False

        # For distinguishing test modes
        self.test_prefix = "best_avg"

        # profiling
        self.profiler = profiler or PassThroughProfiler()

        # make sure we have both hparam dict for tensorboard/ckpt and hpar for tab completion
        self.hparams = Hpars.to_dict(hparams)
        self.config = Hpars.to_cls(hparams)
        
        self.model_extrapolated = False
        
        self.classes = classes
        self.num_classes = len(classes)
        self.train_targets = train_targets
        self.num_training_batches = num_training_batches

        # LCVaR loss
        self.lcvar_loss = None
        if self.config.use_lcvar:
            # Use analytical form that does not require gradients of hyperparams
            self.lcvar_loss = LCVaRLoss(classes=self.num_classes,
                                alpha=self.config.cvar_alpha,
                                strat='analytic')

        # CVaR loss
        self.cvar_loss = None
        if self.config.use_cvar:
            self.cvar_loss = RobustLoss(
                size=self.config.cvar_alpha, 
                reg=0.0,
                geometry='cvar')

        if self.config.use_te:
            self.te = TemporalEnsembling(
                momentum=self.config.te.momentum,
                weight=self.config.te.weight,
                rampup_min_epoch=self.config.te.rampup_min_epoch,
                rampup_max_epoch=self.config.te.rampup_max_epoch,
                num_classes=self.num_classes,
                num_samples=len(train_targets),
            )

        self.sampler = sampler
        self.sampler_schedulers = []

        # Model
        # To list possible models: timm.list_models('*resne*t*')
        # TODO: change to base on dataset.dim size
        model_kwargs = dict(
            num_classes=self.num_classes,
            pool_adapt=bool(self.config.dataset in ['stl10', 'tinyimagenet', 'imagenette']))
        if self.config.model == 'linear':
            dim = torch.prod(torch.tensor(sampler.dataset.data.shape[1:]))
            model = nn.Sequential(
                nn.Flatten(),
                nn.Linear(dim, self.num_classes),
            )
        elif self.config.model == 'resnet18':
            model = ResNet18(**model_kwargs)
        elif self.config.model == 'vgg16':
            from torchvision.models import vgg16
            del model_kwargs['pool_adapt']
            model = vgg16(**model_kwargs)
        elif self.config.model == 'vgg16_bn':
            from fae.vgg import VGG
            del model_kwargs['pool_adapt']
            model = VGG('VGG16')
        elif self.config.model == 'resnet50':
            model = ResNet50(**model_kwargs)
        elif self.config.model == 'resnet50-madry':
            del model_kwargs['pool_adapt']
            model = ResNet50Madry(**model_kwargs)
        elif self.config.model == 'preactresnet18':
            if model_kwargs['pool_adapt']:
                raise ValueError("preactresnet18 does not support pool_adapt")
            del model_kwargs['pool_adapt']
            model = PreActResNet18(**model_kwargs)
        elif self.config.model == 'wideresnet':
            if model_kwargs['pool_adapt']:
                raise ValueError("wideresnet does not support pool_adapt")
            del model_kwargs['pool_adapt']
            model = WideResNet(**model_kwargs)
        elif self.config.model == 'skipinit18':
            model = SkipInit18(**model_kwargs)
        elif self.config.model == 'skipinit50':
            model = SkipInit50(**model_kwargs)
        else:
            model = timm.create_model(
                self.config.model,
                pretrained=False,
                **model_kwargs,
            )

        # Dataset specific normalization
        if self.config.normalize:
            normalizer = transforms.Normalize(*NORMALIZERS[self.config.dataset])
        else:
            normalizer = transforms.Normalize((0.0,0.0,0.0), (1.0,1.0,1.0))
        self.model = nn.Sequential(normalizer, model)

        # Initialize SkipInit
        if self.config.skipinit == 'inverse-sqrt':
            res_multiplier_keys = [k for (k,v) in model.named_parameters() if k.endswith('res_multiplier')]
            d = len(res_multiplier_keys)

            for k,v in model.named_parameters():
                if k.endswith('res_multiplier'):
                    v.data.fill_(1/math.sqrt(d))

        # Load potential checkpoint
        # TODO: disable since already taken care of by Trainer
        if self.config.ckpt_path is not None:
            print('loading model from %s' % self.config.ckpt_path)
            ckpt = torch.load(self.config.ckpt_path)['state_dict']
            state = {k.replace('model.', ''):v for k,v in ckpt.items() if k.startswith('model.')}
            self.model.load_state_dict(state)

        # Logging (see https://pytorch-lightning.readthedocs.io/en/stable/extensions/metrics.html)
        self.metrics = {
            (Stage.train, LossType.adv): pl.metrics.Accuracy(),
            (Stage.val, LossType.adv): pl.metrics.Accuracy(),
            (Stage.test, LossType.adv): pl.metrics.Accuracy(),

            (Stage.train, LossType.clean): pl.metrics.Accuracy(),
            (Stage.val, LossType.clean): pl.metrics.Accuracy(),
            (Stage.test, LossType.clean): pl.metrics.Accuracy(),

        }
        cm = ConfusionMatrix(num_classes=self.num_classes, normalize="true", compute_on_step=False)
        self.confusion_matrices = {
            (Stage.train, LossType.adv): cm.clone(),
            (Stage.val, LossType.adv): cm.clone(),
            (Stage.test, LossType.adv): cm.clone(),

            (Stage.train, LossType.clean): cm.clone(),
            (Stage.val, LossType.clean): cm.clone(),
            (Stage.test, LossType.clean): cm.clone(),
        }

        self.sampler_realised_class_hist = Histogram(num_bins=self.num_classes)

        self.val_adv_acc_argmax_epoch = torch.tensor(-1.0)
        self.val_adv_acc_max = torch.tensor(-1.0)
        self.val_adv_min_class_precision_max_avg = torch.tensor(-1.0)

        # Register to have the right device type
        self.metrics_modules = nn.ModuleList([metric for metric in self.metrics.values()])
        self.confusion_matrices_modules = nn.ModuleList([metric for metric in self.confusion_matrices.values()])

    def forward(self, x):
        # return logits
        return self.model(x)

    def loss(self, idxs, logits, y, reduction="mean", stage=Stage.train):
        if self.config.loss == 'CE':
            loss = F.cross_entropy(logits, y, reduction=reduction)
        elif self.config.loss == 'focal':
            #loss = FocalLoss(gamma=self.config.focal_gamma, reduction=reduction)
            #return loss(logits, y)
            # CB_loss
            raise NotImplementedError
        else:
            raise ValueError("Invalid config.loss")

        if self.config.use_te and stage == Stage.train:
            loss = loss + self.te(self.current_epoch, idxs, logits, reduction=reduction)
        
        return loss

    def predict(self, logits):
        return logits.argmax(dim=-1)

    def configure_optimizers(self):
        opt = self.config.model_opt.make(self.model)

        # Model scheduler
        lr_steps = self.config.epochs * self.num_training_batches
        if self.config.model_opt.scheduler == 'step':
            milestones = np.array(self.config.model_opt.scheduler_milestones)
            
            if milestones.dtype == int:
                # Treat as epochs
                milestones = milestones * self.num_training_batches
            else:
                # Treat as percentage of total steps
                milestones = milestones * lr_steps
            
            milestones = milestones.astype(int)

            # Reset sampler on step-size schedule
            if self.config.class_sampler_reset_on_decay:
                self.sampler_schedulers = [SamplerResetScheduler(self.sampler, milestones)]

            # TODO: for some obscure reason scheduler.step() seems to be called twice so multiple by 2
            hack_factor = 2
            milestones *= hack_factor
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                    opt, 
                    milestones,
                    gamma=self.config.model_opt.scheduler_factor)
            schedulers = [{
                'scheduler': scheduler,
                'name': 'model_lr',
                'interval':'step',
                'frequency': 1,
            }]

        elif self.config.model_opt.scheduler == 'cyclic':
            scheduler = torch.optim.lr_scheduler.CyclicLR(
                opt, 
                base_lr=self.config.model_opt.lr_min, max_lr=self.config.model_opt.lr_max,
                step_size_up=lr_steps / 2, 
                step_size_down=lr_steps / 2)
            schedulers = [{
                'scheduler': scheduler,
                'name': 'model_lr',
                'interval':'step',
                'frequency': 1,
            }]
        else:
            schedulers = []

        return [opt], schedulers

    def _attack(self, batch, stage):
        img, label, idxs = batch
        
        if stage == Stage.train:
            attack_config: AttackArgs = self.config.attack_opt
        else:
            attack_config: AttackArgs = self.config.test_attack_opt

        loss = lambda logits, y: self.loss(idxs, logits, y, reduction="mean", stage=stage)

        attack = attack_config.make(self.model, loss, num_classes=self.num_classes)

        prev_training = bool(self.training)
        self.eval()

        with torch.enable_grad():
            adv_img = attack(img, label)
            if prev_training:
                self.train()
            return adv_img, label, idxs

    def _get_TRADES_loss(self, batch, reduction="mean", stage=Stage.train):
        if reduction == "mean":
            reduction = "batchmean"

        img, label, idxs = batch
        criterion_kl = nn.KLDivLoss(reduction=reduction)
        
        loss = lambda logits, y: criterion_kl(F.log_softmax(logits, dim=1),
                                              F.softmax(self.model(img), dim=1))
        
        return loss

    def _TRADES_attack(self, batch, stage):
        img, label, idxs = batch
        
        if stage == Stage.train:
            attack_config: AttackArgs = self.config.attack_opt
        else:
            attack_config: AttackArgs = self.config.test_attack_opt

        loss = self._get_TRADES_loss(batch, reduction="sum", stage=stage)
        attack = attack_config.make(self.model, loss, num_classes=self.num_classes)

        prev_training = bool(self.training)
        self.eval()

        with torch.enable_grad():
            adv_img = attack(img, label)
            if prev_training:
                self.train()
            return adv_img, label, idxs

    def training_step(self, batch, batch_idx):
        # TODO: the flag=True prevents extrapolation
        opt_model = self.optimizers(use_pl_optimizer=False)
        
        if self.config.trades:
            # Note that temporal ensemble is applied to the clean loss currently
            assert self.config.method == 'ERM', "TRADES only compatible with ERM currently"
            assert self.config.loss_type == 'both'
            clean, clean_logits = self._compute_loss(batch, batch_idx, stage=Stage.train, type=LossType.clean, reduction="none", return_logits=True)
            adv, adv_logits = self._compute_TRADES_regularization(batch, batch_idx, stage=Stage.train, type=LossType.adv, reduction="none", return_logits=True)
            
            # TODO: current `mean` since KLloss contains logits dimension as well
            clean = clean.mean()
            adv = self.config.trades_reg * adv.mean() 
        else:
            # Compute and log clean loss
            with self.profiler.profile('fae_compute_clean_loss'):
                # Don't update batch stats using clean if only adv loss is used
                if self.config.loss_type != 'adv':
                    clean, clean_logits = self._compute_loss(batch, batch_idx, stage=Stage.train, type=LossType.clean, reduction="none", return_logits=True)
                else:
                    clean, clean_logits = None, None    
            
            # Compute and log adv loss
            with self.profiler.profile('fae_compute_adv_loss'):
                if self.config.loss_type != 'clean':
                    # Only compute adv loss if necessary
                    adv, adv_logits = self._compute_loss(batch, batch_idx, stage=Stage.train, type=LossType.adv, reduction="none", return_logits=True)
                else:
                    adv, adv_logits = None, None

        # Ensure attack has not affected the model gradient
        opt_model.zero_grad()

        # Construct train loss
        loss = 0.0        
        if self.config.loss_type in ['clean', 'both']:
            loss += clean
        if self.config.loss_type in ['adv', 'both']:
            loss += adv
        
        # Weight examples if using ClassSampler
        if self.config.use_class_sampler and self.config.class_sampler_reweight:
            sampler: ClassSampler = self.sampler
            targets = batch[1]
            weighted_loss = self.num_classes * sampler.batch_weight(targets).type_as(loss) * loss
            weighted_loss.mean().backward()
        else:
            loss.mean().backward()

        with self.profiler.profile('fae_model_step'):
            # Optional gradient clipping
            self.config.model_opt.clip(self.model)
            
            # Step
            self._model_step(opt_model)
            self.log("grad_norm", compute_grad_norm(self.model))
            opt_model.zero_grad()

        # Update the adversarial sampler
        with self.profiler.profile('fae_focused_sampler_update'):
            if self.config.use_focused_sampler and self.config.update_focused_sampler:
                sampler: FocusedSampler = self.sampler

                # Use the adversarial logits if available
                if adv_logits is not None:
                    logits = adv_logits
                else:
                    logits = clean_logits

                # FOL loss based on prediction
                y = batch[1]
                class_loss = self.predict(logits) != y
                nominator = self.config.focused_sampler_lr * class_loss

                idxs = batch[2]
                for i, idx in enumerate(idxs):
                    nom = nominator[i]
                    sampler.update(idx, nom.item())

        # Update class sampler 
        with self.profiler.profile('fae_class_sampler_update'):
            if self.config.use_class_sampler:
                sampler: ClassSampler = self.sampler

                # Use the adversarial logits if available
                if adv_logits is not None:
                    logits = adv_logits
                else:
                    logits = clean_logits

                # FOL loss based on prediction
                y = batch[1]
                class_loss = self.predict(logits) != y
                eta_times_loss_arms = self.config.class_sampler_lr * class_loss + self.config.class_sampler_beta
                sampler.batch_update(y, eta_times_loss_arms)


        # TE
        if self.config.use_te:
            # Use the adversarial logits if available
            if adv_logits is not None:
                logits = adv_logits
            else:
                logits = clean_logits
            indices = batch[2]
            self.te.update(indices, logits, self.current_epoch)

        # Log
        targets = batch[1]
        self.sampler_realised_class_hist.update(targets)

        # Step schedulers manually since automatic_optimization=False
        for lr_scheduler in self.trainer.lr_schedulers:
            lr_scheduler['scheduler'].step()

        for scheduler in self.sampler_schedulers:
            scheduler.step()

    def _compute_loss(self, batch, batch_idx, stage=Stage.train, type=LossType.clean, reduction="mean", return_logits=False):
        """Be aware that this function will reduce irrespectively of `reduction=none` if LCVaR loss is used.
        However, this reduction is to support FOL methods which are not used in conjunction with CVaR anyway.
        Calling `mean` or `sum` subsequentially still works since they are noops.
        """
        # Attack
        if type == LossType.adv:
            batch = self._attack(batch, stage)

        # Compute loss
        x, y, idxs = batch
        logits = self(x)
        #y_hat = F.softmax(logits, dim=-1)
        y_hat = self.predict(logits)

        force_erm = self.current_epoch < self.config.erm_pretrain_epochs

        if not force_erm and self.config.use_lcvar and stage == Stage.train:
            # LCVaR
            losses = self.loss(idxs, logits, y, reduction='none', stage=stage)
            loss = self.lcvar_loss(losses, y)
        elif not force_erm and self.config.use_cvar and stage == Stage.train:
            # CVaR
            losses = self.loss(idxs, logits, y, reduction='none', stage=stage)
            loss = self.cvar_loss(losses)
        else:
            # Normal cross entropy
            loss = self.loss(idxs, logits, y, reduction=reduction, stage=stage)
        
        # Log loss
        on_step = stage == Stage.train
        self.metrics[(stage, type)](y_hat, y)
        self.log(f'{self.stage_name(stage)}_{type.name}_acc', self.metrics[(stage, type)], 
                 on_step=on_step, on_epoch=True, prog_bar=True)
        self.log(f'{self.stage_name(stage)}_{type.name}_loss', loss.mean(), prog_bar=True)

        if batch_idx == 0:
            self._log_examples(x, stage, type)

        # Log confusion matrix
        self.confusion_matrices[(stage, type)](y_hat, y)

        if return_logits:
            return loss, logits
        else:
            return loss


    def _compute_TRADES_regularization(self, batch, batch_idx, stage=Stage.train, type=LossType.clean, reduction="mean", return_logits=False):
        """TRADES does not support CVaR methods currently.
        """
        trades_reg_loss = self._get_TRADES_loss(batch, reduction=reduction, stage=stage)
        if self.config.attack_opt.type == 'tpgd':
            batch = self._attack(batch, stage)
        else:
            batch = self._TRADES_attack(batch, stage)

        # Compute loss
        x, y, idxs = batch
        logits = self(x)
        #y_hat = F.softmax(logits, dim=-1)
        y_hat = self.predict(logits)

        # KL w.r.t to prediction on clean
        loss = trades_reg_loss(logits, y)
        
        # Log loss
        on_step = stage == Stage.train
        self.metrics[(stage, type)](y_hat, y)
        self.log(f'{self.stage_name(stage)}_{type.name}_acc', self.metrics[(stage, type)], 
                 on_step=on_step, on_epoch=True, prog_bar=True)
        self.log(f'{self.stage_name(stage)}_{type.name}_loss', loss.mean(), prog_bar=True)

        if batch_idx == 0:
            self._log_examples(x, stage, type)

        # Log confusion matrix
        self.confusion_matrices[(stage, type)](y_hat, y)

        if return_logits:
            return loss, logits
        else:
            return loss

    def _model_step(self, opt_model):
        if hasattr(opt_model, "extrapolation") and not self.model_extrapolated:
            opt_model.extrapolation()
            self.model_extrapolated=True
        else:
            opt_model.step()
            self.model_extrapolated=False

    def on_train_epoch_end(self, epoch_output):        

        # Log sampler related at the end of epoch
        if self.current_epoch % 5 == 0:
            # Log realized class sampling for this epoch
            hist = self.sampler_realised_class_hist.compute().detach().cpu().numpy()
            bins = np.arange(self.num_classes + 1)
            np_histogram = hist, bins
            self.logger.experiment.log({"sampler_realised_class_hist": wandb.Histogram(np_histogram=np_histogram)},
                                    commit=False)

            self._log_class_bar_plot("sampler_realised_class_hist_fig", "frequency", hist)

            # Log focused sampler
            if self.config.use_focused_sampler:
                sampler: FocusedSampler = self.sampler
                
                # Log FocusedSampler current probs 
                qs = np.array([n.q for n in sampler.tree.leaf_nodes])
                self.logger.experiment.log({"focused_sampler_prob_hist": wandb.Histogram(qs)}, commit=False)

                # Create histogram over classes
                qs = np.array([n.q for n in sampler.tree.leaf_nodes])
                class_hist = np.bincount(self.train_targets, weights=qs)
                class_hist = class_hist / class_hist.sum()
                bins = np.arange(self.num_classes + 1)
                np_histogram = class_hist, bins
                self.logger.experiment.log({"sampler_class_prob_hist": wandb.Histogram(np_histogram=np_histogram)}, 
                                        commit=False)
                self._log_class_bar_plot("sampler_class_prob_hist_fig", "probability", class_hist)

            # Log class sampler distributions
            elif self.config.use_class_sampler:
                hist = self.sampler.q
                bins = np.arange(self.num_classes + 1)
                np_histogram = hist, bins
                self.logger.experiment.log({"sampler_class_prob_hist": wandb.Histogram(np_histogram=np_histogram)},
                                        commit=False)
                hist = hist.detach().cpu().numpy()
                self._log_class_bar_plot("sampler_class_prob_hist_fig", "probability", hist)
        
        # Reset histogram
        self.sampler_realised_class_hist.reset()

        # Log confusion matrix
        if self.current_epoch % 5 == 0:
            self._log_confusion_matrix(Stage.train, LossType.clean)
            if self.config.loss_type != 'clean':
                self._log_confusion_matrix(Stage.train, LossType.adv)
        self.confusion_matrices[(Stage.train, LossType.clean)].reset()
        self.confusion_matrices[(Stage.train, LossType.adv)].reset()

        self.log('epoch', self.current_epoch)

    def on_validation_epoch_end(self) -> None:
        self._log_confusion_matrix(Stage.val, LossType.clean)
        if self.config.val_adv:
            precision = self._log_confusion_matrix(Stage.val, LossType.adv)

            # Store val stats for current best accuracy
            avg_prec = np.average(precision)
            min_prec = np.min(precision)
    
            if avg_prec >= self.val_adv_acc_max:
                self.val_adv_acc_argmax_epoch = self.current_epoch
                self.val_adv_acc_max = avg_prec
                self.val_adv_min_class_precision_max_avg = min_prec

            self.log('val_adv_acc_argmax_epoch', self.val_adv_acc_argmax_epoch)
            self.log('val_adv_acc_max', self.val_adv_acc_max)
            self.log('val_adv_min_class_precision_max_avg', self.val_adv_min_class_precision_max_avg)
                

    def on_test_epoch_end(self) -> None:
        self._log_confusion_matrix(Stage.test, LossType.clean)
        if self.config.val_adv:
            self._log_confusion_matrix(Stage.test, LossType.adv)

    def validation_step(self, batch, batch_idx):
        self._compute_loss(batch, batch_idx, stage=Stage.val, type=LossType.clean)
        if self.config.val_adv:
            self._compute_loss(batch, batch_idx, stage=Stage.val, type=LossType.adv)

    def test_step(self, batch, batch_idx):
        self._compute_loss(batch, batch_idx, stage=Stage.test, type=LossType.clean)
        if self.config.val_adv:
            self._compute_loss(batch, batch_idx, stage=Stage.test, type=LossType.adv)

    def _log_confusion_matrix(self, stage: Stage, type: LossType):
        cm = self.confusion_matrices[(stage, type)].compute().detach().cpu().numpy()

        # We cannot use self.log here because of pickling issues
        # Use skilearn 
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                    display_labels=self.classes)
        if self.config.plot_verbose:
            disp.plot(include_values=False, xticks_rotation='vertical',colorbar=True)
            self.logger.experiment.log({f'{self.stage_name(stage)}_{type.name}_confusion_matrix': disp.figure_}, commit=False)

        # # Use plotly heatmap
        # self.logger.experiment.log({f'{self.stage_name(stage)}_confusion_matrix_plotly': wandb.Image(wandb.plots.HeatMap(self.classes, self.classes, cm, show_text=False))}, step=self.global_step)

        # Use table based plotting for test
        if stage == Stage.test and self.config.log_test_confusion_matrix:
            self.logger.experiment.log({f'{self.stage_name(stage)}_confusion_matrix_table': plot_confusion_matrix(cm, self.classes)}, step=self.global_step)

        # Log precision and min precision
        precision = [cm[i,i] for i in range(len(self.classes))]
        self._log_class_bar_plot(f'{self.stage_name(stage)}_{type.name}_class_precision', "precision", precision)
        self.log(f'{self.stage_name(stage)}_{type.name}_min_class_precision', np.min(precision))
        self.log(f'{self.stage_name(stage)}_{type.name}_max_class_precision', np.max(precision))
        self.log(f'{self.stage_name(stage)}_{type.name}_avg_class_precision', np.mean(precision))

        # Reset every epoch
        self.confusion_matrices[(stage, type)].reset()

        return precision

    def _log_class_bar_plot(self, logname, name, data):

        if self.config.plot_verbose:
            x_pos = range(len(self.classes))
            fig, ax = plt.subplots()
            ax.bar(x_pos, data)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(self.classes, rotation='vertical')
            ax.set_xlabel("class_name")
            ax.set_ylabel(name)
            ax.set_ylim(0, 1)
            self.logger.experiment.log({logname: wandb.Image(fig)}, commit=False)
            plt.close(fig)
        
        # Save to a table for plotting in the future
        data = list(zip(data, self.classes))
        table = wandb.Table(data=data, columns=["class_name", name])
        self.logger.experiment.log({f'{logname}_table' : table}, commit=False)


    def _log_examples(self, img, stage: Stage, type: LossType, N=20):
        # For debugging possible augmentations and the attack
        if self.config.log_verbose:
            img_sample = img[:N]
            grid = torchvision.utils.make_grid(img_sample)
            self._log_image(f'{self.stage_name(stage)}_{type.name}_img', grid)

    def _log_image(self, title, image):
        self.logger.experiment.log({title: wandb.Image(image)}, commit=False)

    def stage_name(self, stage: Stage):
        """Allow for storing multiple test metrices under different names
        """
        name = stage.name
        if stage == Stage.test:
            name = f"{name}_{self.test_prefix}"
        return name
