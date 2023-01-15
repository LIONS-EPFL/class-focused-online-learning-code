from cfol.pgd import PGD
import os
from typing import List, Sequence, Union
from types import SimpleNamespace

import torch
import attr
from timm.utils.agc import adaptive_clip_grad
from timm.models.helpers import model_parameters
from timm.optim.optim_factory import create_optimizer
from torchattacks.attacks.tpgd import TPGD
from torchattacks.attacks.autoattack import AutoAttack


class Base(object):
    @classmethod
    def to_dict(cls, x):
        if isinstance(x, dict):
            return x
        else:
            return attr.asdict(x)

    @classmethod
    def to_cls(cls, x):
        if isinstance(x, cls):
            return x
        else:
            return cls(**x)

    @classmethod
    def factory(cls, **kwargs):
        return cls()


@attr.s()
class TEArgs(Base):
    momentum = attr.ib(type=float, default=0.9)
    weight = attr.ib(type=float, default=30.0)
    rampup_max_epoch = attr.ib(type=int, default=50)
    rampup_min_epoch = attr.ib(type=int, default=0)


@attr.s()
class OptArgs(Base):
    type = attr.ib(type=str, default="sgd", validator=attr.validators.in_({"sgd", "rmsproptf"}))
    lr = attr.ib(type=float, default=0.1)
    momentum = attr.ib(type=float, default=0.9)
    weight_decay = attr.ib(type=float, default=5e-4)
    scheduler = attr.ib(type=str, default='step', validator=attr.validators.in_({None, "step", "cyclic"}))

    clip_mode = attr.ib(type=str, default=None, validator=attr.validators.in_({None, "global", "AGC"}))
    grad_clip_val = attr.ib(type=float, default=None)

    # Step scheduler
    scheduler_milestones = attr.ib(type=list, default=[0.33, 0.66])
    scheduler_factor = attr.ib(type=float, default=0.1)

    # Cyclic scheduler
    lr_min = attr.ib(type=float, default=0.0)
    lr_max = attr.ib(type=float, default=0.2)

    def make(self, model):
        # Configurate model optimizer
        if self.type == "sgd":
            opt = torch.optim.SGD(
                model.parameters(), 
                lr=self.lr,
                momentum=self.momentum,
                weight_decay=self.weight_decay,
            )
        elif self.type == "rmsproptf":
            args = SimpleNamespace()
            args.opt = 'rmsproptf' 
            args.weight_decay = self.weight_decay
            args.lr = self.lr
            args.momentum = self.momentum
            opt = create_optimizer(args, model)
        else:
            raise ValueError("Invalid model opt", self)
        
        return opt

    def clip(self, model):
        """Call after backwards and prior to optimizer step.
        """
        if self.clip_mode == 'global':
            assert self.grad_clip_val is not None
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=self.grad_clip_val, norm_type=2)
        elif self.clip_mode == 'AGC':
            assert self.grad_clip_val is not None
            adaptive_clip_grad(model_parameters(model, exclude_head=True), clip_factor=self.grad_clip_val, norm_type=2)


@attr.s()
class AttackArgs(Base):
    type=attr.ib(default="pgd",validator=attr.validators.in_({'pgd', 'tpgd', 'autoattack'}))
    eps=attr.ib(default=0.031)
    lr=attr.ib(type=float, default=None)
    num_steps=attr.ib(default=7)
    use_best=attr.ib(default=True)

    def make(self, model, loss=None, num_classes=10):
        if self.type == "pgd":
            opt = PGD(
                model,
                loss=loss,
                eps=self.eps,
                alpha=self.lr,
                steps=self.num_steps,
                random_start=True,
                use_best=self.use_best)
        elif self.type == "tpgd":
            opt = TPGD(
                model,
                eps=self.eps,
                alpha=self.lr,
                steps=self.num_steps)
        elif self.type == "autoattack":
            # from autoattack import AutoAttack
            # adversary = AutoAttack(model, norm='Linf', eps=self.eps, version='standard')
            # opt = lambda images, labels: adversary.run_standard_evaluation(images, labels, bs=bs)
            opt = AutoAttack(model, norm='Linf', eps=self.eps, version='standard', n_classes=num_classes)
        else:
            raise ValueError("Invalid attack args", self)
        return opt

    def __attrs_post_init__(self):
        if self.lr is None:
            self.lr = 2.5 * self.eps / self.num_steps


ATTACK_DEFAULT = {'eps': 0.031, 'num_steps': 7}
TEST_ATTACK_DEFAULT = {'eps': 0.031, 'num_steps': 20}


@attr.s() #(auto_attribs=True)
class Hpars(Base):
    gpus = attr.ib(type=int, default=1, metadata=dict(help='number of gpus'))
    exp_dir = attr.ib(type=str, default='EXP', metadata=dict(help='directory of experiment'))
    project = attr.ib(type=str, default='focused-adversarial-examples', validator=attr.validators.in_({'longtail', 'focused-adversarial-examples'}))
    exp_name = attr.ib(type=str, default='debug', metadata=dict(help='name of experiment'))
    ckpt_path = attr.ib(type=str, default=None)
    
    # If set training will not run and ckpt will be updated
    test_best_avg_ckpt_path = attr.ib(type=str, default=None)
    test_best_min_ckpt_path = attr.ib(type=str, default=None)
    ckpt_is_pt_lightning = attr.ib(type=bool, default=True)
    ckpt_has_state_dict = attr.ib(type=bool, default=True)
    ckpt_remove_prefix = attr.ib(type=str, default=None)

    # See https://docs.wandb.ai/guides/track/advanced/resuming
    wandb_resume_id = attr.ib(type=str, default=None)

    precision = attr.ib(type=int, default=32)
    amp_level = attr.ib(type=str, default='O2', validator=attr.validators.in_({'O1', 'O2'}))

    val_split = attr.ib(type=Union[int, float], default=0.2)
    batch_size = attr.ib(type=int, default=128)    
    
    # epochs used for step-size scheduler while max_epochs can be overwritten to early stop training (e.g. for AT)
    epochs = attr.ib(type=int, default=150, metadata=dict(help='number of epochs'))
    max_epochs = attr.ib(type=int, default=None)
 
    # Logging
    progress_bar_refresh_rate = attr.ib(type=int, default=20, metadata=dict(help='Refresh rate of progress bar'))
    val_rate = attr.ib(type=int, default=5)
    log_verbose = attr.ib(type=bool, default=False) # Log images
    plot_verbose = attr.ib(type=bool, default=False)
    profiler = attr.ib(type=str, default=None, validator=attr.validators.in_({None, 'simple', 'pytorch-mem'}))

    # For debugging
    limit_train_batches = attr.ib(type=float, default=1.0)
    limit_val_batches = attr.ib(type=float, default=1.0)
    limit_test_batches = attr.ib(type=float, default=1.0)
    fast_dev_run = attr.ib(type=int, default=False)  # pt-lightning config
 
    # Dataset and transformations
    dataset = attr.ib(type=str, default="cifar10", validator=attr.validators.in_({'cifar10', 'cifar100', 'tinyimagenet', 'stl10', 'GTSRB', 'imagenette'}))
    num_workers = attr.ib(type=int, default=5)
    augment = attr.ib(type=bool, default=False)
    augment_type = attr.ib(type=str, default="v0", validator=attr.validators.in_({'v0', 'v1', 'v2', 'cutout', 'cutout2'}))
    cutout = attr.ib(type=bool, default=False)
    cutout_scale = attr.ib(type=float, default=0.5)
    use_te  = attr.ib(type=bool, default=False)
    te = attr.ib(type=TEArgs, default={}, converter=TEArgs.to_cls)
    normalize = attr.ib(type=bool, default=True)

    dataset_n_reduce = attr.ib(type=int, default=None)
    dataset_imbalance_factor = attr.ib(type=float, default=None)
    reduce_val = attr.ib(type=bool, default=False)

    # Model configs
    model = attr.ib(type=str, default="resnet50")
    model_opt = attr.ib(type=OptArgs, default={}, converter=OptArgs.to_cls)
    attack_opt = attr.ib(type=AttackArgs, default=ATTACK_DEFAULT, converter=AttackArgs.to_cls)
    test_attack_opt = attr.ib(type=AttackArgs, default=TEST_ATTACK_DEFAULT, converter=AttackArgs.to_cls)
    skipinit = attr.ib(type="str", default="zero", validator=attr.validators.in_({'zero', 'inverse-sqrt'}))

    # Method
    method = attr.ib(type=str, default='ERM', validator=attr.validators.in_({'ERM', 'CVaR', 'LCVaR', 'CFOL', 'FOL'}))
    trades = attr.ib(type=bool, default=False)
    trades_reg = attr.ib(type=float, default=6.0)

    # Class sampler  
    use_class_sampler = attr.ib(type=bool, default=False) 
    class_sampler_lr = attr.ib(type=float, default=0.00001)
    class_sampler_gamma = attr.ib(type=float, default=1/2)
    class_sampler_base_dist = attr.ib(type=str, default="uniform", validator=attr.validators.in_({"uniform", "empirical"}))
    class_sampler_prior = attr.ib(type=Union[str, Sequence[float]], default="uniform")
    class_sampler_reweight = attr.ib(type=bool, default=False)
    class_sampler_beta = attr.ib(type=float, default=0.0)
    class_sampler_reset_on_decay = attr.ib(type=bool, default=False)

    # Focused sampler
    use_focused_sampler = attr.ib(type=bool, default=False)
    update_focused_sampler = attr.ib(type=bool, default=False)
    focused_sampler_lr = attr.ib(type=float, default=0.00001)

    # CVaR
    use_cvar = attr.ib(type=bool, default=False)
    use_lcvar = attr.ib(type=bool, default=False)
    cvar_alpha = attr.ib(type=float, default=0.1)
    erm_pretrain_epochs = attr.ib(type=int, default=-1) # Only works for CVaR and LCVaR

    # Loss
    loss = attr.ib(type=float, default="CE", validator=attr.validators.in_({"CE", "focal"}))
    focal_gamma = attr.ib(type=float, default=2.0)
    loss_type = attr.ib(type=float, default="adv", validator=attr.validators.in_({"adv", "clean", "both"}))
    val_adv = attr.ib(type=bool, default=True)

    # Test
    test_best_avg = attr.ib(type=bool, default=True)
    test_best_min = attr.ib(type=bool, default=False)
    test_last = attr.ib(type=bool, default=True)
    log_test_confusion_matrix = attr.ib(type=bool, default=False)

    def __attrs_post_init__(self):
        self.out_dir = os.path.join(self.exp_dir, self.exp_name)

        if self.method == 'CFOL':
            self.use_class_sampler = True
        elif self.method == 'FOL':
            self.use_focused_sampler = True
        elif self.method == 'CVaR':
            self.use_cvar = True
        elif self.method == 'LCVaR':
            self.use_lcvar = True

        if sum([self.use_class_sampler, self.use_focused_sampler, self.use_cvar, self.use_lcvar]) > 1:
            raise ValueError("More than one of (use_class_sampler, use_focused_sampler, use_cvar, use_lcvar) cannot be active at the same time")
