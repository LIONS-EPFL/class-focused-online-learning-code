import argparse
import subprocess as sp
import ast


# Argparse boolean helper
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        # raise argparse.ArgumentTypeError('Boolean value expected.')
        return v


# Bulk configs
parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=int, default=None)
parser.add_argument("--adv", type=str2bool, nargs='?', const=True, default=True)
parser.add_argument("--method", type=str, default="ERM")
parser.add_argument("--dataset", type=str, default="cifar10")
parser.add_argument('--optional', nargs='*', default=[])
parser.add_argument("--suffix", type=str, default=None)
args = parser.parse_args()

# Defaults
defaults = {
    "h.loss_type": "clean",
    "h.dataset": "cifar10",
    "h.model": "resnet50",
    "h.batch_size": "128",
    "h.augment": "True",
    "h.augment_type": "v2",
    "h.use_te": "False",
    "h.te.rampup_max_epoch": "50",
    "h.epochs": "100",
    "h.model_opt.scheduler_milestones": '"[0.75,0.90]"',
    "h.model_opt.lr": "0.1",
    "h.val_split": "0.2",
    "h.gpus": "1",
    "h.test_best_min": "True",
    "h.val_rate": "1",
}

adv = {
    "h.loss_type": "adv",
    "h.attack_opt.type": "pgd",
    "h.attack_opt.num_steps": "7",
    "h.attack_opt.eps": "0.031",
    "h.attack_opt.lr": "0.00797",
    "h.test_attack_opt.eps": "0.031",
}

# Methods
ERM = {
    "h.method": "ERM",
}

CFOL = {
    "h.method": "CFOL",
    "h.class_sampler_lr": "0.000005",
    "h.class_sampler_base_dist": "uniform",
}

FOL = {
    "h.method": "FOL",
    "h.update_focused_sampler": "True",
    "h.focused_sampler_lr": "0.000005",
}

LCVaR = {
    "h.method": "LCVaR",
    "h.cvar_alpha": "0.2"
}

CVaR = {
    "h.method": "CVaR",
    "h.cvar_alpha": "0.2"
}

methods = {
    'ERM': ERM,
    'CFOL': CFOL,
    'FOL': FOL,
    'LCVaR': LCVaR,
    'CVaR': CVaR,
}

# Datasets

GTSRB = {
    "h.dataset": "GTSRB",
}

cifar10 = {
    "h.dataset": "cifar10",
}

cifar100 = {
    "h.dataset": "cifar100",
}

tinyimagenet = {
    "h.dataset": "tinyimagenet",
}

imagenette = {
    "h.dataset": "imagenette",
}

stl10 = {
    "h.dataset": "stl10",
}


datasets = {
    'cifar10': cifar10,
    'cifar100': cifar100,
    'tinyimagenet': tinyimagenet,
    'imagenette': imagenette,
    'stl10': stl10,
    'GTSRB': GTSRB,
}

def config_to_name(config):
    n = []

    n.append(config["h.dataset"])
    n.append("aug" + config["h.augment_type"])
    
    # Method
    method = config["h.method"]
    n.append(method)
    if method == "CFOL":
        n.append("eta" + config["h.class_sampler_lr"])
        if config.get("h.class_sampler_gamma") is not None:
            n.append('gamma' + config["h.class_sampler_gamma"])

    elif method == "FOL":
        n.append("eta" + config["h.focused_sampler_lr"])
    elif method in ["LCVaR", "CVaR"]:
        n.append("alpha" + config["h.cvar_alpha"])
    

    # Scheduler
    if config["h.use_te"] == "True":
        n.append("TE" + config["h.te.rampup_max_epoch"])
    n.append("epoch" + config["h.epochs"])
    milestones = "-".join(map(str,ast.literal_eval(config["h.model_opt.scheduler_milestones"][1:-1])))
    n.append("ms-" + milestones)

    # Loss
    n.append(config["h.loss_type"])
    if config["h.loss_type"] == 'adv':
        n.append(config["h.attack_opt.type"] + config["h.attack_opt.num_steps"])
    
    # Model
    n.append(config["h.model"])
    n.append("lr" + config["h.model_opt.lr"])
    if config["h.batch_size"] != "128":
        n.append("bs" + config["h.batch_size"])

    if args.suffix is not None:
        n.append(args.suffix)

    return "/".join(n)

# Execute
configs = {}
configs.update(defaults)
if args.adv:
    configs.update(adv)
configs.update(methods[args.method])
configs.update(datasets[args.dataset])

for a in args.optional:
    k,v = a.split("=")
    configs[k] = v

if args.gpu is not None:
    prefix = f"CUDA_VISIBLE_DEVICES={args.gpu} "
else:
    prefix = ""

cmd = [
    f"{prefix}python fae/runner.py with",
    f"h.exp_name={config_to_name(configs)}",
]
cmd = cmd + [f"{k}={v}" for k,v in configs.items()]
print(" \\\n".join(cmd))
