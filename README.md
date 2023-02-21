# Class focused online learning (CFOL)

Official code for [Revisiting adversarial training for the worst-performing class](https://openreview.net/pdf?id=wkecshlYxI) accepted at TMLR 2023.

## Setup

For CPU:

```bash
conda create -n cfol python=3.6
conda activate cfol
python setup.py develop
```


For GPU:

```bash
conda create -n cfol python=3.6
conda activate cfol
conda install cudatoolkit=11 torchvision -c pytorch
python setup.py develop
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
```

For wandb:

- Delete wandb entry from `/home/<user>/.netrc` if present to prevent auto-login to a different account
- Storage your key with `vim .env`:
  ```bash
  export WANDB_API_KEY=<mykey>
  ```
- Before running a script run `source .env`

## File overview

```
├── bulk_script.py                (Script for generating experiments with standardized configs)
└── cfol                           
    ├── runner.py                 (Entry point setting up the dataset and model)
    ├── model.py                  (Specifies the training and testing)
    ├── config.py                 (Available configurations)
    ├── class_sampler.py          (CFOL implementation)
    └── focused_sampler.py        (FOL implementation)
```

## Usage of `CFOL`

The code below contains minimal boilerplate for using CFOL in another codebase.

```python
######################### Setup ##########################
from cfol.focused_sampler import InMemoryDataset
from cfol.class_sampler import ClassSampler
dataset = InMemoryDataset(data, targets, transforms=...)
sampler = ClassSampler(dataset, gamma=0.5)
dataloader = DataLoader(dataset, ..., sampler=sampler)


################## Inside training loop ##################

# Compute loss with reduction="none" (such that it maintains the batch dimension)
loss = F.cross_entropy(logits, y, reduction="none")

# Possibly weight losses
if sampler.reweight:
    weighted_loss = self.num_classes * sampler.batch_weight(y).type_as(loss) * loss
    weighted_loss.mean().backward()
else:
    loss.mean().backward()
  
# Update sampler
class_sampler_lr = 0.0000001
class_loss = self.predict(logits) != y
eta_times_loss_arms = class_sampler_lr * class_loss
sampler.batch_update(y, eta_times_loss_arms)
```

## Run

```bash
python cfol/runner.py print_config
python cfol/runner.py with h.gpus=0 h.model_opt.lr=0.01
```

The command for a particular experiment can be generated with default configurations using `cmd_generator.py` (see below).


## Experiments

```bash
# CIFAR 10
python cmd_generator.py --method CFOL --dataset cifar10 --optional "h.epochs=200" "h.model_opt.scheduler_milestones='[0.5,0.75]'" "h.model=resnet18" "h.model_opt.lr=0.1"
python cmd_generator.py --method ERM --dataset cifar10 --optional "h.epochs=200" "h.model_opt.scheduler_milestones='[0.5,0.75]'" "h.model=resnet18"
python cmd_generator.py --method FOL --dataset cifar10 --optional "h.epochs=200" "h.model_opt.scheduler_milestones='[0.5,0.75]'" "h.model=resnet18" "h.model_opt.lr=0.1" "h.focused_sampler_lr=0.0000005"
python cmd_generator.py --method CVaR --dataset cifar10 --optional "h.epochs=200" "h.model_opt.scheduler_milestones='[0.5,0.75]'" "h.model=resnet18" "h.model_opt.lr=0.1" "h.cvar_alpha=0.5"
python cmd_generator.py --method LCVaR --dataset cifar10 --optional "h.epochs=200" "h.model_opt.scheduler_milestones='[0.5,0.75]'" "h.model=resnet18" "h.model_opt.lr=0.1" "h.cvar_alpha=0.2"
```

## Citing

If you use this code please cite the following work: 

```bibtex
@article{pethick2023revisiting,
  title={Revisiting adversarial training for the worst-performing class},
  author={Pethick, Thomas and Chrysos, Grigorios and Cevher, Volkan},
  journal={Transactions on Machine Learning Research},
  year={2023}
}
```


## Acknowledgements

The codebase relies on the following implementations:

- CVaR implementation: https://github.com/daniellevy/fast-dro
- LCVaR implementation: https://github.com/neilzxu/robust_weighted_classification
