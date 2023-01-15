import torch
import torch.nn as nn

from torchattacks.attack import Attack


class PGD(Attack):
    r"""
    PGD in the paper 'Towards Deep Learning Models Resistant to Adversarial Attacks'
    [https://arxiv.org/abs/1706.06083]

    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (DEFAULT: 0.3)
        alpha (float): step size. (DEFAULT: 2/255)
        steps (int): number of steps. (DEFAULT: 40)
        random_start (bool): using random initialization of delta. (DEFAULT: False)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Modification:
        - Allows for passing in a loss instead of cross entropy.
        - It can pick the best attack along the trajectory

    Examples::
        >>> attack = PGD(model, eps=8/255, alpha=1/255, steps=40, random_start=False)
        >>> adv_images = attack(images, labels)

    """

    def __init__(self, model, loss=None, eps=0.3, alpha=2 / 255, steps=40, random_start=False, use_best=True):
        super(PGD, self).__init__("PGD", model)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
        self.use_best = use_best

        if loss is not None:
            self.loss = loss
        else:
            self.loss = nn.CrossEntropyLoss()

    def forward(self, images, labels):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        labels = self._transform_label(images, labels)

        adv_images = images.clone().detach()

        best_loss = None
        best_x = None

        # A function that updates the best loss and best input
        def replace_best(loss, bloss, x, bx):
            if bloss is None:
                bx = x.clone().detach()
                bloss = loss.clone().detach()
            else:
                replace = bloss > loss
                bx[replace] = x[replace].clone().detach()
                bloss[replace] = loss[replace]

            return bloss, bx

        if self.random_start:
            # Starting at a uniformly random point
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(-self.eps, self.eps)
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        for i in range(self.steps):
            adv_images.requires_grad = True
            outputs = self.model(adv_images)

            # cost to be *minimized*
            cost = self._targeted * self.loss(outputs, labels)

            grad = torch.autograd.grad(cost, adv_images,
                                       retain_graph=False, create_graph=False)[0]

            adv_images = adv_images.detach() - self.alpha * grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

            args = [cost, best_loss, adv_images, best_x]
            best_loss, best_x = replace_best(*args) if self.use_best else (cost, adv_images)

        return best_x
