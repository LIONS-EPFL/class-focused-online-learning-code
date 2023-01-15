import torch
import wandb


def compute_grad_norm(model):
    total_norm = torch.tensor(0.0)
    for p in model.parameters():
        if p.grad is not None:
            param_norm = (p.grad.detach().data ** 2).sum().cpu()
            total_norm += param_norm
    return total_norm ** 0.5


def plot_confusion_matrix(counts, class_names, title=None):
    """Based on `wandb.plot.confusion_matrix`.
    """
    assert counts.shape[0] == counts.shape[1]    
    n_classes = counts.shape[0]

    # Rest is taken from `wandb.plot.confusion_matrix`
    data = []
    for i in range(n_classes):
        for j in range(n_classes):
            data.append([class_names[i], class_names[j], counts[i, j]])

    fields = {
        "Actual": "Actual",
        "Predicted": "Predicted",
        "nPredictions": "nPredictions",
    }
    title = title or ""
    return wandb.plot_table(
        "wandb/confusion_matrix/v1",
        wandb.Table(columns=["Actual", "Predicted", "nPredictions"], data=data),
        fields,
        {"title": title},
    )
