import random

import evaluate
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef


def move(device, *tensors: torch.Tensor) -> list[torch.Tensor]:
    moved_tensors = []
    for tensor in tensors:
        moved_tensor = tensor.to(device)
        moved_tensors.append(moved_tensor)
    return moved_tensors


def accuracy(y: torch.Tensor, y_hat: torch.Tensor) -> float:
    _, predicted = torch.max(y_hat, 1)
    return 100 * accuracy_score(y.cpu(), predicted.cpu())


def f1(y: torch.Tensor, y_hat: torch.Tensor) -> float:
    _, predicted = torch.max(y_hat, 1)
    results = f1_score(y.cpu(), predicted.cpu(), average="weighted", zero_division=0)
    return 100 * results


def matthews_corrcoeff(y: torch.Tensor, y_hat: torch.Tensor) -> float:
    _, predicted = torch.max(y_hat, 1)
    return matthews_corrcoef(y.cpu(), predicted.cpu())


def pearson_corrcoeff(y: torch.Tensor, y_hat: torch.Tensor) -> float:
    metric = evaluate.load("pearsonr")
    result = metric.compute(references=y.cpu(), predictions=y_hat.cpu())
    return 100 * result["pearsonr"]


def eval_func(metric_name: str):
    if metric_name == "acc":
        return accuracy
    elif metric_name == "f1":
        return f1
    elif metric_name == "mcorr":
        return matthews_corrcoeff
    elif metric_name == "pearson":
        return pearson_corrcoeff
    else:
        raise ValueError(f"Unknown metric name: {metric_name}")


def set_seeds(seed_no: int):
    random.seed(seed_no)
    np.random.seed(seed_no)
    torch.manual_seed(seed_no)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    return device
