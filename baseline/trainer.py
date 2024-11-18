from dataclasses import dataclass
from typing import Union

import torch
import torch.nn as nn
from torch.nn import MSELoss, NLLLoss
from torch.utils.data.dataloader import DataLoader
from tqdm.auto import tqdm

from baseline import ResultMetricName
from baseline.utils import move


@dataclass
class TrainerArgs:
    train_data_loader: DataLoader
    val_data_loader: DataLoader
    metric_name: ResultMetricName
    evaluation_function: callable
    lr: float
    device: torch.device


class Trainer:
    def __init__(
        self, model: nn.Module, criterion: Union[NLLLoss, MSELoss], args: TrainerArgs
    ):
        self.args = args
        self.model = model.to(self.args.device)
        self.criterion = criterion
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)

    def train_epoch(self):
        self.model.train()
        e_loss = 0

        for batch in tqdm(self.args.train_data_loader):
            i_ids, a_mask, y_b = batch
            i_ids, a_mask, y_b = move(self.args.device, i_ids, a_mask, y_b)
            e_loss += self.train_batch(i_ids, a_mask, y_b)
        return e_loss

    def train_batch(self, i_ids: torch.Tensor, a_mask: torch.Tensor, y_b: torch.Tensor):
        self.optimizer.zero_grad()
        y_hat, cls_embed = self.model(i_ids, a_mask)
        loss = self.criterion(y_hat, y_b)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def evaluate(self, data_key: str):
        self.model.eval()
        dl = self.check_data_key(data_key)

        with torch.no_grad():
            y_hat, y = [], []
            for batch in dl:
                i_ids, a_mask, y_b = batch
                i_ids, a_mask = move(self.args.device, i_ids, a_mask)
                y_hat_b, cls_embed = self.model(i_ids, a_mask)
                y_hat.append(y_hat_b.to("cpu"))
                y.append(y_b)

            y = torch.cat(y, dim=0)
            y_hat = torch.cat(y_hat, dim=0)
            metric_value = self.args.evaluation_function(y, y_hat)
            return metric_value

    def check_data_key(self, data_key):
        if data_key == "train":
            dl = self.args.train_data_loader
        elif data_key == "validation":
            dl = self.args.val_data_loader
        else:
            raise ValueError(
                f"Invalid data_key: {data_key}. Expected 'train' or 'validation'."
            )
        return dl
