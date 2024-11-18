from dataclasses import dataclass
from typing import Union

import torch
from torch.nn import MSELoss, NLLLoss
from torch.utils.data.dataloader import DataLoader
from tqdm.auto import tqdm

from baseline import ResultMetricName
from baseline.utils import move
from graph_teacher.dataset import GraphDataset
from graph_teacher.models import InterpolationModel


@dataclass
class TrainerArgs:
    train_data_loader_full: DataLoader
    train_data_loader: DataLoader
    val_data_loader: DataLoader
    graph_dataset: GraphDataset
    metric_name: ResultMetricName
    evaluation_function: callable
    gcn_lr: float
    encoder_lr: float
    device: torch.device


class Trainer:
    def __init__(
        self,
        model: InterpolationModel,
        criterion: Union[NLLLoss, MSELoss],
        args: TrainerArgs,
    ):
        self.args = args
        self.model = model.to(self.args.device)
        self.args.graph_dataset.graph_data.to(self.args.device)
        self.criterion = criterion

        param_groups = [
            {"params": self.model.gnn.parameters(), "lr": self.args.gcn_lr},
            {"params": self.model.encoder.parameters(), "lr": self.args.encoder_lr},
        ]

        self.optimizer = torch.optim.AdamW(param_groups, weight_decay=0.0015479)

    def train_epoch(self):
        self.model.train()
        e_loss = 0
        t = tqdm(self.args.train_data_loader, leave=False)
        for batch in t:
            t.set_description("Training for epoch")
            e_loss += self.train_batch(batch)
        return e_loss

    def train_batch(self, batch):
        self.optimizer.zero_grad()
        i_ids, a_mask, y_b, idx = batch
        i_ids, a_mask, y_b, idx = move(self.args.device, i_ids, a_mask, y_b, idx)

        y_hat = self.model(i_ids, a_mask, idx, self.args.graph_dataset.graph_data)
        if self.args.metric_name == ResultMetricName.PEARSON:
            loss = self.criterion(y_hat.reshape(-1), y_b.reshape(-1))
        else:
            loss = self.criterion(y_hat, y_b.reshape(-1))
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def evaluate(self, data_key: str):
        self.model.eval()
        dl = self.check_data_key(data_key)
        t = tqdm(dl, leave=False, desc=f"Evaluating {data_key}")
        with torch.no_grad():
            y_hat, y = [], []
            for i_ids, a_mask, y_b, _ in t:
                i_ids, a_mask = move(self.args.device, i_ids, a_mask)
                y_hat_b, _ = self.model.inference(i_ids, a_mask)
                y_hat.append(y_hat_b.to("cpu"))
                y.append(y_b)

            y = torch.cat(y, dim=0)
            y_hat = torch.cat(y_hat, dim=0)
            metric_value = self.args.evaluation_function(y, y_hat)

            del y, y_hat, dl, t
            torch.mps.empty_cache()
            return metric_value

    def check_data_key(self, data_key):
        if data_key == "train":
            return self.args.train_data_loader
        elif data_key == "validation":
            return self.args.val_data_loader
        else:
            raise ValueError(f"Invalid data_key: {data_key}")

    def update_cls(self):
        self.model.eval()

        with torch.no_grad():
            t = tqdm(self.args.train_data_loader_full, leave=False, desc="Updating CLS")
            for i_ids, a_mask, _, idx in t:
                i_ids, a_mask, idx = move(self.args.device, i_ids, a_mask, idx)
                _, encoder_CLS = self.model.encoder(i_ids, a_mask)
                self.args.graph_dataset.graph_data.x[idx.reshape(-1)] = encoder_CLS
