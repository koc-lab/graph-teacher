from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

from baseline.hyperparams import EncoderModelBackbone


@dataclass
class EncoderClassifierArgs:
    model_checkpoint: EncoderModelBackbone
    n_class: int
    dropout: float


class EncoderClassifier(nn.Module):
    def __init__(self, args: EncoderClassifierArgs):
        super().__init__()
        self.args = args
        self.encoder = AutoModel.from_pretrained(args.model_checkpoint.value)
        self.linear = nn.Linear(768, args.n_class)
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, i_ids: torch.Tensor, a_mask: torch.Tensor):
        cls_embed = self.encoder(i_ids, a_mask)[0][:, 0]
        x = self.dropout(cls_embed)
        x = self.linear(x)
        if self.args.n_class != 1:
            x = F.log_softmax(x, dim=1)

        return x, cls_embed
