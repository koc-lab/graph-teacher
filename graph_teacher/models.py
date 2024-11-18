from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import ChebConv, GCNConv, ResGatedGraphConv

from baseline.models import EncoderClassifier, EncoderClassifierArgs
from graph_teacher import GNNBackBone


@dataclass
class GNNArgs:
    backbone: GNNBackBone
    fan_in: int
    fan_mid: int
    fan_out: int
    dropout: float = 0.1


class GNN(nn.Module):
    def __init__(self, args: GNNArgs):
        super().__init__()

        self.args = args

        self.conv1 = self.get_conv_layer_1()
        self.ln1 = nn.LayerNorm(args.fan_mid)

        self.conv2 = self.get_conv_layer_2()
        self.ln2 = nn.LayerNorm(args.fan_out)
        self.dropout = nn.Dropout(args.dropout)

    def get_conv_layer_1(self):
        if self.args.backbone == GNNBackBone.ChebConv:
            return ChebConv(self.args.fan_in, self.args.fan_mid, K=2)
        elif self.args.backbone == GNNBackBone.ResGatedGraphConv:
            return ResGatedGraphConv(self.args.fan_in, self.args.fan_mid)
        elif self.args.backbone == GNNBackBone.GCNConv:
            return GCNConv(self.args.fan_in, self.args.fan_mid)

    def get_conv_layer_2(self):
        if self.args.fan_out == 1:
            return nn.Linear(self.args.fan_mid, self.args.fan_out)
        else:
            if self.args.backbone == GNNBackBone.ChebConv:
                return ChebConv(self.args.fan_mid, self.args.fan_out, K=2)
            elif self.args.backbone == GNNBackBone.ResGatedGraphConv:
                return ResGatedGraphConv(self.args.fan_mid, self.args.fan_out)
            elif self.args.backbone == GNNBackBone.GCNConv:
                return GCNConv(self.args.fan_mid, self.args.fan_out)

    def forward(self, g: Data):
        x = F.elu(self.ln1(self.conv1(g.x, g.edge_index)))
        x = self.dropout(x)

        if self.args.fan_out != 1:
            x = F.elu(self.ln2(self.conv2(x, g.edge_index)))
            x = F.log_softmax(x, dim=1)
        else:
            x = self.conv2(x)

        return x


@dataclass
class InterpolationModelArgs:
    gnn_args: GNNArgs
    encoder_args: EncoderClassifierArgs
    lmbd: float


class InterpolationModel(nn.Module):
    def __init__(self, args: InterpolationModelArgs):
        super().__init__()
        self.args = args
        self.gnn = GNN(args=args.gnn_args)
        self.encoder = EncoderClassifier(args=args.encoder_args)

    def forward(self, i_ids, a_mask, idx: torch.Tensor, g: Data) -> torch.Tensor:
        bert_pred, _ = self.encoder(i_ids, a_mask)
        gnn_pred = self.gnn(g)[idx.reshape(-1)]
        pred = self.interpolate(gnn_pred, bert_pred, self.args.lmbd)
        return pred

    def interpolate(self, x, y, lmbd):
        return lmbd * x + (1 - lmbd) * y

    def inference(self, i_ids, a_mask):
        return self.encoder(i_ids, a_mask)
