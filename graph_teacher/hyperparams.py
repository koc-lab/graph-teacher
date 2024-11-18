from dataclasses import dataclass

from graph_teacher import GNNBackBone


@dataclass
class GNNHyperParams:
    backbone: GNNBackBone
    fan_mid: int
    dropout: float
    lr: float
    connection_threshold: float
    lmbd: float
