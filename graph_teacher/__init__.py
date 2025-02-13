import logging
from enum import Enum

from dotenv import load_dotenv
import torch
from torch_geometric.nn import ChebConv, GCNConv, ResGatedGraphConv

load_dotenv()
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)


# check devices and set the device
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


DEVICE = get_device()
print(f"Device: {DEVICE}")


class GNNBackBone(Enum):
    ChebConv = ChebConv
    ResGatedGraphConv = ResGatedGraphConv
    GCNConv = GCNConv
