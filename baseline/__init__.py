import logging
from enum import Enum

import torch
from torch.nn import MSELoss, NLLLoss

from baseline.utils import eval_func, get_device

logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)


def get_devicee():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


DEVICE = get_devicee()
print(f"Device: {DEVICE}")


class EncoderModelBackbone(Enum):
    ROBERTA_BASE = "roberta-base"
    BERT_BASE_UNCASED = "bert-base-uncased"
    DISTILBERT_BASE_UNCASED = "distilbert-base-uncased"


class GLUEDatasetName(Enum):
    QNLI = "qnli"
    MRPC = "mrpc"
    STSB = "stsb"
    COLA = "cola"
    RTE = "rte"
    WNLI = "wnli"
    SST2 = "sst2"


class TrainDatasetPercentage(Enum):
    FIVE = 0.05
    TEN = 0.1
    TWENTY = 0.2
    FIFTY = 0.5
    FULL = 1.0


class ResultMetricName(Enum):
    ACC = "acc"
    F1 = "f1"
    MCORR = "mcorr"
    PEARSON = "pearson"


class GLUEConfig:
    def __init__(
        self,
        keys: str,
        criterion,
        n_class: int,
        metric_name: ResultMetricName,
    ):
        self.keys = keys
        self.criterion = criterion
        self.n_class = n_class
        self.metric_name = metric_name
        self.evaluation_function = eval_func(self.metric_name.value)
        self.device = get_device()


DATASET_CONFIGS = {
    GLUEDatasetName.COLA: GLUEConfig(
        ("sentence", None), NLLLoss(), 2, ResultMetricName.MCORR
    ),
    GLUEDatasetName.MRPC: GLUEConfig(
        ("sentence1", "sentence2"), NLLLoss(), 2, ResultMetricName.F1
    ),
    GLUEDatasetName.STSB: GLUEConfig(
        ("sentence1", "sentence2"), MSELoss(), 1, ResultMetricName.PEARSON
    ),
    GLUEDatasetName.QNLI: GLUEConfig(
        ("question", "sentence"), NLLLoss(), 2, ResultMetricName.ACC
    ),
    GLUEDatasetName.RTE: GLUEConfig(
        ("sentence1", "sentence2"), NLLLoss(), 2, ResultMetricName.ACC
    ),
    GLUEDatasetName.WNLI: GLUEConfig(
        ("sentence1", "sentence2"), NLLLoss(), 2, ResultMetricName.ACC
    ),
    GLUEDatasetName.SST2: GLUEConfig(
        ("sentence", None), NLLLoss(), 2, ResultMetricName.ACC
    ),
}
