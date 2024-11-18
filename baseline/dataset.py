from dataclasses import dataclass

import torch
from datasets import load_dataset
from torch.utils.data import Dataset

from baseline import DATASET_CONFIGS
from baseline.hyperparams import GLUEDatasetName, TrainDatasetPercentage
from baseline.tokenizer import TokenizerWrapper


@dataclass
class GLUETextDatasetArgs:
    dataset_name: GLUEDatasetName
    split: str
    train_percentage: TrainDatasetPercentage


class GLUETextDataset(Dataset):
    def __init__(
        self,
        args: GLUETextDatasetArgs,
        tokenizer: TokenizerWrapper,
    ):
        self.args = args
        self.tokenizer = tokenizer
        self.df = self.load_df()
        self.indices = torch.tensor(self.df.index.values.tolist(), dtype=torch.long)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx) -> tuple[torch.Tensor]:
        sentence1, sentence2 = self.get_sentences(idx)
        y_b = self.df.iloc[idx]["label"]
        i_ids, a_mask = self.tokenizer(sentence1, sentence2)
        return i_ids, a_mask, y_b, self.indices[idx]

    def get_sentences(self, idx):
        key1, key2 = DATASET_CONFIGS[self.args.dataset_name].keys
        if key2 is None:
            sentence1 = self.df.iloc[idx][key1]
            sentence2 = None
        else:
            sentence1 = self.df.iloc[idx][key1]
            sentence2 = self.df.iloc[idx][key2]
        return sentence1, sentence2

    def load_df(self):
        if self.args.split == "train_full":
            df = load_dataset("glue", self.args.dataset_name.value)["train"].to_pandas()
        elif self.args.split == "train":
            df = load_dataset("glue", self.args.dataset_name.value)["train"].to_pandas()
            df = df.sample(frac=self.args.train_percentage.value, random_state=42)
        elif self.args.split == "validation":
            df = load_dataset("glue", self.args.dataset_name.value)[
                "validation"
            ].to_pandas()
        else:
            raise ValueError(f"Invalid split: {self.args.split}")

        return df
