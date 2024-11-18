from pathlib import Path

import torch
from torch.utils.data import DataLoader

from baseline.dataset import GLUETextDataset, GLUETextDatasetArgs
from baseline.tokenizer import TokenizerWrapper, TokenizerWrapperArgs


def create_dataloader_directory(
    dataset_args: GLUETextDatasetArgs, tokenizer_args: TokenizerWrapperArgs
) -> Path:

    dataset_name = dataset_args.dataset_name.value
    split = dataset_args.split
    model_checkpoint = tokenizer_args.model_checkpoint.value
    dir_path = Path("data/dataloaders", dataset_name, model_checkpoint, split)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def create_or_load_data_loader(
    dataset_args: GLUETextDatasetArgs,
    batch_size: int,
    tokenizer_wrapper: TokenizerWrapper,
) -> DataLoader:

    dir_path = create_dataloader_directory(dataset_args, tokenizer_wrapper.args)
    max_length = tokenizer_wrapper.args.max_length
    train_percentage = dataset_args.train_percentage

    if dataset_args.split == "train":
        dl_file_name = f"train_percentage_{train_percentage}_batch_size_{batch_size}_max_length_{max_length}.pt"
    else:
        dl_file_name = f"batch_size_{batch_size}_max_length_{max_length}.pt"

    dl_path = dir_path / dl_file_name

    if dl_path.exists():
        data_loader = torch.load(dl_path, weights_only=False)
        # print(f"Dataloader loaded from {dl_path}")
    else:
        dataset = GLUETextDataset(args=dataset_args, tokenizer=tokenizer_wrapper)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        torch.save(data_loader, dl_path)

    return data_loader
