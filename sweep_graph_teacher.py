# %%


import wandb
from baseline.dataset import GLUEDatasetName, TrainDatasetPercentage
from baseline.hyperparams import EncoderHyperParams, EncoderModelBackbone
from baseline.utils import set_seeds
from graph_teacher.hyperparams import GNNBackBone, GNNHyperParams
from graph_teacher.runner import run_single_configuration

# Set the logging level to ERROR to suppress warnings


dataset_name = GLUEDatasetName.SST2
gnn_backbone = GNNBackBone.GCNConv
encoder_backbone = EncoderModelBackbone.BERT_BASE_UNCASED
train_percentage = TrainDatasetPercentage.FIFTY


sweep_name = f"36_{dataset_name.value}_{gnn_backbone.name}_{encoder_backbone.value}_{train_percentage.value}_new"

# %%
sweep_config = {
    "name": sweep_name,
    "method": "bayes",
    "metric": {"name": "val_metric", "goal": "maximize"},
    "parameters": {
        # Encoder Parameters
        "encoder_lr": {"min": 5e-5, "max": 5e-4, "distribution": "uniform"},
        "encoder_dropout": {"max": 0.7, "min": 0.1, "distribution": "uniform"},
        # GNN Parameters
        "fan_mid": {"values": [64, 128]},
        "gcn_dropout": {"max": 0.7, "min": 0.1, "distribution": "uniform"},
        "gcn_lr": {"min": 5e-3, "max": 5e-2, "distribution": "uniform"},
        "connection_threshold": {"values": [0.3]},
        "lmbd": {"min": 0.3, "max": 0.9, "distribution": "uniform"},
        "sweep_name": {"value": sweep_name},
        # Pipeline Parameters
    },
}


def main():
    global encoder_backbone, dataset_name, train_percentage, gnn_backbone
    wandb.init(tags=sweep_name)
    set_seeds(42)
    encoder_hp = EncoderHyperParams(
        encoder_backbone=encoder_backbone,
        dataset_name=dataset_name,
        train_percentage=train_percentage,
        max_length=64,
        batch_size=32,
        dropout=wandb.config.encoder_dropout,
        lr=wandb.config.encoder_lr,
    )

    gnn_hp = GNNHyperParams(
        backbone=gnn_backbone,
        fan_mid=wandb.config.fan_mid,
        dropout=wandb.config.gcn_dropout,
        lr=wandb.config.gcn_lr,
        connection_threshold=wandb.config.connection_threshold,
        lmbd=wandb.config.lmbd,
    )

    run_single_configuration(
        encoder_hp, gnn_hp, max_epochs=15, patience=2, wandb_log_flag=True
    )


sweep_id = wandb.sweep(sweep_config, project="graph-teacher-revision-ours")
wandb.agent(sweep_id, function=main)
