import wandb
from baseline.hyperparams import (
    EncoderHyperParams,
    EncoderModelBackbone,
    GLUEDatasetName,
    TrainDatasetPercentage,
)
from baseline.runner import single_run
from baseline.utils import set_seeds

# Define the sweep configuration template
sweep_config = {
    "name": "baseline-encoder-models",
    "method": "grid",
    "metric": {"name": "val_metric", "goal": "maximize"},
    "parameters": {
        "model_name": {
            "values": ["roberta-base", "bert-base-uncased", "distilbert-base-uncased"]
        },
        "dataset_name": {"values": ["qnli", "sst2"]},
        "train_percentage": {"values": [0.05, 0.1, 0.2, 0.5]},
        "max_length": {"values": [64]},
        "batch_size": {"values": [32]},
        "dropout": {"values": [0.1]},
        "lr": {"values": [1e-5]},
        "max_epochs": {"values": [15]},
        "patience": {"values": [5]},
        "seed": {"values": [42, 43, 44, 45, 46]},
    },
}


def main():
    wandb.init()
    run_name = f"{wandb.config.model_name}-{wandb.config.dataset_name}-{wandb.config.train_percentage}-{wandb.config.seed}"
    wandb.run.name = run_name
    set_seeds(wandb.config.seed)
    hp = EncoderHyperParams(
        encoder_backbone=EncoderModelBackbone(wandb.config.model_name),
        dataset_name=GLUEDatasetName(wandb.config.dataset_name),
        train_percentage=TrainDatasetPercentage(wandb.config.train_percentage),
        max_length=wandb.config.max_length,
        batch_size=wandb.config.batch_size,
        dropout=wandb.config.dropout,
        lr=wandb.config.lr,
        max_epochs=wandb.config.max_epochs,
        patience=wandb.config.patience,
    )
    _ = single_run(hp, wandb_flag=True)
    # save the best model to local with name best_val_res metric
    #! model_trainer.save_best_model(model_trainer.best_model, model_trainer.best_val_res)


# Start the sweep
sweep_id = wandb.sweep(sweep_config, project="graph-teacher-revision")
wandb.agent(sweep_id, function=main)
