# %%
from baseline import DATASET_CONFIGS
from baseline.hyperparams import (
    EncoderModelBackbone,
    GLUEDatasetName,
    EncoderHyperParams,
    TrainDatasetPercentage,
)
from baseline.dataset import get_data_loader_dict
from baseline.models import EncoderClassifier
from baseline.tokenizer import TokenizerWrapper
from baseline.trainer import Trainer

hp = EncoderHyperParams(
    dataset_name=GLUEDatasetName.SST2,
    encoder_backbone=EncoderModelBackbone.ROBERTA_BASE,
    train_percentage=TrainDatasetPercentage.TEN,
    max_length=128,
    batch_size=32,
    dropout=0.1,
    lr=1e-05,
    max_epochs=3,
    patience=3,
)


def single_run(hp: EncoderHyperParams, wandb_flag: bool):
    fix_config = DATASET_CONFIGS[hp.dataset_name]
    tokenizer = TokenizerWrapper(
        model_name=hp.encoder_backbone, max_length=hp.max_length
    )
    data_loader_dict = get_data_loader_dict(
        dataset_name=hp.dataset_name,
        train_percentage=hp.train_percentage,
        batch_size=hp.batch_size,
        tokenizer=tokenizer,
    )

    model = EncoderClassifier(
        model_name=hp.encoder_backbone,
        n_class=fix_config.n_class,
        dropout=hp.dropout,
    )

    model_trainer = ModelTrainer(
        data_loader_dict=data_loader_dict,
        criterion=fix_config.criterion,
        model=model,
        lr=hp.lr,
        metric_name=fix_config.metric_name,
        evaluation_function=fix_config.evaluation_function,
        device=fix_config.device,
    )

    model_trainer.pipeline(
        max_epochs=hp.max_epochs, patience=hp.patience, wandb_flag=wandb_flag
    )
    return model_trainer


def test_best_model(saved_best_model, saved_best_val_res, model_trainer: ModelTrainer):

    model_trainer.model = saved_best_model

    found_train_metric_val = model_trainer.evaluate("train")
    found_validation_metric_val = model_trainer.evaluate("validation")

    print(f"Train {model_trainer.metric_name}: {found_train_metric_val}")
    print(f"Validation {model_trainer.metric_name}: {found_validation_metric_val}")

    print(f"Best Validation {model_trainer.metric_name}: {saved_best_val_res}")


# %%
