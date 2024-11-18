from baseline import DATASET_CONFIGS, TrainDatasetPercentage
from baseline.dataloader import create_or_load_data_loader
from baseline.dataset import GLUETextDatasetArgs
from baseline.hyperparams import EncoderHyperParams
from baseline.models import EncoderClassifierArgs
from baseline.tokenizer import TokenizerWrapper, TokenizerWrapperArgs
from graph_teacher import DEVICE
from graph_teacher.dataset import GraphDataset
from graph_teacher.hyperparams import GNNHyperParams
from graph_teacher.models import GNNArgs, InterpolationModel, InterpolationModelArgs
from graph_teacher.pipeline import Pipeline
from graph_teacher.trainer import Trainer, TrainerArgs


def run_single_configuration(
    encoder_hp: EncoderHyperParams,
    gnn_hp: GNNHyperParams,
    max_epochs,
    patience,
    wandb_log_flag,
):
    FIX_CONFIG = DATASET_CONFIGS[encoder_hp.dataset_name]

    tokenizer_wrapper = TokenizerWrapper(
        args=TokenizerWrapperArgs(
            model_checkpoint=encoder_hp.encoder_backbone,
            max_length=encoder_hp.max_length,
        )
    )

    train_dl = create_or_load_data_loader(
        dataset_args=GLUETextDatasetArgs(
            dataset_name=encoder_hp.dataset_name,
            split="train",
            train_percentage=encoder_hp.train_percentage,
        ),
        batch_size=encoder_hp.batch_size,
        tokenizer_wrapper=tokenizer_wrapper,
    )

    train_dl_full = create_or_load_data_loader(
        dataset_args=GLUETextDatasetArgs(
            dataset_name=encoder_hp.dataset_name,
            split="train_full",
            train_percentage=TrainDatasetPercentage.FULL,
        ),
        batch_size=encoder_hp.batch_size,
        tokenizer_wrapper=tokenizer_wrapper,
    )

    val_dl = create_or_load_data_loader(
        dataset_args=GLUETextDatasetArgs(
            dataset_name=encoder_hp.dataset_name,
            split="validation",
            train_percentage=None,
        ),
        batch_size=encoder_hp.batch_size,
        tokenizer_wrapper=tokenizer_wrapper,
    )

    graph_dataset = GraphDataset(
        text_dataset=train_dl_full.dataset,
        connection_threshold=gnn_hp.connection_threshold,
    )

    gnn_args = GNNArgs(
        backbone=gnn_hp.backbone,
        fan_in=768,
        fan_mid=gnn_hp.fan_mid,
        fan_out=FIX_CONFIG.n_class,
        dropout=gnn_hp.dropout,
    )

    encoder_args = EncoderClassifierArgs(
        model_checkpoint=encoder_hp.encoder_backbone,
        n_class=FIX_CONFIG.n_class,
        dropout=encoder_hp.dropout,
    )

    model = InterpolationModel(
        args=InterpolationModelArgs(
            gnn_args=gnn_args,
            encoder_args=encoder_args,
            lmbd=gnn_hp.lmbd,
        )
    )
    trainer_args = TrainerArgs(
        train_data_loader_full=train_dl_full,
        train_data_loader=train_dl,
        val_data_loader=val_dl,
        graph_dataset=graph_dataset,
        metric_name=FIX_CONFIG.metric_name,
        evaluation_function=FIX_CONFIG.evaluation_function,
        gcn_lr=gnn_hp.lr,
        encoder_lr=encoder_hp.lr,
        device=DEVICE,
    )

    trainer = Trainer(model=model, criterion=FIX_CONFIG.criterion, args=trainer_args)
    pipeline = Pipeline(
        trainer=trainer,
        max_epochs=max_epochs,
        patience=patience,
        wandb_log_flag=wandb_log_flag,
    )
    pipeline.run_single_configuration()


# %%
