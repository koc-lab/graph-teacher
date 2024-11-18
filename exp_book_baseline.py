# %%
from baseline import (
    DATASET_CONFIGS,
    DEVICE,
    EncoderModelBackbone,
    GLUEDatasetName,
    TrainDatasetPercentage,
)
from baseline.dataloader import create_or_load_data_loader
from baseline.dataset import GLUETextDatasetArgs
from baseline.hyperparams import EncoderHyperParams
from baseline.models import EncoderClassifier, EncoderClassifierArgs
from baseline.pipeline import Pipeline
from baseline.tokenizer import TokenizerWrapper, TokenizerWrapperArgs
from baseline.trainer import Trainer, TrainerArgs
from baseline.utils import set_seeds

set_seeds(42)


encoder_hp = EncoderHyperParams(
    encoder_backbone=EncoderModelBackbone.DISTILBERT_BASE_UNCASED,
    dataset_name=GLUEDatasetName.MRPC,
    train_percentage=TrainDatasetPercentage.FIVE,
    max_length=128,
    batch_size=32,
    dropout=0.17057,
    lr=0.000024225,
)

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


val_dl = create_or_load_data_loader(
    dataset_args=GLUETextDatasetArgs(
        dataset_name=encoder_hp.dataset_name,
        split="validation",
        train_percentage=None,
    ),
    batch_size=encoder_hp.batch_size,
    tokenizer_wrapper=tokenizer_wrapper,
)

encoder_args = EncoderClassifierArgs(
    model_checkpoint=encoder_hp.encoder_backbone,
    n_class=FIX_CONFIG.n_class,
    dropout=encoder_hp.dropout,
)


model = EncoderClassifier(encoder_args)


trainer_args = TrainerArgs(
    train_data_loader=train_dl,
    val_data_loader=val_dl,
    metric_name=FIX_CONFIG.metric_name,
    evaluation_function=FIX_CONFIG.evaluation_function,
    lr=encoder_hp.lr,
    device=DEVICE,
)

trainer = Trainer(model=model, criterion=FIX_CONFIG.criterion, args=trainer_args)
pipeline = Pipeline(
    trainer=trainer,
    max_epochs=15,
    patience=5,
    wandb_log_flag=False,
)
pipeline.run_single_configuration()
# %%
