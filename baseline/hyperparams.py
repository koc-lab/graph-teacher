from dataclasses import dataclass

from baseline import EncoderModelBackbone, GLUEDatasetName, TrainDatasetPercentage


@dataclass
class EncoderHyperParams:
    encoder_backbone: EncoderModelBackbone
    dataset_name: GLUEDatasetName
    train_percentage: TrainDatasetPercentage
    max_length: int
    batch_size: int
    dropout: float
    lr: float

    def __post_init__(self):
        if not isinstance(self.encoder_backbone, EncoderModelBackbone):
            raise ValueError(
                f"model_name must be an instance of ModelName Enum, got {type(self.encoder_backbone)}"
            )
        if not isinstance(self.dataset_name, GLUEDatasetName):
            raise ValueError(
                f"dataset_name must be an instance of DatasetName Enum, got {type(self.dataset_name)}"
            )

        if not isinstance(self.train_percentage, TrainDatasetPercentage):
            raise ValueError(
                f"train_percentage must be an instance of TrainPercentage Enum, got {type(self.train_percentage)}"
            )
