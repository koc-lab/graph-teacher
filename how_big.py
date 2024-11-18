# %%
from baseline import EncoderModelBackbone, GLUEDatasetName, TrainDatasetPercentage
from baseline.dataloader import create_or_load_data_loader
from baseline.dataset import GLUETextDatasetArgs
from baseline.hyperparams import EncoderHyperParams
from baseline.tokenizer import TokenizerWrapper, TokenizerWrapperArgs
from graph_teacher.dataset import GraphDataset
from graph_teacher.hyperparams import GNNBackBone, GNNHyperParams


def get_graph_dataset(dataset_name: GLUEDatasetName, connection_threshold: float):
    encoder_hp = EncoderHyperParams(
        encoder_backbone=EncoderModelBackbone.DISTILBERT_BASE_UNCASED,
        dataset_name=dataset_name,
        train_percentage=TrainDatasetPercentage.FIVE,
        max_length=128,
        batch_size=32,
        dropout=0.17057,
        lr=0.000024225,
    )

    gnn_hp = GNNHyperParams(
        backbone=GNNBackBone.GCNConv,
        fan_mid=256,
        dropout=0.344,
        lr=0.000024225,
        connection_threshold=0.0,
        lmbd=0.5978,
    )

    tokenizer_wrapper = TokenizerWrapper(
        args=TokenizerWrapperArgs(
            model_checkpoint=encoder_hp.encoder_backbone,
            max_length=encoder_hp.max_length,
        )
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

    graph_dataset = GraphDataset(
        text_dataset=train_dl_full.dataset,
        connection_threshold=gnn_hp.connection_threshold,
    )
    return graph_dataset


dataset_names = [
    GLUEDatasetName.MRPC,
    GLUEDatasetName.SST2,
    GLUEDatasetName.COLA,
    GLUEDatasetName.RTE,
    GLUEDatasetName.WNLI,
    GLUEDatasetName.SST2,
]


def percentage(threshold, edge_attr):
    return len(edge_attr[edge_attr > threshold])


for dataset_name in dataset_names:
    graph_dataset = get_graph_dataset(dataset_name, 0.0)
    edge_attr = graph_dataset.graph_data.edge_attr
    edge_index = graph_dataset.graph_data.edge_index

    non_zeros = len(edge_attr[edge_attr != 0])
    total = graph_dataset.graph_data.x.shape[0] * (
        graph_dataset.graph_data.x.shape[0] - 1
    )
    zeros_percentage = (total - non_zeros) / total

    print(f"Dataset: {dataset_name.value.upper()}")
    print(f"Total number of possible connections: {total:,}")
    print(f"Number of non-zeros: {non_zeros:,}")
    print(f"Percentage of zeros in graph: {100*zeros_percentage:.2f}%")
    print(f"Number of edge_attr greater than 0.1: {percentage(0.1, edge_attr):,}")
    print(f"Number of edge_attr greater than 0.2: {percentage(0.2, edge_attr):,}")
    print(f"Number of edge_attr greater than 0.3: {percentage(0.3, edge_attr):,}")
    print(
        f"Percentage of nodes connected when th 0.1: {(percentage(0.1, edge_attr)/total) * 100:.2f} %"
    )
    print("\n")


# %%
percentage_of_zeros = (graph_dataset.graph_data.edge_attr == 0).sum().item() / len(
    graph_dataset.graph_data.edge_attr
)
print(f"Percentage of zeros in edge_attr: {percentage_of_zeros:.2f}")
# %%
