from pathlib import Path

import numpy as np
import torch
from scipy.sparse import coo_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from torch_geometric.data import Data

from baseline import DATASET_CONFIGS
from baseline.dataset import GLUETextDataset


class GraphDataset:
    def __init__(self, text_dataset: GLUETextDataset, connection_threshold: float):
        self.text_dataset = text_dataset
        self.connection_threshold = connection_threshold

        folder_path = (
            Path("data/graph_data") / self.text_dataset.args.dataset_name.value
        )
        file_name = f"th_{self.connection_threshold}.pt"
        self.path = Path.joinpath(folder_path, file_name)
        self.graph_data = self.generate_graph_data()

    # def get_graph_data(self):
    #     if self.path.exists():
    #         return torch.load(self.path, weights_only=False)
    #         pass
    #     else:
    #         # create path
    #         self.path.parent.mkdir(parents=True, exist_ok=True)
    #         return self.generate_graph_data()

    def generate_graph_data(self):
        key1, key2 = DATASET_CONFIGS[self.text_dataset.args.dataset_name].keys
        if key2 is None:
            raw_texts = self.text_dataset.df[key1].tolist()
        else:
            text1 = self.text_dataset.df[key1].tolist()
            text2 = self.text_dataset.df[key2].tolist()
            raw_texts = [f"{t1} {t2}" for t1, t2 in zip(text1, text2)]

        vectorizer = TfidfVectorizer(stop_words="english")
        tfidf_matrix = vectorizer.fit_transform(raw_texts)
        sim_matrix = tfidf_matrix @ tfidf_matrix.T

        sim_matrix_coo = coo_matrix(sim_matrix)
        sim_matrix_coo.data[sim_matrix_coo.data < self.connection_threshold] = 0
        sim_matrix_coo.eliminate_zeros()

        rows_cols = np.array([sim_matrix_coo.row, sim_matrix_coo.col])

        edge_index = torch.tensor(rows_cols, dtype=torch.long)
        edge_attr = torch.tensor(sim_matrix_coo.data, dtype=torch.float32).view(-1)
        x = torch.zeros(len(raw_texts), 768)

        graph_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        # torch.save(graph_data, self.path)

        return graph_data
