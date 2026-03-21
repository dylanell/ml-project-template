from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset


from ..utils import get_storage_options


class TabularDataset(Dataset):
    def __init__(self, csv_path: str, target_col: str):
        df = pd.read_csv(csv_path, storage_options=get_storage_options(csv_path))

        self.feature_names = [col for col in df.columns if col != target_col]
        self.X = df[self.feature_names].to_numpy().astype(np.float32)

        label_encoder = LabelEncoder()
        self.y = label_encoder.fit_transform(df[target_col])
        self.class_names = list(label_encoder.classes_)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, index):
        return {"features": self.X[index], "targets": self.y[index]}
