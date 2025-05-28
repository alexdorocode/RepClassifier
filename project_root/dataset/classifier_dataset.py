import torch
from torch.utils.data import Dataset

class ClassifierDataset(Dataset):
    def __init__(self, processed_dataset, feature_cols=None, label_col="class"):
        """
        Initializes the ExperimentalDataset from a ProcessedDataset.
        
        Args:
            processed_dataset: ProcessedDataset instance containing preprocessed data.
            feature_cols: List of feature column names to include (default: all columns except label_col).
            label_col: Name of the label column.
        """
        self.data = processed_dataset.data  # Assumes ProcessedDataset has a .data attribute (pandas DataFrame)
        self.label_col = label_col
        
        if feature_cols is None:
            # Automatically select feature columns (all except label)
            self.feature_cols = [col for col in self.data.columns if col != label_col]
        else:
            self.feature_cols = feature_cols

        # Extract features and labels
        self.features = self._prepare_features()
        self.labels = self._prepare_labels()

    def _prepare_features(self):
        """
        Prepares and returns a torch tensor of concatenated feature vectors.
        Handles both flat features and list-like embedding columns.
        """
        feature_list = []
        for col in self.feature_cols:
            if isinstance(self.data.iloc[0][col], torch.Tensor):
                # If the column contains tensors, stack them directly
                feature_col_tensor = torch.stack(self.data[col].to_list())
            elif isinstance(self.data.iloc[0][col], list):
                # Convert list-like to tensor
                feature_col_tensor = torch.tensor(self.data[col].to_list(), dtype=torch.float32)
            else:
                # Scalar numeric column
                feature_col_tensor = torch.tensor(self.data[col].values, dtype=torch.float32).unsqueeze(1)
            feature_list.append(feature_col_tensor)

        # Concatenate all feature tensors along the last dimension
        features = torch.cat(feature_list, dim=1)
        return features

    def _prepare_labels(self):
        """
        Prepares and returns a torch tensor of labels.
        """
        labels = self.data[self.label_col].values
        return torch.tensor(labels, dtype=torch.long)  # Assumes labels are integer classes

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
