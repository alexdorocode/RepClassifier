import torch
import numpy as np
from torch.utils.data import Dataset

class ClassifierDataset(Dataset):
    def __init__(self, processed_df, feature_cols=None, label_col="label", balance_col=None, production=False):
        """
        Args:
            processed_df: pandas DataFrame with all processed features and labels.
            feature_cols: list of column names to include as features (default: all columns except label_col).
            label_col: name of the label column (default: 'label').
            production: if True, assumes no labels (for classifying new data).
        """
        self.production = production
        self.df = processed_df
        self.label_col = label_col
        self.balance_col = balance_col
        self.production = production
        
        # Determine excluded columns
        excluded = []
        if label_col in processed_df.columns and not production:
            excluded.append(label_col)
        if balance_col in processed_df.columns:
            excluded.append(balance_col)
        
        # Determine feature columns
        if feature_cols is None:
            self.feature_cols = [col for col in processed_df.columns if col not in excluded]
        else:
            self.feature_cols = feature_cols
        
        print(f"Initializing ClassifierDataset with features: {self.feature_cols}, label: {self.label_col}, balance_col: {self.balance_col}")
        
        # Prepare features and labels
        self.features = self._prepare_features()
        if not production:
            self.labels = self._prepare_labels()
        else:
            self.labels = None
        
        # Optional: balance values for stratification
        if balance_col and balance_col in processed_df.columns:
            self.balance_values = processed_df[balance_col].values
        else:
            self.balance_values = None

    def _prepare_features(self):
        feature_list = []
        for col in self.feature_cols:
            sample_value = self.df[col].iloc[0]
            if isinstance(sample_value, (list, np.ndarray)):
                arr_list = [np.array(x) if x is not None else np.zeros_like(sample_value) for x in self.df[col]]
            
            elif isinstance(sample_value, (list, np.ndarray)):
                # Ensure all entries are arrays of the same size
                arr_list = [np.array(x) if x is not None else np.zeros_like(sample_value) for x in self.df[col]]
                stacked_array = np.stack(arr_list)
                feature_col_tensor = torch.from_numpy(stacked_array).float()
            
            else:
                # Scalar (int, float, etc.)
                feature_col_tensor = torch.tensor(self.df[col].fillna(0).astype(float).values, dtype=torch.float32).unsqueeze(1)
            
            feature_list.append(feature_col_tensor)
        
        # Concatenate along last dimension
        features = torch.cat(feature_list, dim=1)
        return features

    def _prepare_labels(self):
        labels = self.df[self.label_col].values
        return torch.tensor(labels, dtype=torch.long)  # Or torch.float32 if needed

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if self.production:
            return self.features[idx]
        else:
            return self.features[idx], self.labels[idx]
    
    def get_X_y(self):
        """
        Returns the features and labels as numpy arrays.
        Useful for compatibility with sklearn or other libraries.
        """
        if self.production:
            return self.features.numpy(), None
        else:
            return self.features.numpy(), self.labels.numpy()
    
