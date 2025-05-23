import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class MetricProcessor:
    def __init__(self, df: pd.DataFrame, metric_cols: list, nan_strategy="drop", scaler="standard", trim_config=None):
        self.df = df.copy()
        self.metric_cols = metric_cols
        self.nan_strategy = nan_strategy  # "drop", "fill_mean", "fill_zero"
        self.scaler = self._init_scaler(scaler)
        
        if trim_config:
            column = trim_config["column"]
            if column not in self.df.columns:
                print(f"Available columns: {self.df.columns}")
                raise ValueError(f"Column '{column}' specified in trim_config does not exist in the DataFrame.")
            
            old_len = len(self.df)
            self.df = self.df[self.df[column].between(trim_config["min"], trim_config["max"])]
            print(f"Trimmed {old_len - len(self.df)} rows based on {column} between {trim_config['min']} and {trim_config['max']}")
    
    def _init_scaler(self, scaler_name):
        if scaler_name == "standard":
            return StandardScaler()
        elif scaler_name == "minmax":
            return MinMaxScaler()
        else:
            raise ValueError(f"Unsupported scaler: {scaler_name}")

    def handle_nans(self):
        if self.nan_strategy == "drop":
            self.df = self.df.dropna(subset=self.metric_cols)
        elif self.nan_strategy == "fill_mean":
            for col in self.metric_cols:
                self.df[col] = self.df[col].fillna(self.df[col].mean())
        elif self.nan_strategy == "fill_zero":
            self.df[self.metric_cols] = self.df[self.metric_cols].fillna(0)
        elif self.nan_strategy == "fill_minus_one":
            for col in self.metric_cols:
                self.df[col] = self.df[col].fillna(-1)
        else:
            raise ValueError(f"Unknown NaN strategy: {self.nan_strategy}")
        return self

    def normalize(self):
        if self.df[self.metric_cols].isnull().values.any():
            raise ValueError("DataFrame contains NaN values. Please handle them before normalization.")
        
        # Normalize the data
        normalized_data = self.scaler.fit_transform(self.df[self.metric_cols])
        
        # Assign normalized values back to each column
        for i, col in enumerate(self.metric_cols):
            self.df[col] = normalized_data[:, i]
        
        return self
    
    def get_processed_df(self):
        return self.df
