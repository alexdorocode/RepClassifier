import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class MetricProcessor:
    """
    Processes and normalizes metric columns in a DataFrame.
    Handles NaN strategies, scaling, and optional row trimming based on config.

    :param df: Input pandas DataFrame.
    :param metric_cols: List of metric column names to process.
    :param nan_strategy: Strategy for handling NaNs ("drop", "fill_mean", "fill_zero", "fill_minus_one").
    :param scaler: Scaling method ("standard" or "minmax").
    :param trim_config: Optional list of dicts for row trimming (each with 'column', 'min', 'max').
    """

    def __init__(self, df: pd.DataFrame, metric_cols: list, nan_strategy="drop", scaler="standard", trim_config=None):
        """
        Initialize the MetricProcessor.

        :param df: Input pandas DataFrame.
        :param metric_cols: List of metric column names to process.
        :param nan_strategy: Strategy for handling NaNs.
        :param scaler: Scaling method.
        :param trim_config: Optional list of dicts for row trimming.
        """
        self.df = df.copy()
        self.metric_cols = metric_cols
        self.nan_strategy = nan_strategy  # "drop", "fill_mean", "fill_zero", "fill_minus_one"
        self.scaler = self._init_scaler(scaler)
        
        print(f"All columns: {self.df.columns.tolist()}")

        if trim_config:
            old_len = len(self.df)
            # Collect all rows to be trimmed
            trimmed_mask = pd.Series(True, index=self.df.index)  # Start with all rows included
            for config in trim_config:
                print(f"Applying trim config: {config}")
                self._valid_trim_config(config)
                column = config["column"]
                trimmed_mask &= self.df[column].between(config["min"], config["max"])
            
            # Identify rows to be trimmed (invert the mask)
            rows_to_trim = ~trimmed_mask
            trimmed_df = self.df[rows_to_trim]
            
            # Print trimmed organisms divided by 'class'
            print("Trimmed rows by organism and class:")
            print(trimmed_df.groupby(['organism', 'class']).size())
            
            # Keep only the rows that are not trimmed
            self.df = self.df[trimmed_mask]
            print(f"Trimmed {old_len - len(self.df)} rows based on trim_config.")
            
    def _init_scaler(self, scaler_name):
        """
        Initialize the scaler based on the scaler name.

        :param scaler_name: "standard" or "minmax"
        :return: Scaler instance
        """
        if scaler_name == "standard":
            return StandardScaler()
        elif scaler_name == "minmax":
            return MinMaxScaler()
        else:
            raise ValueError(f"Unsupported scaler: {scaler_name}")

    def handle_nans(self):
        """
        Handle NaN values in the metric columns according to the chosen strategy.

        :return: self
        """
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
        """
        Normalize the metric columns using the selected scaler.

        :return: self
        :raises ValueError: If NaNs remain in the metric columns.
        """
        if self.df[self.metric_cols].isnull().values.any():
            raise ValueError("DataFrame contains NaN values. Please handle them before normalization.")
        
        # Normalize the data
        normalized_data = self.scaler.fit_transform(self.df[self.metric_cols])
        
        # Assign normalized values back to each column
        for i, col in enumerate(self.metric_cols):
            self.df[col] = normalized_data[:, i]
        
        return self
    
    def get_processed_df(self):
        """
        Returns the processed DataFrame.

        :return: pandas DataFrame
        """
        return self.df
    
    def _valid_trim_config(self, config):
        """
        Validates a trim configuration dictionary.

        :param config: Dict with 'column', 'min', and 'max' keys.
        :raises ValueError: If config is invalid.
        """
        if "column" not in config or "min" not in config or "max" not in config:
            raise ValueError("Each trim config must contain 'column', 'min', and 'max' keys.")
        if not isinstance(config["min"], (int, float)) or not isinstance(config["max"], (int, float)):
            raise ValueError("'min' and 'max' values must be numeric.")
        if config["min"] >= config["max"]:
            raise ValueError("'min' value must be less than 'max' value.")