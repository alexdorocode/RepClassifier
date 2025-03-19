import torch
from torch.utils.data import DataLoader

class DataLoaderFactory:
    """
    Factory class for creating PyTorch DataLoaders from ProteinDataset.
    """

    @staticmethod
    def create_dataloader(dataset, batch_size=32, shuffle=True, num_workers=0):
        """
        Creates a PyTorch DataLoader from a ProteinDataset instance.

        Args:
            dataset (ProteinDataset): The dataset object.
            batch_size (int): Batch size for training.
            shuffle (bool): Whether to shuffle the data.
            num_workers (int): Number of workers for data loading.

        Returns:
            DataLoader: PyTorch DataLoader instance.
        """
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
