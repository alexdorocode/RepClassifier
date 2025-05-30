import hydra  # type: ignore
from omegaconf import DictConfig  # type: ignore
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score

from project_root.dataset.dataset_config import DatasetConfigReader
from project_root.dataset.dataset_handler import DatasetHandler

# Add the project root directory to PYTHONPATH
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
if project_root not in sys.path:
    sys.path.append(project_root)

@hydra.main(config_path="../config", config_name="config_experiment", version_base="1.3")
def test_sequence_loader(cfg: DictConfig):
    print("🔍 Initializing DatasetConfigReader and DatasetHandler...")
    config_reader = DatasetConfigReader(cfg)
    handler = DatasetHandler(config_reader)

    # Load ClassifierDataset
    print("📦 Loading ClassifierDataset...")
    classifier_dataset = handler.load_classifier_dataset()  # Add optional arguments as needed

    # Split dataset into train/test
    dataset_size = len(classifier_dataset)
    test_size = int(0.2 * dataset_size)
    train_size = dataset_size - test_size
    train_dataset, test_dataset = random_split(classifier_dataset, [train_size, test_size])

    # Wrap in DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Define simple logistic regression model
    input_dim = classifier_dataset.features.shape[1]
    num_classes = len(torch.unique(classifier_dataset.labels))
    print(f"📊 Input dim: {input_dim}, Number of classes: {num_classes}")

    class LogisticRegression(nn.Module):
        def __init__(self, input_dim, num_classes):
            super(LogisticRegression, self).__init__()
            self.linear = nn.Linear(input_dim, num_classes)

        def forward(self, x):
            return self.linear(x)

    model = LogisticRegression(input_dim, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train
    num_epochs = 50
    print("🚀 Training Logistic Regression Model...")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for features, labels in train_loader:
            outputs = model(features)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss:.4f}")

    # Evaluate on test set
    print("🔍 Evaluating on test set...")
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for features, labels in test_loader:
            outputs = model(features)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    acc = accuracy_score(all_labels, all_preds)
    print(f"🎯 Logistic Regression Test Accuracy: {acc:.2%}")

if __name__ == "__main__":
    test_sequence_loader()
