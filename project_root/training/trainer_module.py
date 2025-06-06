# project_root/models/trainer_module.py

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from collections import Counter
import numpy as np

class TrainerModule:
    def __init__(
        self,
        model,
        model_type,
        device='cpu',
        cv_folds=5,
        tracker=None,
        cross_val_balance=None,
        random_seed=42,
        # MLP-specific (optional)
        learning_rate=None,
        num_epochs=None,
        optimizer_name=None,
        criterion_name=None,
        batch_size=None,
        early_stopping_patience=None,
    ):
        """
        Initializes a training wrapper for both PyTorch and sklearn-style models.

        Args:
            model: Initialized model (PyTorch or sklearn/XGB/LGBM)
            model_type: str, e.g., 'mlp', 'logistic_regression', 'svm', etc.
            device: str, 'cpu' or 'cuda' (used for PyTorch)
            cv_folds: int, number of folds for cross-validation
            tracker: optional TrackerModule instance
            cross_val_balance: str, optional stratification feature (e.g., 'organism')
            random_seed: int, random state for reproducibility

            # Only required for PyTorch-based models like MLP
            learning_rate: float, optimizer learning rate
            num_epochs: int, number of epochs
            optimizer_name: str, e.g., 'Adam', 'SGD'
            criterion_name: str, e.g., 'CrossEntropyLoss', 'MSELoss'
            batch_size: int, training batch size
            early_stopping_patience: int, early stopping patience
        """
        self.model = model
        self.model_type = model_type
        self.device = device
        self.cv_folds = cv_folds
        self.tracker = tracker
        self.cross_val_balance = cross_val_balance
        self.random_seed = random_seed

        # Set MLP-specific training hyperparameters only when applicable
        if self.model_type == "mlp":
            self.learning_rate = learning_rate or 0.001
            self.num_epochs = num_epochs or 10
            self.optimizer_name = optimizer_name or "Adam"
            self.criterion_name = criterion_name or "CrossEntropyLoss"
            self.batch_size = batch_size or 32
            self.early_stopping_patience = early_stopping_patience

    def cross_validate(self, X, y, balance_df=None, debug=False):
        print(f"🔄 Starting {self.cv_folds}-fold cross-validation...")

        if self.cross_val_balance and balance_df is not None:
            stratify_labels = np.array([f"{label}_{balance}" for label, balance in zip(y, balance_df)])
        else:
            stratify_labels = y

        print(f"Using random seed: {self.random_seed}")
        skf = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_seed)

        fold_metrics = []
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, stratify_labels)):
            print(f"\n🔍 Fold {fold+1}/{self.cv_folds}")

            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            balance_train = balance_df[train_idx] if balance_df is not None else None
            balance_val = balance_df[val_idx] if balance_df is not None else None

            # 🏷️ Show balance summary in the train and validation sets
            if balance_train is not None and debug:
                self.show_balance_summary(y_train, balance_train, y_val, balance_val)

            # Now proceed with training...
            if isinstance(self.model, nn.Module):
                train_loader, val_loader = self._create_loaders(X_train, y_train, X_val, y_val)
                model_fold = self._clone_model()
                self._train_pytorch(model_fold, train_loader, val_loader)
                metrics = self._evaluate_pytorch(model_fold, val_loader)
            else:
                model_fold = self._clone_model()
                model_fold.fit(X_train, y_train)
                y_pred = model_fold.predict(X_val)
                acc = accuracy_score(y_val, y_pred)
                f1 = f1_score(y_val, y_pred, average='weighted')
                precision = precision_score(y_val, y_pred, average='weighted', zero_division=0)
                recall = recall_score(y_val, y_pred, average='weighted', zero_division=0)
                cm = confusion_matrix(y_val, y_pred)
                metrics = {
                    'accuracy': acc,
                    'f1': f1,
                    'precision': precision,
                    'recall': recall,
                    'confusion_matrix': cm,
                    'y_true': y_val,
                    'y_pred': y_pred
                }

            print(f"🎯 Fold {fold+1} Accuracy: {metrics['accuracy']:.2%}, F1: {metrics['f1']:.2%}, Precision: {metrics['precision']:.2%}, Recall: {metrics['recall']:.2%}")
            fold_metrics.append(metrics)

            if self.tracker:
                self.tracker.log_metric(f"fold_{fold+1}_accuracy", metrics['accuracy'])
                self.tracker.log_metric(f"fold_{fold+1}_f1", metrics['f1'])
                self.tracker.log_metric(f"fold_{fold+1}_precision", metrics['precision'])
                self.tracker.log_metric(f"fold_{fold+1}_recall", metrics['recall'])

        avg_acc = np.mean([m['accuracy'] for m in fold_metrics])
        avg_f1 = np.mean([m['f1'] for m in fold_metrics])
        avg_precision = np.mean([m['precision'] for m in fold_metrics])
        avg_recall = np.mean([m['recall'] for m in fold_metrics])
        print(f"✅ Cross-Validation Avg Accuracy: {avg_acc:.2%}, Avg F1: {avg_f1:.2%}, Precision: {avg_precision:.2%}, Recall: {avg_recall:.2%}")
        if self.tracker:
            self.tracker.log_metric("cv_avg_accuracy", avg_acc)
            self.tracker.log_metric("cv_avg_f1", avg_f1)
            self.tracker.log_metric("cv_avg_precision", avg_precision)
            self.tracker.log_metric("cv_avg_recall", avg_recall)

        return avg_acc, avg_f1, avg_precision, avg_recall, fold_metrics


    def _create_loaders(self, X_train, y_train, X_val, y_val):
        from torch.utils.data import TensorDataset, DataLoader # type: ignore
        train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train))
        val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val))
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False)
        return train_loader, val_loader

    def _clone_model(self):
        import copy
        if self.model in ['xgboost', 'lightgbm']:
            # Re-initialize with same params instead of deepcopy
            return type(self.model)(**self.model.get_params())
        else:
            return copy.deepcopy(self.model)

    def _get_optimizer(self, model):
        opt_map = {
            'Adam': optim.Adam,
            'SGD': optim.SGD,
            'RMSprop': optim.RMSprop,
            'Adagrad': optim.Adagrad
        }
        opt_cls = opt_map.get(self.optimizer_name, optim.Adam)
        return opt_cls(model.parameters(), lr=self.learning_rate)

    def _get_criterion(self):
        crit_map = {
            'CrossEntropyLoss': nn.CrossEntropyLoss(),
            'MSELoss': nn.MSELoss(),
            'BCEWithLogitsLoss': nn.BCEWithLogitsLoss()
        }
        return crit_map.get(self.criterion_name, nn.CrossEntropyLoss())


    def _train_pytorch(self, model, train_loader, val_loader=None, debug=True):
        model.to(self.device)
        criterion = self._get_criterion()
        optimizer = self._get_optimizer(model)
        model.train()
        best_val_loss = float('inf')
        epochs_no_improve = 0

        print(f"Training PyTorch model with {self.optimizer_name} optimizer and {self.criterion_name} loss function...") if debug else None

        for epoch in range(self.num_epochs):
            total_loss = 0
            for features, labels in train_loader:
                print(f"Epoch {epoch+1}/{self.num_epochs}...") if debug else None
                features, labels = features.to(self.device), labels.to(self.device)
                print(f"  Input type: {type(features)}, dtype: {features.dtype}, shape: {features.shape}") if debug else None
                outputs = model(features)
                print(f"  Output shape: {outputs.shape}") if debug else None
                loss = criterion(outputs, labels)
                print(f"  Loss: {loss.item():.4f}") if debug else None
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"Epoch [{epoch+1}/{self.num_epochs}], Loss: {total_loss:.4f}")

            # Early stopping check (if val_loader provided)
            if self.early_stopping_patience and val_loader is not None:
                val_loss = self._compute_validation_loss(model, val_loader, criterion)
                print(f"Validation Loss: {val_loss:.4f}")
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= self.early_stopping_patience:
                        print(f"⏹️ Early stopping triggered at epoch {epoch+1}")
                        break

    def _compute_validation_loss(self, model, val_loader, criterion):
        model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(self.device), labels.to(self.device)
                outputs = model(features)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
        return total_loss / len(val_loader)

    def _evaluate_pytorch(self, model, val_loader):
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(self.device), labels.to(self.device)
                outputs = model(features)
                _, predicted = torch.max(outputs, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted')
        precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
        cm = confusion_matrix(all_labels, all_preds)
        return {
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'confusion_matrix': cm,
            'y_true': all_labels,
            'y_pred': all_preds
        }
    
    def evaluate_on_dataset(self, dataset):
        print("🔍 Evaluating on zero-shot dataset...")
        X = dataset.tensors[0].numpy()
        y = dataset.tensors[1].numpy()
        if isinstance(self.model, nn.Module):
            loader = self._create_loaders(X, y, X, y)[1]
            metrics = self._evaluate_pytorch(self.model, loader)
        else:
            y_pred = self.model.predict(X)
            acc = accuracy_score(y, y_pred)
            f1 = f1_score(y, y_pred, average='weighted')
            precision = precision_score(y, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y, y_pred, average='weighted', zero_division=0)
            cm = confusion_matrix(y, y_pred)
            metrics = {
                'accuracy': acc,
                'f1': f1,
                'precision': precision,
                'recall': recall,
                'confusion_matrix': cm,
                'y_true': y,
                'y_pred': y_pred
            }
        print(f"🎯 Zero-Shot Accuracy: {metrics['accuracy']:.2%}, F1: {metrics['f1']:.2%}, Precision: {metrics['precision']:.2%}, Recall: {metrics['recall']:.2%}")
        if self.tracker:
            self.tracker.log_metric("zero_shot_accuracy", metrics['accuracy'])
            self.tracker.log_metric("zero_shot_f1", metrics['f1'])
            self.tracker.log_metric("zero_shot_precision", metrics['precision'])
            self.tracker.log_metric("zero_shot_recall", metrics['recall'])
            # Optionally log confusion matrix as an artifact or image
        return metrics
    
    def show_balance_summary(self, y_train, balance_train, y_val=None, balance_val=None):
        """
        Show balance summary for training and validation sets.
        Args:
            y_train: Labels for training set
            balance_train: Balance column values for training set
            y_val: Labels for validation set (optional)
            balance_val: Balance column values for validation set (optional)
        """
        from collections import Counter
    
        def sorted_items(counter):
            # Sort by balance (organism), then by label (0 before 1)
            return sorted(
                counter.items(),
                key=lambda x: (x[0][1], x[0][0])  # (balance, label)
            )
    
        train_summary = Counter(zip(y_train, balance_train))
        val_summary = Counter(zip(y_val, balance_val))
        print("🔎 Training Fold Balance Summary (label, balance_col):")
        for (label, balance), count in sorted_items(train_summary):
            print(f"  Label: {label}, Balance: {balance}, Count: {count}")
        print("🔎 Validation Fold Balance Summary (label, balance_col):")
        for (label, balance), count in sorted_items(val_summary):
            print(f"  Label: {label}, Balance: {balance}, Count: {count}")