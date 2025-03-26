import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns # type: ignore
from sklearn.metrics import confusion_matrix, classification_report, cohen_kappa_score # type: ignore

from project_root.explainability.model_explainability import ModelExplainability 

class Trainer:
    def __init__(self, model_class, model_size='small', lr=5e-5, explainability=False):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model_class(model_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.explainability = explainability
        self.explainer = None  # Will be initialized after seeing data

    def train_and_validate(self, train_dataloader, test_dataloader, num_epochs):
        train_losses, val_losses, train_accuracies, val_accuracies = [], [], [], []
        input_example = None

        for epoch in range(num_epochs):
            train_loss, train_accuracy = self.train_one_epoch(train_dataloader)
            val_loss, val_accuracy = self.validate_one_epoch(test_dataloader)

            # Capture one input sample for explainability
            if epoch == 0:
                for x, _ in train_dataloader:
                    input_example = x[:1]
                    break

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accuracies.append(train_accuracy)
            val_accuracies.append(val_accuracy)
            print(f'Epoch {epoch + 1}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, '
                  f'Training Accuracy: {train_accuracy:.4f}, Validation Accuracy: {val_accuracy:.4f}')

        if self.explainability and input_example is not None:
            self.explainer = ModelExplainability(self.model, self.device, input_example)
            self.run_explainability(train_dataloader)

        return train_losses, val_losses, train_accuracies, val_accuracies

    def run_explainability(self, dataloader):
        """
        Runs SHAP, Layer Activations, Integrated Gradients and Grad-CAM on a batch from dataloader.
        Modify this to suit specific outputs you're interested in.
        """
        # Get a small batch for analysis
        for x, y in dataloader:
            x = x[:10]
            y = y[:10]
            break

        print("\nRunning SHAP explainability...")
        shap_values = self.explainer.explain_with_shap(background_data=x, test_data=x)

        print("\nVisualizing layer activations...")
        self.explainer.visualize_layer_activations(input_tensor=x)

        print("\nIntegrated Gradients for class 0:")
        self.explainer.explain_with_integrated_gradients(input_tensor=x[:1], target_class=0)

        print("\nGrad-CAM for class 0:")
        self.explainer.explain_with_gradcam(input_tensor=x[:1], target_class=0)


    def train_one_epoch(self, dataloader):
        self.model.train()
        total_loss, correct_predictions, total_predictions = 0, 0, 0
        for sequences, labels in dataloader:
            sequences, labels = sequences.to(self.device), labels.to(self.device).float().unsqueeze(-1)
            self.optimizer.zero_grad()
            logits = self.model(sequences)
            loss = self.loss_fn(logits, labels)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            preds = torch.sigmoid(logits) >= 0.5
            correct_predictions += (preds == labels.bool()).sum().item()
            total_predictions += labels.size(0)
        avg_loss = total_loss / len(dataloader)
        accuracy = correct_predictions / total_predictions
        return avg_loss, accuracy

    def validate_one_epoch(self, dataloader):
        self.model.eval()
        total_loss, correct_predictions, total_predictions = 0, 0, 0
        with torch.no_grad():
            for sequences, labels in dataloader:
                sequences, labels = sequences.to(self.device), labels.to(self.device).float().unsqueeze(-1)
                logits = self.model(sequences)
                loss = self.loss_fn(logits, labels)
                total_loss += loss.item()
                preds = torch.sigmoid(logits) >= 0.5
                correct_predictions += (preds == labels.bool()).sum().item()
                total_predictions += labels.size(0)
        avg_loss = total_loss / len(dataloader)
        accuracy = correct_predictions / total_predictions
        return avg_loss, accuracy
    

    def plot_results(self, train_losses, val_losses, train_accuracies, val_accuracies):
        plt.figure(figsize=(16, 8))

        # Gráfico de pérdidas
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.title('Training and Validation Loss', fontsize=22)
        plt.xlabel('Epochs', fontsize=20)
        plt.ylabel('Loss', fontsize=20)
        plt.legend(fontsize=16)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)

        # Gráfico de precisión
        plt.subplot(1, 2, 2)
        plt.plot(train_accuracies, label='Training Accuracy')
        plt.plot(val_accuracies, label='Validation Accuracy')
        plt.title('Training and Validation Accuracy', fontsize=22)
        plt.xlabel('Epochs', fontsize=20)
        plt.ylabel('Accuracy', fontsize=20)
        plt.legend(fontsize=16)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)

        plt.tight_layout()
        plt.show()

    def evaluate_and_plot_confusion_matrix(self, dataloader):
        self.model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for sequences, labels in dataloader:
                sequences = sequences.to(self.device)
                labels = labels.to(self.device).float().unsqueeze(-1)
                logits = self.model(sequences)
                preds = torch.sigmoid(logits) >= 0.5

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        cm = confusion_matrix(all_labels, all_preds)
        tn, fp, fn, tp = cm.ravel()

        specificity = tn / (tn + fp)
        kappa = cohen_kappa_score(all_labels, all_preds)

        plt.figure(figsize=(12, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Negative', 'Positive'],
                    yticklabels=['Negative', 'Positive'],
                    annot_kws={"size": 24})
        plt.xlabel('Predicted Label', fontsize=20)
        plt.ylabel('True Label', fontsize=20)
        plt.title('Confusion Matrix', fontsize=22)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.show()

        print("\nMétricas de Evaluación:")
        print(classification_report(all_labels, all_preds, target_names=['Negative', 'Positive']))
        print(f"Especificidad: {specificity:.4f}")
        print(f"Estadístico Kappa: {kappa:.4f}")


# Example usage:
# trainer = Trainer(ProteinClassifier, model_size='small')
# num_epochs = 200
# train_losses, val_losses, train_accuracies, val_accuracies = trainer.train_and_validate(train_dataloader, test_dataloader, num_epochs)
# trainer.plot_results(train_losses, val_losses, train_accuracies, val_accuracies)
# trainer.evaluate_and_plot_confusion_matrix(test_dataloader)