import random
import numpy as np
import torch
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score

from project_root.models.classifier_model_loader import ClassifierModelLoader
from project_root.training.trainer_module import TrainerModule
from project_root.explainability.explainability_module import ExplainabilityModule
from project_root.training.tracker_module import TrackerModule

class ClassifierLauncher:
    """
    Orchestrates model training, evaluation, zero-shot testing, and explainability for a given configuration.
    """

    def __init__(self, config_reader, dataset_handler, zero_shot_test=False, random_seed=None):
        """
        Initialize the ClassifierLauncher.

        :param config_reader: Configuration reader object
        :param dataset_handler: Dataset handler object
        :param zero_shot_test: If True, perform zero-shot testing after training
        :param random_seed: Random seed for reproducibility
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Initialize config reader to access parsed dictionaries
        self.config_reader = config_reader
        self.dataset_handler = dataset_handler
        self.zero_shot_test = zero_shot_test

        self.random_seed = random_seed
        self._set_all_seeds(self.random_seed)

        print(f"🔍 Setting random seed to {self.random_seed} for reproducibility.")

        # Initialize dataset
        self._initialize_dataset()

        # Initialize model
        self._initialize_model()

        # Initialize tracker
        tracker_cfg = self.config_reader.tracker
        self.tracker = TrackerModule(
            project_name=tracker_cfg['project_name'],
            run_name=tracker_cfg['run_name'],
            offline=tracker_cfg.get('offline', False),
            config=self.config_reader,  # Store the entire config_reader as a reference
            random_seed=self.random_seed,
            tags=['phase_final_eval_develop', 'zero_shot', 'classifier', f'{self.config_reader.model["type"]}']
        )
        
    def _initialize_dataset(self):
        """
        Load and split the dataset into train and test sets.
        """
        self.classifier_dataset = self.dataset_handler.load_classifier_dataset()
        X, y = self.classifier_dataset.get_X_y()

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.config_reader.training['test_size_ratio'], random_state=self.random_seed, stratify=y
        )
        
    def _initialize_model(self):
        """
        Initialize the model using the configuration.
        """
        input_size = self.classifier_dataset.features.shape[1]
        output_size = len(torch.unique(self.classifier_dataset.labels))
        model_cfg = self.config_reader.model
        self.model_loader = ClassifierModelLoader(
            model_type=model_cfg['type'],
            input_size=input_size,
            output_size=output_size,
            device=self.device,
            model_params=model_cfg['params']
        )
        self.model = self.model_loader.get_model()

    def _set_all_seeds(self, seed):
        """
        Set seeds for reproducibility across random, numpy, and torch.

        :param seed: Random seed (int)
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def run(self):
        """
        Run training, (optionally) zero-shot testing, and explainability.

        :return: Tuple (metrics dict, run_results list)
        """
        try:
            training_cfg = self.config_reader.training
            mlp_params = training_cfg.get('mlp_params', {})
            trainer = TrainerModule(
                model=self.model,
                model_type=self.config_reader.model['type'],
                device=self.device,
                learning_rate=mlp_params['learning_rate'] if 'learning_rate' in mlp_params else None,
                num_epochs=mlp_params['num_epochs'] if 'num_epochs' in mlp_params else None,
                cv_folds=training_cfg['cv_folds'],
                tracker=self.tracker,
                optimizer_name=mlp_params['optimizer'] if 'optimizer' in mlp_params else None,
                criterion_name=mlp_params['criterion'] if 'criterion' in mlp_params else None,
                early_stopping_patience=mlp_params['early_stopping_patience'] if 'early_stopping_patience' in mlp_params else None,
                batch_size=mlp_params['batch_size'] if 'batch_size' in mlp_params else None,
                random_seed=self.random_seed,
            )
            avg_acc, avg_f1, avg_precision, avg_recall, fold_metrics = trainer.cross_validate(self.X_train, self.y_train, self.classifier_dataset.balance_values)
            print(f"✅ Training done! Accuracy: {avg_acc:.2%}, F1: {avg_f1:.2%}, Precision: {avg_precision:.2%}, Recall: {avg_recall:.2%}")

            metrics = {
                "accuracy": avg_acc,
                "f1": avg_f1,
                "precision": avg_precision,
                "recall": avg_recall
            }

            if self.zero_shot_test:
                # Run zero-shot testing
                acc, f1, precision, recall, run_results = self._run_zero_shot()
                print(f"🔍 Zero-shot testing results - Accuracy: {acc:.2%}, F1: {f1:.2%}, Precision: {precision:.2%}, Recall: {recall:.2%}")
                self.tracker.log_metric("zero_shot_accuracy", acc)
                self.tracker.log_metric("zero_shot_f1", f1)
                self.tracker.log_metric("zero_shot_precision", precision)
                self.tracker.log_metric("zero_shot_recall", recall)

                metrics.update({
                    "zero_shot_accuracy": acc,
                    "zero_shot_f1": f1,
                    "zero_shot_precision": precision,
                    "zero_shot_recall": recall
                })
            else:
                run_results = []

            # Explainability (optional)
            if self.config_reader.explainability:
                self._run_explainability()
            
            return metrics, run_results

        finally:
            self.tracker.finish()
        
    def _run_zero_shot(self):
        """
        Perform zero-shot testing on the zero-shot dataset.

        :return: Tuple (accuracy, f1, precision, recall, run_results)
        """
        print("🧪 Performing zero-shot testing...")
        zero_shot_dataset = self.dataset_handler.load_classifier_dataset(zero_shot=True)
        X_zero, y_zero = zero_shot_dataset.get_X_y()
        protein_ids = zero_shot_dataset.get_ids()
        
        # Check if the model is a torch.nn.Module
        print(f"🔍 Running zero-shot evaluation for {type(self.model)}")
        if isinstance(self.model, torch.nn.Module):
            self.model.eval()
            with torch.no_grad():
                inputs = torch.tensor(X_zero, dtype=torch.float32).to(self.device)
                outputs = self.model(inputs)
                preds_class = torch.argmax(outputs, dim=1).cpu().numpy()
        else:
            # For sklearn-like models (e.g., XGBClassifier)
            self.model.fit(self.X_train, self.y_train) 
            preds_class = self.model.predict(X_zero)

        acc = accuracy_score(y_zero, preds_class)
        f1 = f1_score(y_zero, preds_class, average='weighted')
        precision = precision_score(y_zero, preds_class, average='weighted')
        recall = recall_score(y_zero, preds_class, average='weighted')
        
        # Collect protein-level info
        run_results = []
        for pid, pred, true in zip(protein_ids, preds_class, y_zero):
            run_results.append({
                "protein_id": pid,
                "true_label": int(true),
                "pred_label": int(pred),
                "is_correct": int(pred == true)
            })

        return acc, f1, precision, recall, run_results

    def _run_explainability(self):
        """
        Run explainability on the model and print the results.
        """
        print("🧠 Running explainability...")
        explainer = ExplainabilityModule(self.model, self.config_reader.model['type'], device=self.device)
        explanation_df = explainer.explain(
            self.classifier_dataset.features[:5], 
            feature_names=self.classifier_dataset.feature_cols,
            target=1  # or 0, depending on which class you want to explain
        )
        print(f"🧠 Explainability output:\n{explanation_df.head()}")