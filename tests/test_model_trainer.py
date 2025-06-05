import pytest
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd

from project_root.models.classifier_model_loader import ClassifierModelLoader
from project_root.training.trainer_module import TrainerModule
from project_root.explainability.explainability_module import ExplainabilityModule
from project_root.training.tracker_module import TrackerModule

# Dummy dataset generator
def generate_dummy_dataset(num_samples=100, input_size=10, num_classes=2):
    X = np.random.rand(num_samples, input_size).astype(np.float32)
    y = np.random.randint(0, num_classes, size=(num_samples,))
    return X, y

@pytest.mark.parametrize("model_type, model_params", [
    ("logistic", {"max_iter": 100}),
    ("svm", {"kernel": "linear", "probability": True}),
    ("mlp", {
        "num_hidden_layers": 2, "dropout_rate": 0.1, "hidden_layers_mode": "quadratic_increase",
        "activation_function": "ReLU", "use_batch_norm": False, "output_activation": None, "initialization": None
    }),
    ("xgboost", {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.1}),
    ("random_forest", {"n_estimators": 100, "max_depth": 5, "criterion": "gini"}),
    ("knn", {"n_neighbors": 5, "weights": "uniform"})
])
def test_model_trainer_cross_val(model_type, model_params):
    print(f"🔍 Testing model type: {model_type} with params: {model_params}")

    input_size = 10
    num_classes = 2
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model_loader = ClassifierModelLoader(
        model_type=model_type,
        input_size=input_size,
        output_size=num_classes,
        device=device,
        model_params=model_params
    )
    model = model_loader.get_model()
    
    X, y = generate_dummy_dataset(num_samples=1000, input_size=input_size, num_classes=num_classes)
    dataset = TensorDataset(torch.tensor(X), torch.tensor(y))
    feature_names = [f'feature_{i}' for i in range(input_size)]

    # Initialize W&B tracker
    tracker = TrackerModule(project_name="ProteinClassifier", run_name=f"run_{model_type}", offline=True,
                            config={"model": model_type, "lr": 0.01, "epochs": 5})
    try:
        trainer = TrainerModule(
            model=model,
            model_type=model_type,
            device=device,
            learning_rate=0.01,
            num_epochs=1,
            cv_folds=3,
            tracker=tracker,
            optimizer_name="Adam",
            criterion_name="CrossEntropyLoss"
        )
        
        avg_acc, avg_f1, avg_precision, avg_recall, fold_metrics = trainer.cross_validate(dataset)
        assert 0.0 <= avg_acc <= 1.0
        assert 0.0 <= avg_f1 <= 1.0
        print(f"✅ Passed: {model_type} with avg_acc={avg_acc:.2%}, avg_f1={avg_f1:.2%}")

        # 🔥 Explainability test
        print(f"🧠 Running Explainability for {model_type}...")
        # Fit the model if needed for explainability
        if model_type in ["logistic", "svm", "xgboost", "random_forest", "knn"]:
            model.fit(X, y)

        explainer = ExplainabilityModule(model, model_type, device=device)

        # SVM Explainability: use KernelExplainer, skip LinearExplainer warning
        if model_type == "svm":
            print("⚠️ SVM: Using KernelExplainer; explanations might be approximate.")

        # For MLP, pass a target if needed
        if model_type == "mlp":
            explanation_df = explainer.explain(X[:5], feature_names=feature_names, target=0)
        else:
            explanation_df = explainer.explain(X[:5], feature_names=feature_names)

        assert isinstance(explanation_df, pd.DataFrame)
        assert all(f in explanation_df.columns for f in feature_names)
        print(f"✅ Explainability completed for {model_type}, top rows:\n{explanation_df.head()}")

    finally:
        tracker.finish()

if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__]))  # Makes exit code correct for CI systems
