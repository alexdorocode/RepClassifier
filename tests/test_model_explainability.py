import pytest
import torch
import torch.nn as nn
from project_root.explainability.model_explainability import ModelExplainability 

# A small dummy model with predictable behavior
class DummyClassifier(nn.Module):
    def __init__(self, input_size=10, output_size=1):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_size, 20),
            nn.ReLU(),
            nn.Linear(20, output_size)
        )

    def forward(self, x):
        return self.classifier(x)

@pytest.fixture
def setup_model_and_data():
    input_size = 10
    batch_size = 8
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DummyClassifier(input_size=input_size).to(device)

    input_example = torch.randn(1, input_size)
    test_data = torch.randn(batch_size, input_size)
    target_class = 0

    explainer = ModelExplainability(model, device, input_example)
    return explainer, test_data, target_class

def test_shap_runs(setup_model_and_data):
    explainer, test_data, _ = setup_model_and_data
    # Use the same data for background and test
    shap_values = explainer.explain_with_shap(test_data, test_data, plot_summary=False)
    assert shap_values is not None
    assert hasattr(shap_values, 'values')

def test_layer_activations_run(setup_model_and_data):
    explainer, test_data, _ = setup_model_and_data
    try:
        explainer.visualize_layer_activations(test_data)
    except Exception as e:
        pytest.fail(f"Layer activation visualization failed: {e}")

def test_integrated_gradients_runs(setup_model_and_data):
    explainer, test_data, target_class = setup_model_and_data
    try:
        attributions, delta = explainer.explain_with_integrated_gradients(test_data[:1], target_class)
        assert attributions.shape == test_data[:1].shape
    except Exception as e:
        pytest.fail(f"Integrated Gradients failed: {e}")

def test_gradcam_runs(setup_model_and_data):
    explainer, test_data, target_class = setup_model_and_data
    try:
        attribution = explainer.explain_with_gradcam(test_data[:1], target_class)
        assert attribution is not None
    except Exception as e:
        pytest.fail(f"Grad-CAM failed: {e}")
