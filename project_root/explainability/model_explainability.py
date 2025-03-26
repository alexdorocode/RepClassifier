import shap
import torch
import matplotlib.pyplot as plt
import numpy as np
from torch import nn
from captum.attr import IntegratedGradients, LayerGradCam
from typing import List

class ModelExplainability:
    """
    A class for explaining the predictions and internal behavior of the ProteinClassifier model.
    It includes SHAP-based feature importance, intermediate layer analysis, and gradient-based methods.

    Parameters:
    - model: The trained ProteinClassifier.
    - device: The device used (CPU/GPU).
    - input_example: A representative input tensor for the model, used for SHAP explainer setup.
    """
    
    def __init__(self, model: nn.Module, device: torch.device, input_example: torch.Tensor):
        self.model = model
        self.device = device
        self.input_example = input_example.to(device)
        self.model.eval()

    def explain_with_shap(self, background_data: torch.Tensor, test_data: torch.Tensor, plot_summary=True):
        """
        Uses SHAP to compute feature importance values.
        """
        background_data = background_data.to(self.device)
        test_data = test_data.to(self.device)

        # Convert to numpy for SHAP
        background_np = background_data.detach().cpu().numpy()
        test_np = test_data.detach().cpu().numpy()

        def model_forward(x):
            x_tensor = torch.tensor(x, dtype=torch.float32).to(self.device)
            with torch.no_grad():
                return self.model(x_tensor).cpu().numpy()

        masker = shap.maskers.Independent(background_np)
        explainer = shap.Explainer(model_forward, masker)

        shap_values = explainer(test_np)

        if plot_summary:
            shap.plots.beeswarm(shap_values)

        return shap_values


    def visualize_layer_activations(self, input_tensor: torch.Tensor, layer_indices: List[int] = None):
        """
        Extracts and visualizes activations from specified intermediate layers.

        This helps interpret how information flows and transforms through the network.
        Can reveal whether certain layers specialize in detecting certain types of patterns.

        Parameters:
        - input_tensor: The input to feed forward.
        - layer_indices: List of layer indices to visualize (default is all linear layers).
        """
        activations = []
        hooks = []
        input_tensor = input_tensor.to(self.device)

        def get_activation_hook(storage):
            def hook(model, input, output):
                storage.append(output.detach().cpu())
            return hook

        for idx, layer in enumerate(self.model.classifier):
            if isinstance(layer, nn.Linear) and (layer_indices is None or idx in layer_indices):
                hooks.append(layer.register_forward_hook(get_activation_hook(activations)))

        with torch.no_grad():
            _ = self.model(input_tensor)

        for hook in hooks:
            hook.remove()

        for i, activation in enumerate(activations):
            plt.figure()
            plt.title(f"Activation of Layer {i}")
            plt.imshow(activation.numpy(), aspect='auto', cmap='viridis')
            plt.colorbar()
            plt.xlabel("Neurons")
            plt.ylabel("Samples")
            plt.show()

    def explain_with_integrated_gradients(self, input_tensor: torch.Tensor, target_class: int):
        """
        Applies Integrated Gradients to show which features contribute most to a prediction.

        This method provides attribution scores for each input feature. It's based on the path integral
        of gradients from a baseline input to the actual input.

        Parameters:
        - input_tensor: The input sample to explain.
        - target_class: The class index to attribute the explanation to.
        """
        input_tensor = input_tensor.to(self.device)
        ig = IntegratedGradients(self.model)
        attributions, delta = ig.attribute(inputs=input_tensor, target=target_class, return_convergence_delta=True)

        plt.figure()
        plt.title("Integrated Gradients Attributions")
        plt.bar(range(attributions.shape[-1]), attributions.squeeze().detach().cpu().numpy())
        plt.xlabel("Feature Index")
        plt.ylabel("Attribution Score")
        plt.show()

        return attributions, delta

    def explain_with_gradcam(self, input_tensor: torch.Tensor, target_class: int, layer_index: int = -3):
        """
        Uses Grad-CAM to visualize which neurons (features) are most important for a decision,
        based on gradients flowing into a specific layer.

        Grad-CAM is especially useful for understanding spatial patterns or feature clusters
        the model is using to make predictions.

        Parameters:
        - input_tensor: The input sample to analyze.
        - target_class: The target class index.
        - layer_index: The classifier layer index to analyze (typically one of the last layers).
        """
        input_tensor = input_tensor.to(self.device)

        linear_layers = [layer for layer in self.model.classifier if isinstance(layer, nn.Linear)]
        if len(linear_layers) == 0:
            raise ValueError("No linear layers found for Grad-CAM analysis.")

        if abs(layer_index) > len(linear_layers):
            print(f"[Grad-CAM] Only {len(linear_layers)} Linear layers found. Using last layer instead.")
            target_layer = linear_layers[-1]
        else:
            target_layer = linear_layers[layer_index]

        gradcam = LayerGradCam(self.model, target_layer)
        attribution = gradcam.attribute(inputs=input_tensor, target=target_class)

        plt.figure()
        plt.title("Grad-CAM Attribution")
        plt.plot(attribution.squeeze().detach().cpu().numpy())
        plt.xlabel("Feature Index")
        plt.ylabel("Importance")
        plt.show()

        return attribution
