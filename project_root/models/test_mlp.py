# Final commit – Master’s Thesis by Àlex Domínguez Roig

import torch
from project_root.models.mlp_protein_classifier import MLPProteinClassifier

device = "cpu"
X = torch.randn(32, 8).to(device)

model = MLPProteinClassifier(
    input_size=8,
    output_size=2,
    num_hidden_layers=2,
    hidden_layers_mode="quadratic_increase",
    initialization="kaiming",
    activation_function="ReLU",
    use_batch_norm=False
).to(device)

out = model(X)
print("✅ Output shape:", out.shape)
