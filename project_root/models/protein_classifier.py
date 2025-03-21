import torch
import torch.nn as nn

class ProteinClassifier(nn.Module):
    def __init__(self, device, input_size, output_size, num_hidden_layers, dropout_rate=0.1, hidden_layers_mode="quadratic_increase", custom_hidden_layers=None):
        super(ProteinClassifier, self).__init__()
        hidden_layers_sizes = self.set_hidden_layers_size(hidden_layers_mode, num_hidden_layers, input_size, custom_hidden_layers)
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_size, hidden_layers_sizes[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))
        
        # Hidden layers
        for i in range(1, len(hidden_layers_sizes)):
            layers.append(nn.Linear(hidden_layers_sizes[i-1], hidden_layers_sizes[i]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
        
        # Output layer
        layers.append(nn.Linear(hidden_layers_sizes[-1], output_size))
        
        self.classifier = nn.Sequential(*layers).to(device)


    def forward(self, input):
        # Pasamos los embeddings por el clasificador para obtener los logits
        logits = self.classifier(input)
        return logits
    

    def set_hidden_layers_size(self, hidden_layers_mode, num_hidden_layers, input_size, custom_hidden_layers=None):

        if hidden_layers_mode == "quadratic_increase":
            # Calculate hidden layer sizes
            hidden_layer_sizes = [input_size // (2 ** (num_hidden_layers - i)) for i in range(num_hidden_layers)]
        elif hidden_layers_mode == "custom" and custom_hidden_layers is not None:
            hidden_layer_sizes = custom_hidden_layers
        else:
            raise ValueError("Invalid hidden_layers_mode or custom_hidden_layers not provided.")
        
        return hidden_layer_sizes
