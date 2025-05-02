import torch
import torch.nn as nn
from torchinfo import summary

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

        # Print torch summary
        print(summary(self, input_size=(1, input_size), device=device))

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

"""

import torch
import torch.nn as nn
from torchinfo import summary

class AttentionAggregator(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.attention_layer = nn.Linear(embedding_dim, 1)  # Linear layer to compute attention scores

    def forward(self, embeddings, mask=None):
        # embeddings: Tensor of shape (num_terms, embedding_dim)
        # mask: Tensor of shape (num_terms,) indicating valid embeddings (1 for valid, 0 for padded)
        
        # Compute attention scores
        attention_scores = self.attention_layer(embeddings)  # Shape: (num_terms, 1)
        attention_scores = attention_scores.squeeze(-1)  # Shape: (num_terms,)
        
        if mask is not None:
            # Apply mask to attention scores
            attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))
        
        # Compute attention weights using softmax
        attention_weights = torch.softmax(attention_scores, dim=0)  # Shape: (num_terms,)
        
        # Compute weighted sum of embeddings
        weighted_sum = torch.sum(attention_weights.unsqueeze(-1) * embeddings, dim=0)  # Shape: (embedding_dim,)
        return weighted_sum


class ProteinClassifier(nn.Module):
    def __init__(self, device, input_size, output_size, num_hidden_layers, dropout_rate=0.1, hidden_layers_mode="quadratic_increase", custom_hidden_layers=None):
        super(ProteinClassifier, self).__init__()
        
        # Add AttentionAggregator
        self.aggregator = AttentionAggregator(input_size)
        
        # Define hidden layers
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

        # Print torch summary
        print(summary(self, input_size=(1, input_size), device=device))

    def forward(self, embeddings, mask=None):
        # Aggregate embeddings using AttentionAggregator
        aggregated_embedding = self.aggregator(embeddings, mask)  # Shape: (embedding_dim,)
        
        # Pass aggregated embedding through the classifier
        logits = self.classifier(aggregated_embedding)
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
"""