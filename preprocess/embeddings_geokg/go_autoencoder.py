import torch
import torch.nn as nn
import math

def next_power_of_two(n):
    return 2 ** math.ceil(math.log2(n))

def generate_hidden_dims(input_dim, latent_dim):
    """Generate a list of hidden dimensions from input_dim down to latent_dim using powers of 2"""
    dims = []
    current = next_power_of_two(input_dim)
    target = max(latent_dim * 2, 32)

    while current > latent_dim:
        next_dim = current // 2
        if next_dim < latent_dim:
            break
        dims.append(next_dim)
        current = next_dim

    return dims

def get_activation_function(name):
    """Return the activation function based on the name"""
    activations = {
        "relu": nn.ReLU(),
        "sigmoid": nn.Sigmoid(),
        "tanh": nn.Tanh(),
        "leaky_relu": nn.LeakyReLU(),
    }
    return activations.get(name.lower(), nn.ReLU())  # Default to ReLU if not found

class GOAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim, activation_name="relu"):
        super().__init__()
        
        # Get the activation function
        activation = get_activation_function(activation_name)
        
        # Build encoder layers
        encoder_layers = []
        last_dim = input_dim
        hidden_dims = generate_hidden_dims(input_dim, latent_dim)
        
        for h in hidden_dims:
            encoder_layers.append(nn.Linear(last_dim, h))
            encoder_layers.append(activation)
            last_dim = h
        
        encoder_layers.append(nn.Linear(last_dim, latent_dim))  # Final latent layer
        self.encoder = nn.Sequential(*encoder_layers)

        # Build decoder layers (reverse of encoder)
        decoder_layers = []
        last_dim = latent_dim
        for h in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(last_dim, h))
            decoder_layers.append(activation)
            last_dim = h
        
        decoder_layers.append(nn.Linear(last_dim, input_dim))  # Reconstruct to input
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)