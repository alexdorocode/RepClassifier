import torch
import torch.nn as nn

class MLPProteinClassifier(nn.Module):
    def __init__(self, input_size, output_size, num_hidden_layers,
                 dropout_rate=0.1, hidden_layers_mode="quadratic_increase",
                 custom_hidden_layers=None, activation_function="ReLU",
                 use_batch_norm=False, output_activation=None, device=None,
                 initialization=None):
        super().__init__()

        self.hidden_layers_sizes = self._set_hidden_layers_size(
            hidden_layers_mode, num_hidden_layers, input_size, custom_hidden_layers
        )

        self.classifier = self._build_classifier(
            input_size=input_size,
            output_size=output_size,
            hidden_layers_sizes=self.hidden_layers_sizes,
            dropout_rate=dropout_rate,
            activation_function=activation_function,
            use_batch_norm=use_batch_norm,
            output_activation=output_activation
        )

        if initialization:
            self.apply(self._get_initializer(initialization))

    def forward(self, x):
        return self.classifier(x)

    def _set_hidden_layers_size(self, mode, num_layers, input_size, custom_sizes):
        if mode == "quadratic_increase":
            return [input_size // (2 ** (num_layers - i)) for i in range(num_layers)]
        elif mode == "custom":
            if custom_sizes is None:
                raise ValueError("Custom hidden layer sizes must be provided for 'custom' mode.")
            return custom_sizes
        else:
            raise ValueError(f"Unsupported hidden_layers_mode: {mode}")

    def _build_classifier(self, input_size, output_size, hidden_layers_sizes,
                          dropout_rate, activation_function,
                          use_batch_norm, output_activation):
        layers = []

        # Define activation function class
        activation_cls = getattr(nn, activation_function, None)
        if activation_cls is None:
            raise ValueError(f"Invalid activation function: {activation_function}")

        # Input layer
        layers.append(nn.Linear(input_size, hidden_layers_sizes[0]))
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(hidden_layers_sizes[0]))
        layers.append(activation_cls())
        layers.append(nn.Dropout(dropout_rate))

        # Hidden layers
        for i in range(1, len(hidden_layers_sizes)):
            layers.append(nn.Linear(hidden_layers_sizes[i-1], hidden_layers_sizes[i]))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_layers_sizes[i]))
            layers.append(activation_cls())
            layers.append(nn.Dropout(dropout_rate))

        # Output layer
        layers.append(nn.Linear(hidden_layers_sizes[-1], output_size))

        # Optional output activation (e.g., Sigmoid for binary classification)
        if output_activation:
            output_cls = getattr(nn, output_activation, None)
            if output_cls is None:
                raise ValueError(f"Invalid output activation: {output_activation}")
            layers.append(output_cls())

        return nn.Sequential(*layers)

    def _get_initializer(self, method):
        def init_weights(m):
            if isinstance(m, nn.Linear):
                if method == "xavier":
                    nn.init.xavier_uniform_(m.weight)
                elif method == "kaiming":
                    nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                elif method == "normal":
                    nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        return init_weights
