# Final commit – Master’s Thesis by Àlex Domínguez Roig

import wandb

# Define your sweep configuration
sweep_config = {
    "method": "random",
    "metric": {
        "name": "loss",
        "goal": "minimize"
    },
    "parameters": {
        "emb_model": {
            "values": list(['ESM', 'Prot-T5', 'Prost-T5'])  # Explicitly cast to list
        },
        "latent_dim": {
            "values": list([32, 64, 128])
        },
        "activation": {
            "values": list(["relu", "leaky_relu", "sigmoid", "tanh"])
        },
        "optimizer": {
            "values": list(["Adam", "SGD"])
        },
        "learning_rate": {
            "min": 1e-4,
            "max": 1e-2
        },
        "batch_size": {
            "values": list([32, 64])
        },
        "epochs": {
            "values": list([10, 20, 30, 40, 50])
        }
    }
}

# 🪄 Name your project
project_name = "TFM-seq-autoencoder-tuning"

# 🔁 Launch sweep
sweep_id = wandb.sweep(sweep_config, project=project_name)

# 👇 Import your training function
from train_seq_autoencoder_wandb import train  # must match the training script

# Run the sweep agent
for emb_model in ["ESM", "Prot-T5", "Prost-T5"]:
    sweep_config["parameters"]["emb_model"]["values"] = [emb_model]
    sweep_id = wandb.sweep(sweep_config, project=project_name)
    wandb.agent(sweep_id, function=train, count=100)