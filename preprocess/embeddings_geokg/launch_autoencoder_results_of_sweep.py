# launch_autoencoder_results.py

from train_go_autoencoder_wandb import train  # your training function

# Define the 8 configurations
configs = [
    {
        "name": "clean-sweep-1",
        "activation": "tanh",
        "batch_size": 32,
        "input_dim": 200,
        "latent_dim": 32,
        "learning_rate": 0.000887,
        "epochs": 20,
        "optimizer": "Adam"
    },
    {
        "name": "cosmic-sweep-18",
        "activation": "tanh",
        "batch_size": 64,
        "input_dim": 500,
        "latent_dim": 32,
        "learning_rate": 0.000212,
        "epochs": 50,
        "optimizer": "Adam"
    },
    {
        "name": "dry-sweep-6",
        "activation": "relu",
        "batch_size": 64,
        "input_dim": 1000,
        "latent_dim": 32,
        "learning_rate": 0.001273,
        "epochs": 50,
        "optimizer": "Adam"
    },
    {
        "name": "flowing-sweep-77",
        "activation": "relu",
        "batch_size": 32,
        "input_dim": 200,
        "latent_dim": 64,
        "learning_rate": 0.001209,
        "epochs": 40,
        "optimizer": "Adam"
    },
    {
        "name": "colorful-sweep-7",
        "activation": "leaky_relu",
        "batch_size": 64,
        "input_dim": 500,
        "latent_dim": 64,
        "learning_rate": 0.004335,
        "epochs": 10,
        "optimizer": "Adam"
    },
    {
        "name": "peach-sweep-30",
        "activation": "leaky_relu",
        "batch_size": 64,
        "input_dim": 200,
        "latent_dim": 128,
        "learning_rate": 0.000653,
        "epochs": 10,
        "optimizer": "Adam"
    },
    {
        "name": "colorful-sweep-26",
        "activation": "leaky_relu",
        "batch_size": 64,
        "input_dim": 500,
        "latent_dim": 128,
        "learning_rate": 0.000515,
        "epochs": 50,
        "optimizer": "Adam"
    },
    {
        "name": "amber-sweep-10",
        "activation": "tanh",
        "batch_size": 64,
        "input_dim": 1000,
        "latent_dim": 128,
        "learning_rate": 0.00178,
        "epochs": 40,
        "optimizer": "Adam"
    }
]

# 🚀 Launch training for each configuration
for i, config in enumerate(configs):
    print(f"Training model {i+1}/{len(configs)}: {config['name']}")
    print("Configuration:", config)
    train(wandb_run=False, config=config)
