import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from preprocess.embeddings_geokg.go_autoencoder import GOAutoencoder

import wandb

EMB_PATH = "../DATASETS/embeddings/GeOKG/goa_embedding/"
SAVE_PATH = "../DATASETS/embeddings/GeOKG/autoencoders/"
EMB_DIMENSIONS = [50, 100, 200, 500, 1000]


# ==== Sweep Training Function ====
def train():
    wandb.init()
    config = wandb.config

    # Load data
    emb_path = f"../DATASETS/embeddings/GeOKG/goa_embedding/GeOKG_{config.input_dim}dim.npy"
    embeddings = np.load(emb_path)
    data = torch.tensor(embeddings, dtype=torch.float32)
    dataset = TensorDataset(data)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    # Model
    model = GOAutoencoder(input_dim=config.input_dim, latent_dim=config.latent_dim, activation_name=config.activation)
    optimizer = getattr(torch.optim, config.optimizer)(model.parameters(), lr=config.learning_rate)
    loss_fn = nn.MSELoss()

    model.train()
    for epoch in range(config.epochs):
        total_loss = 0
        for batch in loader:
            x = batch[0]
            loss = loss_fn(model(x), x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(loader)
        wandb.log({"epoch": epoch, "loss": avg_loss})

    # Save encoder
    torch.save(model.state_dict(), f"{SAVE_PATH}geokg_{config.input_dim}dim_autoencoder.pt")
    print("Autoencoder GeOKG {dim}dim saved to:", f"{SAVE_PATH}geokg_{config.input_dim}dim_autoencoder.pt")