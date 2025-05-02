import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from preprocess.embeddings_geokg.go_autoencoder import GOAutoencoder

import wandb

EMB_PATH = "../DATASETS/embeddings/GeOKG/goa_embedding/"
SAVE_PATH = "../DATASETS/embeddings/GeOKG/autoencoders/"

# ==== Sweep Training Function ====
def train(wandb_run=True, config=None):

    # Initialize wandb
    if wandb_run:
        wandb.init()
        config = dict(wandb.config)  # 👈 Convert wandb.config to a normal dictionary

    # Load data
    emb_name = f"GeOKG_{config['input_dim']}dim.npy"
    emb_path = f"{EMB_PATH}{emb_name}"
    embeddings = np.load(emb_path)
    data = torch.tensor(embeddings, dtype=torch.float32)
    dataset = TensorDataset(data)
    loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)

    # Model
    model = GOAutoencoder(
        input_dim=config['input_dim'],
        latent_dim=config['latent_dim'],
        activation_name=config['activation']
    )
    optimizer = getattr(torch.optim, config['optimizer'])(model.parameters(), lr=config['learning_rate'])
    loss_fn = nn.MSELoss()

    model.train()
    for epoch in range(config['epochs']):
        total_loss = 0
        for batch in loader:
            x = batch[0]
            loss = loss_fn(model(x), x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(loader)
        if wandb_run:
            wandb.log({"epoch": epoch, "loss": avg_loss})
        else:
            print(f"Epoch {epoch+1}/{config['epochs']}, Loss: {avg_loss:.4f}")

    # Save encoder
    save_name = f"geokg_IN_{config['input_dim']}dim_OUT_{config['latent_dim']}dim.pt"
    torch.save(model.state_dict(), f"{SAVE_PATH}{save_name}")
    print(f"Autoencoder GeOKG {config['input_dim']}dim saved to: {SAVE_PATH}{save_name}")
    
    if wandb_run:
        wandb.finish()