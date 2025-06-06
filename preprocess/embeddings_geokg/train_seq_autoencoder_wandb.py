import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from preprocess.embeddings_geokg.go_autoencoder import GOAutoencoder

import wandb
import os
import ast

seq_input_dim = {
    "ESM": 320,
    "Prot-T5": 1024,
    "Prost-T5": 1024
}

# === Sweep Training Function ===
def train(wandb_run=True, config=None):

    if wandb_run:
        wandb.init()
        config = dict(wandb.config)  # wandb.config to dict

    # Load CSV embeddings
    emb_model = config['emb_model']
    EMB_PATH = config['emb_path']
    SAVE_PATH = config['save_path']

    # Check if the save path exists
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)
        print(f"Created directory: {SAVE_PATH}")

    if emb_model == "GeOKG":
        
        # Load data
        emb_name = f"GeOKG_{config['input_dim']}dim.npy"
        emb_path = f"{EMB_PATH}{emb_name}"
        input_dim = config['input_dim']
        
        embeddings = np.load(emb_path)
        data = torch.tensor(embeddings, dtype=torch.float32)
        dataset = TensorDataset(data)
        loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)
        
    
    elif emb_model in ["ESM", "Prot-T5", "Prost-T5"]:
        emb_file = f"{emb_model.lower().replace('-', '').replace(' ', '')}_embeddings.csv"
        emb_path = os.path.join(EMB_PATH, emb_file)
        input_dim = seq_input_dim[emb_model]

        df = pd.read_csv(emb_path)
        
        # Parse the 'esm_embedding' column to convert strings to lists
        df['esm_embedding'] = df['esm_embedding'].apply(ast.literal_eval)

        # Convert the list of embeddings into a numpy array
        embeddings = np.array(df['esm_embedding'].tolist(), dtype=np.float32)

        # Dataset
        data = torch.tensor(embeddings, dtype=torch.float32)
        dataset = TensorDataset(data)
        loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)

    else:
        raise ValueError(f"Unsupported embedding model: {emb_model}")

    # Model
    model = GOAutoencoder(
        input_dim=input_dim,
        latent_dim=config['latent_dim'],
        activation_name=config['activation']
    )

    optimizer = getattr(torch.optim, config['optimizer'])(model.parameters(), lr=config['learning_rate'])
    loss_fn = nn.MSELoss()

    # Training loop
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
    if emb_model == "GeOKG":
        save_name = f"geokg_IN_{config['input_dim']}dim_OUT_{config['latent_dim']}dim.pt"
        torch.save(model.state_dict(), f"{SAVE_PATH}{save_name}")
        print(f"Autoencoder GeOKG {config['input_dim']}dim saved to: {SAVE_PATH}{save_name}")
        
    else:
        filename = f"{emb_model.replace('-', '').replace(' ', '')}_{config['latent_dim']}dim.pt"
        model_path = os.path.join(SAVE_PATH, filename)
        torch.save(model.state_dict(), model_path)
        print(f"Autoencoder {emb_model} saved to: {model_path}")

    if wandb_run:
        wandb.finish()
