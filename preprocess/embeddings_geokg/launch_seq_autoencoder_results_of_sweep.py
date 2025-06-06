from train_seq_autoencoder_wandb import train  # Adjust if your function is in another file

# Define the best configurations for each embedding model
configs = [
    # ESM
    {
        "name": "eternal-sweep-74",
        "activation": "tanh",
        "batch_size": 64,
        "epochs": 20,
        "latent_dim": 128,
        "learning_rate": 0.00361,
        "optimizer": "Adam",
        "emb_model": "ESM",
        "input_dim": 320,
        "emb_path": "../DATASETS/embeddings/sequence_embeddings",
        "save_path": "../DATASETS/embeddings/sequence_embeddings/autoencoder_results"
    },
    {
        "name": "wild-sweep-60",
        "activation": "tanh",
        "batch_size": 64,
        "epochs": 50,
        "latent_dim": 64,
        "learning_rate": 0.00756,
        "optimizer": "Adam",
        "emb_model": "ESM",
        "input_dim": 320,
        "emb_path": "../DATASETS/embeddings/sequence_embeddings",
        "save_path": "../DATASETS/embeddings/sequence_embeddings/autoencoder_results"
    },
    {
        "name": "dutiful-sweep-4",
        "activation": "tanh",
        "batch_size": 32,
        "epochs": 30,
        "latent_dim": 32,
        "learning_rate": 0.00533,
        "optimizer": "Adam",
        "emb_model": "ESM",
        "input_dim": 320,
        "emb_path": "../DATASETS/embeddings/sequence_embeddings",
        "save_path": "../DATASETS/embeddings/sequence_embeddings/autoencoder_results"
    },

    # ProtT5
    {
        "name": "glorious-sweep-75",
        "activation": "tanh",
        "batch_size": 64,
        "epochs": 10,
        "latent_dim": 128,
        "learning_rate": 0.00116,
        "optimizer": "Adam",
        "emb_model": "Prot-T5",
        "input_dim": 1024,
        "emb_path": "../DATASETS/embeddings/sequence_embeddings",
        "save_path": "../DATASETS/embeddings/sequence_embeddings/autoencoder_results"
    },
    {
        "name": "lively-sweep-76",
        "activation": "relu",
        "batch_size": 32,
        "epochs": 50,
        "latent_dim": 64,
        "learning_rate": 0.00061,
        "optimizer": "Adam",
        "emb_model": "Prot-T5",
        "input_dim": 1024,
        "emb_path": "../DATASETS/embeddings/sequence_embeddings",
        "save_path": "../DATASETS/embeddings/sequence_embeddings/autoencoder_results"
    },
    {
        "name": "restful-sweep-34",
        "activation": "relu",
        "batch_size": 64,
        "epochs": 40,
        "latent_dim": 32,
        "learning_rate": 0.00068,
        "optimizer": "Adam",
        "emb_model": "Prot-T5",
        "input_dim": 1024,
        "emb_path": "../DATASETS/embeddings/sequence_embeddings",
        "save_path": "../DATASETS/embeddings/sequence_embeddings/autoencoder_results"
    },

    # ProstT5
    {
        "name": "royal-sweep-23",
        "activation": "tanh",
        "batch_size": 64,
        "epochs": 40,
        "latent_dim": 128,
        "learning_rate": 0.00220,
        "optimizer": "Adam",
        "emb_model": "Prost-T5",
        "input_dim": 1024,
        "emb_path": "../DATASETS/embeddings/sequence_embeddings",
        "save_path": "../DATASETS/embeddings/sequence_embeddings/autoencoder_results"
    },
    {
        "name": "skilled-sweep-90",
        "activation": "tanh",
        "batch_size": 32,
        "epochs": 50,
        "latent_dim": 64,
        "learning_rate": 0.00197,
        "optimizer": "Adam",
        "emb_model": "Prost-T5",
        "input_dim": 1024,
        "emb_path": "../DATASETS/embeddings/sequence_embeddings",
        "save_path": "../DATASETS/embeddings/sequence_embeddings/autoencoder_results"
    },
    {
        "name": "vital-sweep-3",
        "activation": "tanh",
        "batch_size": 32,
        "epochs": 20,
        "latent_dim": 32,
        "learning_rate": 0.00393,
        "optimizer": "Adam",
        "emb_model": "Prost-T5",
        "input_dim": 1024,
        "emb_path": "../DATASETS/embeddings/sequence_embeddings",
        "save_path": "../DATASETS/embeddings/sequence_embeddings/autoencoder_results"
    }
]

# 🚀 Launch training for each configuration
for i, config in enumerate(configs):
    print(f"\n🔧 Training model {i+1}/{len(configs)}: {config['name']}")
    print("⚙️ Configuration:", config)
    train(wandb_run=False, config=config)
