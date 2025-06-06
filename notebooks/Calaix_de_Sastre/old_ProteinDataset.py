import os
from matplotlib import pyplot as plt
import torch
import seaborn as sns # type: ignore
import numpy as np
import re
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.decomposition import PCA # type: ignore
from sklearn.manifold import TSNE # type: ignore
from sklearn.cluster import KMeans # type: ignore
import gc

from transformers import T5Tokenizer, T5EncoderModel
from torch.utils.data import Dataset

class ProteinDataset(Dataset):
    def __init__(self, dataframe, 
                 target_column='Class', id_column='UniProt IDs',
                 model_name='Rostlab/ProstT5', 
                 device='cpu', mode='seq', initial_batch_size=8, 
                 embeddings=None, attention_weights=None, save_computed=False, 
                 id_embedding=None, id_attention_weights=None,
                 solve_all_inconsitence=False, save_path="./OUTPUTS/"):
        
        self.save_path = save_path
        os.makedirs(self.save_path, exist_ok=True)
        
        self.embeddings_file = None
        self.attention_weights_file = None
        self.embeddings_produced = False
        self.inconsistency_solved = False

        is_inconsitence = self.ensure_consistency(dataframe, embeddings, attention_weights, target_column, id_column, id_embedding, id_attention_weights)
        
        if is_inconsitence and solve_all_inconsitence:
            dataframe, embeddings, attention_weights = self.solve_inconsitence(dataframe, embeddings, attention_weights, id_column, id_embedding, id_attention_weights)
            self.inconsistency_solved = True

        self.dataframe = dataframe
        self.embeddings = embeddings
        self.attention_weights = attention_weights
        self.labels = dataframe[target_column].tolist()
        self.id = dataframe[id_column].tolist()
        self.tokenizer = T5Tokenizer.from_pretrained(model_name, do_lower_case=False)
        self.model = T5EncoderModel.from_pretrained(model_name).to(device)
        self.model.eval()
        self.device = device
        self.mode = mode
        self.save_computed = save_computed

        if self.embeddings is None or self.attention_weights is None:
            sequences = dataframe[mode].tolist()
            self.embeddings, self.attention_weights = self.get_prott5_embeddings(sequences, initial_batch_size)
            self.embeddings_produced = True
            if self.save_computed:
                self.save_embeddings_and_attention_weights("prott5_embeddings.npy", "prott5_attention_weights.npy")

        self.display_report()

    def ensure_consistency(self, dataframe, embeddings, attention_weights, solve_all_inconsitence,
                            target_column, id_column, id_embedding, id_attention_weights):
        
        if self.check_columns_exist(dataframe, target_column, id_column):
            raise ValueError(f"Dataframe must contain the columns '{target_column}' and '{id_column}'.")

        if self.check_duplicates(dataframe) and solve_all_inconsitence:
            dataframe.drop_duplicates(inplace=True)

        if self.check_lengths(dataframe, embeddings, attention_weights) and solve_all_inconsitence:
            self.solve_inconsitence(dataframe, embeddings, attention_weights, id_column, id_embedding, id_attention_weights)

        return self.check_ids_consistency(dataframe, embeddings, attention_weights, id_column, id_embedding, id_attention_weights)

    def check_lengths(self, dataframe, embeddings, attention_weights):
        if len(dataframe) != len(embeddings) or len(dataframe) != len(attention_weights):
            print("Dataframe, embeddings, and attention_weights must have the same length.")

    def check_duplicates(self, dataframe):
        duplicates = dataframe.duplicated()
        if duplicates.any():
            print("Warning: The dataframe contains duplicates.")
            print(dataframe[duplicates])
            print("Use the remove_duplicates function to remove duplicates.")

    def check_columns_exist(self, dataframe, target_column, id_column):
        if target_column not in dataframe.columns or id_column not in dataframe.columns:
            raise ValueError(f"Dataframe must contain the columns '{target_column}' and '{id_column}'.")

    def check_ids_consistency(self, dataframe, embeddings, attention_weights, id_column, id_embedding, id_attention_weights):
        if id_embedding is None or id_attention_weights is None:
            raise ValueError("id_embedding and id_attention_weights must be provided.")
        
        df_ids = set(dataframe[id_column])
        emb_ids = set(embeddings[id_embedding])
        attn_ids = set(attention_weights[id_attention_weights])
        
        if df_ids != emb_ids or df_ids != attn_ids:
            print("Warning: Inconsistency found between dataframe, embeddings, and attention_weights IDs.")
            return True
        return False

    def solve_inconsitence(self, dataframe, embeddings, attention_weights, id_column, id_embedding, id_attention_weights):
        df_ids = set(dataframe[id_column])
        emb_ids = set(embeddings[id_embedding])
        attn_ids = set(attention_weights[id_attention_weights])
        
        common_ids = df_ids & emb_ids & attn_ids
        
        dataframe = dataframe[dataframe[id_column].isin(common_ids)]
        embeddings = {k: v for k, v in embeddings.items() if k in common_ids}
        attention_weights = {k: v for k, v in attention_weights.items() if k in common_ids}
        
        return dataframe, embeddings, attention_weights

    def display_report(self):
        print("ProteinDataset Report:")
        print(f"Number of samples: {len(self.dataframe)}")
        print(f"Number of embeddings: {len(self.embeddings)}")
        print(f"Number of attention weights: {len(self.attention_weights)}")
        print(f"Target column: {self.labels}")
        print(f"ID column: {self.id}")
        print(f"Save path: {self.save_path}")
        print(f"Inconsistency solved: {self.inconsistency_solved}")
        print(f"Embeddings produced: {self.embeddings_produced}")
        if not self.embeddings_produced:
            print(f"Embeddings file: {self.embeddings_file}")
            print(f"Attention weights file: {self.attention_weights_file}")

    def preprocess_sequence(self, sequence: str) -> str:
        """Preprocess a protein sequence for tokenization."""
        sequence = re.sub(r"[UZOB]", "X", sequence)
        sequence = " ".join(list(sequence))
        if sequence.isupper():
            return "<AA2fold> " + sequence
        else:
            return "<fold2AA> " + sequence

    def process_batch(self, batch):
        """Encodes sequences and extracts embeddings in full precision (FP32)."""
        batch = [self.preprocess_sequence(seq) for seq in batch]
        ids = self.tokenizer.batch_encode_plus(batch, add_special_tokens=True, padding="longest", truncation=True, return_tensors='pt').to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids=ids.input_ids, attention_mask=ids.attention_mask, output_attentions=True)
            embedding_repr = outputs.last_hidden_state
            batch_embeddings = embedding_repr[:, 1:-1, :].mean(dim=1).cpu().numpy()
            batch_attention_weights = outputs.attentions[-1].mean(dim=1).cpu().numpy()  # Use the last attention layer and average over heads

        del ids, embedding_repr, outputs
        torch.cuda.empty_cache()
        gc.collect()

        return batch_embeddings, batch_attention_weights

    def process_single_sequence_fp16(self, sequence):
        """Processes a single sequence using FP16 precision to save memory."""
        batch = [self.preprocess_sequence(sequence)]
        ids = self.tokenizer.batch_encode_plus(batch, add_special_tokens=True, padding="longest", truncation=True, return_tensors='pt').to(self.device)

        with torch.no_grad():
            self.model.half()  # Switch to FP16
            outputs = self.model(input_ids=ids.input_ids, attention_mask=ids.attention_mask, output_attentions=True)
            embedding_repr = outputs.last_hidden_state
            batch_embedding = embedding_repr[:, 1:-1, :].mean(dim=1).cpu().numpy()
            batch_attention_layer = outputs.attentions[-1].mean(dim=1).cpu().numpy()  # Use the last attention layer and average over heads
            self.model.float()  # Switch back to FP32

        del ids, embedding_repr, outputs
        torch.cuda.empty_cache()
        gc.collect()

        return batch_embedding[0], batch_attention_layer[0]

    def get_prott5_embeddings(self, sequences, initial_batch_size=8):
        batch_size = initial_batch_size
        embeddings = [None] * len(sequences)  # Initialize with None to maintain order
        attention_weights = [None] * len(sequences)  # Initialize with None to maintain order
        i = 0

        while i < len(sequences):
            print(f"Processing sequence {i+1}/{len(sequences)}")
            try:
                torch.cuda.empty_cache()
                batch = sequences[i:i+batch_size]
                batch_embeddings, batch_attention_weights = self.process_batch(batch)

                # Save embeddings and attention layers in the correct positions
                embeddings[i:i+batch_size] = batch_embeddings
                attention_weights[i:i+batch_size] = batch_attention_weights
                i += batch_size  # Move to next batch

                # Restore batch size after a successful run
                if batch_size < initial_batch_size:
                    print(f"✅ Restoring batch size to {initial_batch_size}.")
                    batch_size = initial_batch_size

            except torch.cuda.OutOfMemoryError:
                if batch_size > 1:
                    print(f"⚠️ Reducing batch size from {batch_size} to {max(1, batch_size // 2)} due to OOM.")
                    batch_size = max(1, batch_size // 2)  # Reduce batch size but keep at least 1
                else:
                    print(f"⚠️ Running sequence {i+1} in FP16 due to OOM.")
                    torch.cuda.empty_cache()

                    try:
                        embedding, attention_layer = self.process_single_sequence_fp16(sequences[i])
                        embeddings[i] = embedding  # Save even the fallback embedding
                        attention_weights[i] = attention_layer  # Save even the fallback attention layer
                        i += 1  # Move to next sequence

                    except torch.cuda.OutOfMemoryError:
                        print(f"❌ Skipping sequence {i+1} (too large for FP16).")
                        i += 1  # Still move forward, but keep None as a placeholder

        return embeddings, attention_weights

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        # Convertir de numpy a tensor cuando se accede.
        return (torch.tensor(self.embeddings[idx]), torch.tensor(self.attention_weights[idx])), torch.tensor(self.labels[idx], dtype=torch.float)

    def save_embeddings_and_attention_weights(self, embeddings_file, attention_weights_file):
        embeddings_path = os.path.join(self.save_path, embeddings_file)
        attention_weights_path = os.path.join(self.save_path, attention_weights_file)
        np.save(embeddings_path, {"UniProt IDs": self.id, "embeddings": self.embeddings})
        np.save(attention_weights_path, {"UniProt IDs": self.id, "attention_weights": self.attention_weights})
        self.embeddings_file = embeddings_path
        self.attention_weights_file = attention_weights_path

    def dropna(self):
        """Drop rows with NaN or None values in embeddings or attention layers."""
        valid_indices = [i for i, (emb, attn) in enumerate(zip(self.embeddings, self.attention_weights)) if emb is not None and attn is not None]
        self.embeddings = [self.embeddings[i] for i in valid_indices]
        self.attention_weights = [self.attention_weights[i] for i in valid_indices]
        self.labels = [self.labels[i] for i in valid_indices]
        self.id = [self.id[i] for i in valid_indices]

    def split_train_test(self, test_size=0.2):
        """Split the dataset into train and test sets."""
        train_indices, test_indices = train_test_split(range(len(self.dataframe)), test_size=test_size, random_state=42, stratify=self.labels)
        train_dataset = torch.utils.data.Subset(self, train_indices)
        test_dataset = torch.utils.data.Subset(self, test_indices)
        return train_dataset, test_dataset

    def plot_tsne(self, attribute='Class', combined=False):
        """Plot TSNE for embeddings, attention_weights, or both combined."""
        if combined:
            data = np.concatenate([self.embeddings, self.attention_weights], axis=1)
        else:
            data = self.embeddings

        # Ensure data is a NumPy array
        data = np.array(data)

        # Check the shape of the data
        print(f"Data shape: {data.shape}")

        tsne = TSNE(n_components=2, random_state=42)
        tsne_results = tsne.fit_transform(data)

        plt.figure(figsize=(16, 10))
        sns.scatterplot(
            x=tsne_results[:, 0], y=tsne_results[:, 1],
            hue=self.dataframe[attribute],
            palette=sns.color_palette("hsv", len(self.dataframe[attribute].unique())),
            legend="full",
            alpha=0.3
        )
        plt.title(f'TSNE plot for {attribute}')
        plt.show()

    def plot_pca(self, attribute='Class', combined=False):
        """Plot PCA for embeddings, attention_weights, or both combined."""
        if combined:
            data = np.concatenate([self.embeddings, self.attention_weights], axis=1)
        else:
            data = self.embeddings

        # Ensure data is a NumPy array
        data = np.array(data)

        pca = PCA(n_components=2)
        pca_results = pca.fit_transform(data)

        plt.figure(figsize=(16, 10))
        sns.scatterplot(
            x=pca_results[:, 0], y=pca_results[:, 1],
            hue=self.dataframe[attribute],
            palette=sns.color_palette("hsv", len(self.dataframe[attribute].unique())),
            legend="full",
            alpha=0.3
        )
        plt.title(f'PCA plot for {attribute}')
        plt.show()

    def plot_pca_variance(self, combined=False):
        """Plot PCA variance explained for embeddings, attention_weights, or both combined."""
        if combined:
            data = np.concatenate([self.embeddings, self.attention_weights], axis=1)
        else:
            data = self.embeddings

        # Ensure data is a NumPy array
        data = np.array(data)

        pca = PCA()
        pca.fit(data)
        explained_variance = pca.explained_variance_ratio_

        plt.figure(figsize=(16, 10))
        plt.plot(np.cumsum(explained_variance))
        plt.xlabel('Number of Components')
        plt.ylabel('Variance Explained')
        plt.title('PCA Variance Explained')
        plt.show()

    def plot_kmeans(self, n_clusters=3, attribute='Class', combined=False):
        """Plot k-means clustering for embeddings, attention_weights, or both combined."""
        if combined:
            data = np.concatenate([self.embeddings, self.attention_weights], axis=1)
        else:
            data = self.embeddings

        # Ensure data is a NumPy array
        data = np.array(data)

        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(data)
        clusters = kmeans.labels_

        plt.figure(figsize=(16, 10))
        sns.scatterplot(
            x=data[:, 0], y=data[:, 1],
            hue=clusters,
            palette=sns.color_palette("hsv", n_clusters),
            legend="full",
            alpha=0.3
        )
        plt.title(f'K-means clustering with {n_clusters} clusters')
        plt.show()
