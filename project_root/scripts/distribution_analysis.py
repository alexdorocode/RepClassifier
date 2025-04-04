import os
import sys
import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore
from sklearn.decomposition import PCA # type: ignore
from sklearn.manifold import TSNE # type: ignore
from sklearn.cluster import KMeans # type: ignore
from project_root.dataset.dataset_loader import DatasetLoader
from project_root.dataset.representation_dataset import RepresentationDataset
from project_root.dataset.wrapped_representation_dataset import WrappedRepresentationDataset

def plot_2d(data, labels, title, save_path):
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=data[:, 0], y=data[:, 1], hue=labels, palette={True: 'blue', False: 'red'}, alpha=0.7)
    plt.title(title)
    plt.savefig(save_path)
    plt.close()

def create_plots(data, labels, data_name, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # PCA
    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(data)
    plot_2d(pca_data, labels, f'PCA 2D - {data_name}', os.path.join(output_dir, f'pca_{data_name}.png'))

    # TSNE
    tsne = TSNE(n_components=2, random_state=42)
    tsne_data = tsne.fit_transform(data)
    plot_2d(tsne_data, labels, f'TSNE 2D - {data_name}', os.path.join(output_dir, f'tsne_{data_name}.png'))

    # KMeans
    kmeans = KMeans(n_clusters=2, random_state=42)
    kmeans_data = kmeans.fit_transform(data)
    plot_2d(kmeans_data, labels, f'KMeans 2D - {data_name}', os.path.join(output_dir, f'kmeans_{data_name}.png'))

def main(dataset_path):
    # Load data
    loader = DatasetLoader(dataset_path)
    df = loader.load_dataframe()
    embeddings, attention_weights = loader.load_embeddings_and_attention()

    # Create RepresentationDataset instance
    representation_dataset = RepresentationDataset(df, embeddings, attention_weights, solve_inconsistencies=True)

    # Create WrappedRepresentationDataset instance
    wrapped_dataset = WrappedRepresentationDataset(
        dataset=representation_dataset,
        process_attention_weights=True,
        reduce_method='pca',
        pca_method='threshold',
        threshold=0.95,
        random_projection_dim=50
    )

    # Select data
    data_emb = wrapped_dataset.select_data(embedding=True)
    data_att = wrapped_dataset.select_data(attention_weights=True)
    data_go_cc = wrapped_dataset.select_data(one_hot_columns=['GO CC Terms'])
    data_go_mf = wrapped_dataset.select_data(one_hot_columns=['GO MF Terms'])
    data_max_mbl_cc = wrapped_dataset.select_data(additional_columns=['Max_MBL_CC'])
    data_max_mbl_mf = wrapped_dataset.select_data(additional_columns=['Max_MBL_MF'])

    # Get labels
    labels = np.array(representation_dataset.get_labels())

    # Define output directory
    output_base_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'images', 'distribution_analysis')

    # Create plots for each data combination
    create_plots(data_emb, labels, 'embeddings', os.path.join(output_base_dir, 'embeddings'))
    create_plots(data_att, labels, 'attention_weights', os.path.join(output_base_dir, 'attention_weights'))
    create_plots(data_go_cc, labels, 'go_cc_terms', os.path.join(output_base_dir, 'go_cc_terms'))
    create_plots(data_go_mf, labels, 'go_mf_terms', os.path.join(output_base_dir, 'go_mf_terms'))
    create_plots(data_max_mbl_cc, labels, 'max_mbl_cc', os.path.join(output_base_dir, 'max_mbl_cc'))
    create_plots(data_max_mbl_mf, labels, 'max_mbl_mf', os.path.join(output_base_dir, 'max_mbl_mf'))

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <dataset_path>")
        sys.exit(1)
    dataset_path = sys.argv[1]
    main(dataset_path)