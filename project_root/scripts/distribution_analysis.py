import os
import sys
import numpy as np  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import seaborn as sns  # type: ignore
from datetime import datetime
from sklearn.decomposition import PCA  # type: ignore
from sklearn.manifold import TSNE  # type: ignore
from sklearn.cluster import KMeans  # type: ignore
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


def main(dataset_path, analysis_title):
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
        random_projection_dim=1000,
        attributes_to_one_hot=['GO CC Terms', 'GO MF Terms'],
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
    date_str = datetime.now().strftime("%Y-%m-%d")
    output_base_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        'images',
        'distribution_analysis',
        f"{date_str}_{analysis_title}"
    )

    # Create plots for each data combination
    combinations = {
        'embeddings': data_emb,
        'attention_weights': data_att,
        'go_cc_terms': data_go_cc,
        'go_mf_terms': data_go_mf,
        'max_mbl_cc': data_max_mbl_cc,
        'max_mbl_mf': data_max_mbl_mf,
        'embedding_attention': np.hstack((data_emb, data_att)),
        'go_terms': np.hstack((data_go_cc, data_go_mf)),
        'max_mbl': np.hstack((data_max_mbl_cc, data_max_mbl_mf)),
        'go_terms_max_mbl': np.hstack((data_go_cc, data_go_mf, data_max_mbl_cc, data_max_mbl_mf)),
        'embedding_go_terms_max_mbl': np.hstack((data_emb, data_go_cc, data_go_mf, data_max_mbl_cc, data_max_mbl_mf)),
        'attention_go_terms_max_mbl': np.hstack((data_att, data_go_cc, data_go_mf, data_max_mbl_cc, data_max_mbl_mf)),
        'all_combined': np.hstack((data_emb, data_att, data_go_cc, data_go_mf, data_max_mbl_cc, data_max_mbl_mf))
    }

    for combination_name, combination_data in combinations.items():
        output_dir = os.path.join(output_base_dir, combination_name)
        create_plots(combination_data, labels, combination_name, output_dir)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage:         PYTHONPATH=$(pwd) OMP_NUM_THREADS=64 MKL_NUM_THREADS=64 OPENBLAS_NUM_THREADS=64 python ./project_root/scripts/distribution_analysis.py ../DATASETS/ embedding_mean")
        sys.exit(1)
    dataset_path = sys.argv[1]
    analysis_title = sys.argv[2]
    main(dataset_path, analysis_title)