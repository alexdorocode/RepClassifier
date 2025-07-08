# 🧬 Protein Classification Workspace

> This repository contains the modular and experimental workspace developed for my Master’s Thesis in Bioinformatics. It implements a flexible and extensible pipeline for classifying proteins using sequence-based, structural, and functional features.

---

## 📌 Project Context

This project was designed with a **limited development scope**, prioritizing **modular and reusable code** for future extensions. While some paths reference external files, required data has been relocated to the internal `datasets/` and `csv_embeddings/` folders to facilitate evaluation.

---

## 🗂️ Project Structure (Main Folders)

```bash
.
├── csv_embeddings/         # Precomputed sequence and GO embeddings (autoencoded and raw)
├── datasets/               # Labeled protein datasets and metadata
├── preprocess/             # Notebooks and scripts for preprocessing and embedding generation
├── project_root/           # Core pipeline: dataset handling, training, models, scripts, explainability
├── results/                # Final evaluation and agreement prediction CSVs
├── tests/                  # Unit tests for key modules
├── Doxyfile                # Doxygen configuration for documentation generation
├── README.md
├── requirements.txt
└── environment.yml

## 📂 Folder Highlights
🔬 csv_embeddings/

Precomputed embedding files including:

    Raw embeddings: esm_embeddings.csv, prot_embeddings.csv, prostT5_embeddings.csv

    Autoencoded GO term embeddings by dimension, pooling, and category (C, F, P)

    Autoencoded sequence embeddings (ProtT5, ProstT5, ESM) for multiple compression levels

🧬 datasets/

    Includes raw, merged, and labeled datasets such as moonprot_dataset.csv, moondb_dataset.csv, and predictor_dataset.csv

    MBL-scores and UniProt mappings used to define moonlighting candidates and controls

🧪 preprocess/

    analyses/: Notebooks exploring data distributions, GO terms, sequence representations

    embeddings_geokg/: Scripts for embedding extraction, autoencoder training, and sweeps via W&B

🧠 project_root/

    config/: Organized configs for experiments, sweeps, tunable params, and paths

    dataset/: Dataset loaders, handlers, and feature extraction wrappers

    experiment/: YAML fork expander and experiment config management

    models/: Classifiers (MLP, etc.) and autoencoders

    scripts/: Full experiment runners, result processors, agreement calculators

    explainability/: SHAP, agreement visualization, PCA

    training/: W&B tracking, training loop utilities

    utils/: General helper functions (visualization, config, feature processing)

📊 results/

Final .csv outputs of trained models and ensemble agreement experiments (per model type).
📜 Script Overview
🚀 Launcher Scripts

Located in project_root/scripts/:

    run_classifier_launcher.py, run_experiment_launcher.py: Full pipeline runners

    run_wandb_agent_phase*.py: Sweep execution using Weights & Biases

    run_final_evaluation.py, run_prediction_agreement.py: Evaluation and ensemble scoring

📈 Model Agreement & Visualization

    Scripts in model_aggreement/ generate bar plots, stacked plots, incorrect agreement visualizations, etc.

🧪 Embedding Generation

    Found in preprocess/embeddings_geokg/

        train_go_autoencoder_wandb.py, train_seq_autoencoder_wandb.py: Autoencoder training

        launch_*_sweeps.py, launch_*_results_of_sweep.py: W&B-managed sweeps

        go_embeddings_per_protein.py: GO term aggregation and per-protein embedding

⚙️ Setup Instructions
Option A – pip

pip install -r requirements.txt

Option B – conda

conda env create -f environment.yml
conda activate protein-classifier-env

🚀 Basic Usage
Run a classifier

python project_root/scripts/run_classifier_launcher.py --config project_root/config/config_classifier_base.yaml

Run a zero-shot or agreement evaluation

python project_root/scripts/run_final_evaluation.py

Launch hyperparameter sweep

python project_root/scripts/run_sweeps_phase_2.py

📘 Documentation

To generate full code documentation with Doxygen:

doxygen Doxyfile

Then open the generated docs/html/index.html in a browser.

You may optionally host the HTML docs via GitHub Pages and link here.
🧪 Testing

Unit tests for core components:

pytest tests/

Test files:

    test_model_trainer.py, test_model_explainability.py

    test_wrapped_dataset.py, test_data_visualizer.py

📌 Notes for Reviewers

    External paths have been minimized; required data files are placed in datasets/ and csv_embeddings/

    Results are reproducible through launcher scripts and defined config files

    The pipeline uses Weights & Biases for sweep management and result tracking

👤 Author

Àlex Domínguez Roig
Master’s Thesis – Bioinformatics
Universitat Autònoma de Barcelona