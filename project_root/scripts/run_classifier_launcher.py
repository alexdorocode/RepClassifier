# Final commit – Master’s Thesis by Àlex Domínguez Roig

import hydra
from omegaconf import DictConfig, OmegaConf
import os
import sys
import torch
from project_root.dataset.dataset_config import DatasetConfigReader
from project_root.dataset.dataset_handler import DatasetHandler
from project_root.launchers.classifier_launcher import ClassifierLauncher

@hydra.main(config_path="../config", config_name="config_experiment_test", version_base="1.3")
def main(cfg: DictConfig):

    print("🔍 Initializing ClassifierLauncher...")
    print(f"Using config: {OmegaConf.to_yaml(cfg)}")
    config_reader = DatasetConfigReader(cfg)
    dataset_handler = DatasetHandler(config_reader)
    launcher = ClassifierLauncher(config_reader=config_reader,
                                   dataset_handler=dataset_handler,
                                   zero_shot_test=False,
                                   random_seed=42)
    launcher.run()

if __name__ == "__main__":
    main()