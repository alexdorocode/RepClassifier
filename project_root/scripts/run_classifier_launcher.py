import hydra
from omegaconf import DictConfig, OmegaConf
import os
import sys
import torch
from project_root.experiment.classifier_launcher import ClassifierLauncher

@hydra.main(config_path="../config", config_name="config_experiment_test", version_base="1.3")
def main(cfg: DictConfig):

    print("🔍 Initializing ClassifierLauncher...")
    print(f"Using config: {OmegaConf.to_yaml(cfg)}")

    launcher = ClassifierLauncher(cfg)
    launcher.run()

if __name__ == "__main__":
    main()