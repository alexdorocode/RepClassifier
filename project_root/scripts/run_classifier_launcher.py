import hydra
from omegaconf import DictConfig, OmegaConf
import os
import sys
import torch
from project_root.experiment.classifier_launcher import ClassifierLauncher

@hydra.main(config_path="../config", config_name="config_experiment_test", version_base="1.3")
def main(cfg: DictConfig):
    launcher = ClassifierLauncher(cfg)
    launcher.run()

if __name__ == "__main__":
    main()