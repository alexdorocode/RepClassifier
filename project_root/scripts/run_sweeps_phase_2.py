import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
from project_root.experiment.experiment_launcher import ExperimentLauncher
SAVE_CONFIGS_FOLDER = "./project_root/config/sweeps_configs/"
PATH_RESULTS_PHASE_1_EMBEDDINGS_CONFIG = "./project_root/config/phase_results/phase_1_embedding_configs.yaml"


@hydra.main(config_path="/Users/alexdominguez/Documents/GitHub/TFM/RepClassifier/project_root/config/", config_name="config_experiment_base", version_base="1.3")
def main(cfg: DictConfig):
    print("🔍 Initializing ExperimentConfigHandler...")

    launcher = ExperimentLauncher(cfg, phase_1_result_path=PATH_RESULTS_PHASE_1_EMBEDDINGS_CONFIG)
    launcher.run_phase_1_sweep(model_name="lr")  # or 'rf', 'svm', etc.

if __name__ == "__main__":
    main()