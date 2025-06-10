import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
from project_root.experiment.experiment_launcher import ExperimentLauncher
SAVE_CONFIGS_FOLDER = "./project_root/config/sweeps_configs/"
PATH_RESULTS_PHASE_1_EMBEDDINGS_CONFIG = "./project_root/config/phase_results/for_phase_3_embedding_configs.yaml"
PATH_RESULTS_PHASE_2_MODEL_CONFIG = "./project_root/config/phase_results/for_phase_3_model_configs.yaml"


@hydra.main(config_path="/Users/alexdominguez/Documents/GitHub/TFM/RepClassifier/project_root/config/", config_name="config_experiment_base", version_base="1.3")
def main(cfg: DictConfig):
    print("🔍 Initializing ExperimentConfigHandler...")

    launcher = ExperimentLauncher(cfg, phase_1_result_path=PATH_RESULTS_PHASE_1_EMBEDDINGS_CONFIG, 
                                  phase_2_result_path=PATH_RESULTS_PHASE_2_MODEL_CONFIG)
    launcher.run_phase_3_sweep(model_name="xgb")  # or 'rf', 'svm', etc.

if __name__ == "__main__":
    main()