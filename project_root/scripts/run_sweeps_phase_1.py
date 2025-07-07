import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
from project_root.dataset.dataset_config import DatasetConfigReader
from project_root.launchers.classifier_launcher import ClassifierLauncher
from project_root.experiment.experiment_config_handler import ExperimentConfigHandler

SAVE_CONFIGS_FOLDER = "./project_root/config/sweeps_configs/"

@hydra.main(config_path="/Users/alexdominguez/Documents/GitHub/TFM/RepClassifier/project_root/config/", config_name="config_experiment_base", version_base="1.3")
def main(cfg: DictConfig):
    print("🔍 Initializing ExperimentConfigHandler...")
    handler = ExperimentConfigHandler(
        base_config_path=cfg.get('base_path', {}),
        model_fork_paths=cfg.get('model_fork_paths', []),
        embedding_fork_path=cfg.get('embedding_fork_path', {}),
        feature_fork_path=cfg.get('feature_fork_path', {}),
        training_fork_path=cfg.get('training_fork_path', {}),
    )

    # Phase 1 sweep setup
    # Values used: 'lr', 'xgb', 'svm', 'knn', 'rf', 'mlp'
    handler.set_sweeps_config_phase_1(model='mlp')
    base_cfg, final_classifier_configs = handler.get_classifier_ready_configs(cfg)

    # Use W&B to select config index
    run = wandb.init()
    config_index = wandb.config.get("config_index")
    assert 0 <= config_index < len(final_classifier_configs), "Index out of range."

    # Build and run selected config
    cfg_dict = final_classifier_configs[config_index]
    cfg_custom = OmegaConf.create(cfg_dict)
    cfg_custom = OmegaConf.merge(base_cfg, cfg_custom)
    seed = 42 + config_index  # stable and reproducible
    print(f"🔍 Running config index {config_index} with seed {seed}:")
    launcher = ClassifierLauncher(cfg_custom, random_seed=seed)
    launcher.run()

    wandb.finish()

if __name__ == "__main__":
    main()
