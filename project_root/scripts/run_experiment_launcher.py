import hydra
from omegaconf import DictConfig, OmegaConf
from project_root.dataset.dataset_config import DatasetConfigReader
from project_root.experiment.classifier_launcher import ClassifierLauncher
from project_root.experiment.experiment_config_handler import ExperimentConfigHandler

SAVE_CONFIGS_FOLDER = "./project_root/config/sweeps_configs/"

@hydra.main(config_path="../config", config_name="config_experiment_base", version_base="1.3")
def main(cfg: DictConfig):
    print("🔍 Initializing ExperimentConfigHandler...")
    # print(f"Using config: {OmegaConf.to_yaml(cfg)}")
    handler = ExperimentConfigHandler(
        base_config_path=cfg.get('base_path', {}),
        model_fork_paths=cfg.get('model_fork_paths', []),
        embedding_fork_path=cfg.get('embedding_fork_path', {}),
        feature_fork_path=cfg.get('feature_fork_path', {}),
        training_fork_path=cfg.get('training_fork_path', {}),
    )

    # First time: expand and save
    handler.set_sweeps_config_phase_1(model='mlp')

    # print(f"✅ Generated the following configurations with the function get_sweeps_config_phase_1  \n {OmegaConf.to_yaml(handler.get_configs()[0])}")

    base_cfg, final_classifier_configs = handler.get_classifier_ready_configs(cfg)

    # print(f"✅ Generated the following classifier configurations with the function get_classifier_ready_configs  \n {OmegaConf.to_yaml(final_classifier_configs[0])}")

    # Use configs with your launcher
    # handler.save_to_file(SAVE_CONFIGS_FOLDER + "sweeps_config_phase_1.json")

    for cfg_dict in final_classifier_configs[:2]:
        cfg_custom = OmegaConf.create(cfg_dict)
        cfg_custom = OmegaConf.merge(base_cfg, cfg_custom)
        # print(f"🔧 Running with config: {OmegaConf.to_yaml(cfg_custom)}")
        launcher = ClassifierLauncher(cfg_custom, random_seed=42)
        launcher.run()


if __name__ == "__main__":
    main()