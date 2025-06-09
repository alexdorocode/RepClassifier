import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
from project_root.experiment.experiment_config_handler import ExperimentConfigHandler
from project_root.experiment.classifier_launcher import ClassifierLauncher

class ExperimentLauncher:
    def __init__(self, 
                 cfg: DictConfig, 
                 phase_1_result_path: str = None,
                 phase_2_result_path: str = None,
                 phase_3_result_path: str = None,
                 ):
        
        self.cfg = cfg

        self.handler = ExperimentConfigHandler(
            base_config_path=cfg.get('base_path', {}),
            model_fork_paths=cfg.get('model_fork_paths', []),
            embedding_fork_path=cfg.get('embedding_fork_path', {}),
            feature_fork_path=cfg.get('feature_fork_path', {}),
            training_fork_path=cfg.get('training_fork_path', {}),
            phase_1_result_path=phase_1_result_path,
            phase_2_result_path=phase_2_result_path,
            phase_3_result_path=phase_3_result_path,
        )

    def run_phase_1_sweep(self, model_name: str):
        print("🔍 Preparing sweep configuration for Phase 1...")
        self.handler.set_sweeps_config_phase_1(model=model_name)
        base_cfg, final_classifier_configs = self.handler.get_classifier_ready_configs(self.cfg)
        
        run = wandb.init()
        config_index = wandb.config.get("config_index")
        cv_folds = wandb.config["training_cv_folds"]
        cross_val_balance = wandb.config["training_cross_val_balance"]

        assert 0 <= config_index < len(final_classifier_configs), "Index out of range."

        # Overwrite the training config with the one from W&B
        training_config = {
            "cv_folds": cv_folds,
            "cross_val_balance": cross_val_balance
        }

        base_cfg, final_classifier_configs = self.handler.get_classifier_ready_configs(self.cfg)
        cfg_dict = final_classifier_configs[config_index]
        cfg_custom = OmegaConf.create(cfg_dict)
        cfg_custom = OmegaConf.merge(base_cfg, cfg_custom)

        cfg_custom['classifier_definition']['training']['cv_folds'] = training_config['cv_folds']
        cfg_custom['classifier_definition']['training']['cross_val_balance'] = training_config['cross_val_balance']

        print(f"🔍 Running config index {OmegaConf.to_yaml(cfg_custom)}")

        seed = 42 + config_index * training_config['cv_folds'] * (2 if cross_val_balance == 'organism' else 1)
        print(f"🔍 Running config index {config_index} with seed {seed}:")
        launcher = ClassifierLauncher(cfg_custom, random_seed=seed)
        launcher.run()

        wandb.finish()
        
    def run_phase_2_sweep(self, model_name: str):
        print("🔍 Preparing sweep configuration for Phase 2...")
    
        run = wandb.init()

        classifier_config = self.handler.get_sweeps_config_phase_2(
            model=model_name,
            classifier_config=self.cfg['classifier_definition'],
            sweep_config=wandb.config
        )

        self.cfg['classifier_definition'] = classifier_config
    
        seed = self._calculate_seed_from_config(wandb.config)
        print(f"🔍 Running config with seed {seed}:")
        launcher = ClassifierLauncher(self.cfg, random_seed=seed)
        launcher.run()
    
        wandb.finish()
        
    def _calculate_seed_from_config(self, config):
        
        ints = []
        floats = []
        for key in config:
            if isinstance(config[key], float):
                floats.append(config[key])
            elif isinstance(config[key], int):
                ints.append(config[key])
    
        print(f"Seed calculation ints: {ints}, floats: {floats}")
    
        seed = 42
        for i in ints:
            seed *= i if i != 0 else 1
        for f in floats:
            seed *= int(f * 100) if f != 0.0 else 1
        return abs(seed) % (2**31)
    