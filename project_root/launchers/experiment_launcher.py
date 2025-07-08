# Final commit – Master’s Thesis by Àlex Domínguez Roig

from omegaconf import DictConfig, OmegaConf # type: ignore
import wandb
from project_root.experiment.experiment_config_handler import ExperimentConfigHandler
from project_root.launchers.classifier_launcher import ClassifierLauncher
from project_root.dataset.dataset_config import DatasetConfigReader
from project_root.dataset.dataset_handler import DatasetHandler

class ExperimentLauncher:
    """
    Orchestrates experiment sweeps and final evaluations for all phases.
    Handles configuration, dataset, and classifier launching for each phase.
    """

    def __init__(self, 
                 cfg: DictConfig, 
                 phase_1_result_path: str = None,
                 phase_2_result_path: str = None,
                 phase_3_result_path: str = None,
                 ):
        """
        Initialize the ExperimentLauncher.

        :param cfg: Hydra/OmegaConf configuration object
        :param phase_1_result_path: Optional path to phase 1 results
        :param phase_2_result_path: Optional path to phase 2 results
        :param phase_3_result_path: Optional path to phase 3 results
        """
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
        """
        Run a W&B sweep for phase 1 (model/embedding/feature/training search).

        :param model_name: Name of the model to sweep
        """
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

        seed = self._calculate_seed_from_config({
            'config_index': config_index,
            'training_cv_folds': training_config['cv_folds'],
            'training_cross_val_balance': training_config['cross_val_balance']
        }, phase_num = 1)
        print(f"🔍 Running config index {config_index} with seed {seed}:")
        config_reader = DatasetConfigReader(self.cfg)
        dataset_handler = DatasetHandler(config_reader)
        launcher = ClassifierLauncher(config_reader, dataset_handler, random_seed=seed)
        launcher.run()

        wandb.finish()
        
    def run_phase_2_sweep(self, model_name: str):
        """
        Run a W&B sweep for phase 2 (model/embedding/feature/training search).

        :param model_name: Name of the model to sweep
        """
        print("🔍 Preparing sweep configuration for Phase 2...")
    
        run = wandb.init()

        classifier_config = self.handler.get_sweeps_config_phase_2(
            model=model_name,
            classifier_config=self.cfg['classifier_definition'],
            sweep_config=wandb.config
        )

        self.cfg['classifier_definition'] = classifier_config
    
        seed = self._calculate_seed_from_config(wandb.config, phase_num=2)
        print(f"🔍 Running config with seed {seed}:")
        
        config_reader = DatasetConfigReader(self.cfg)
        dataset_handler = DatasetHandler(config_reader)
        launcher = ClassifierLauncher(config_reader, dataset_handler, random_seed=seed)
        launcher.run()
    
        wandb.finish()
        
    def run_phase_3_sweep(self, model_name: str):
        """
        Run a W&B sweep for phase 3 (final model selection and zero-shot evaluation).

        :param model_name: Name of the model to sweep
        """
        print("🔍 Preparing sweep configuration for Phase 2...")
    
        run = wandb.init()

        if self._validate_phase_3_sweep_config(wandb.config):

            classifier_config = self.handler.get_sweeps_config_phase_3(
                model=model_name,
                classifier_config=self.cfg['classifier_definition'],
                sweep_config=wandb.config
            )
            
            self.cfg['classifier_definition'] = classifier_config

            print(OmegaConf.to_yaml(self.cfg['classifier_definition']))
            

            seed = 42
            print(f"🔍 Running config with seed {seed}:")
            config_reader = DatasetConfigReader(self.cfg)
            dataset_handler = DatasetHandler(config_reader, build_zero_shot_dataset=True)
            launcher = ClassifierLauncher(config_reader, dataset_handler, zero_shot_test=True, random_seed=seed)
            launcher.run()

    def run_final_evaluation(self, model_name: str, save_model_folder: str, save_metrics_path: str = None):
        """
        Run final evaluation for the best configurations from all phases.

        :param model_name: Name of the model
        :param save_model_folder: Folder to save trained models
        :param save_metrics_path: Path to save metrics CSV
        """
        print("🔍 Preparing configuration for final evaluation")
    
        phase_1_configs, phase_2_configs, phase_3_configs = self.handler.get_final_evaluation_config(model=model_name, classifier_config=self.cfg['classifier_definition'])
        
        avg_metrics_collection = {}
        
        if phase_1_configs is not None:
            aux_cfg = self.cfg.copy()
            model_config = phase_1_configs['model_config']
        
            for key, value in phase_1_configs['embedding_config'].items():
                metrics_collection = {}

                classifier_config = self.handler._build_classifier_config(
                    classifier_config=aux_cfg['classifier_definition'],
                    model=model_name,
                    model_config=model_config,
                    embedding_config=value,
                )
                print(f"Classifier config: \n", {classifier_config})

                aux_cfg['classifier_definition'] = classifier_config

                for seed in aux_cfg['classifier_definition']['random_seeds']:
                    metrics, _ = self._launch_classifier_launcher(aux_cfg, seed)

                    for metric, value in metrics.items():
                        if metric not in metrics_collection:
                            metrics_collection[metric] = []
                        metrics_collection[metric].append(value)

                avg_metrics = {k: sum(v) / len(v) for k, v in metrics_collection.items()}
                avg_metrics_collection[key] = avg_metrics

        if phase_2_configs is not None:
            aux_cfg = self.cfg.copy()
            embedding_config = phase_2_configs['embedding_config']
        
            for key, value in phase_2_configs['model_configs'].items():
                metrics_collection = {}

                classifier_config = self.handler._build_classifier_config(
                    classifier_config=aux_cfg['classifier_definition'],
                    model=model_name,
                    model_config=value.get('parameters'),
                    embedding_config=embedding_config,
                )

                aux_cfg['classifier_definition'] = classifier_config
            
                for seed in aux_cfg['classifier_definition']['random_seeds']:
                    metrics, _ = self._launch_classifier_launcher(aux_cfg, seed)

                    for metric, value in metrics.items():
                        if metric not in metrics_collection:
                            metrics_collection[metric] = []
                        metrics_collection[metric].append(value)

                avg_metrics = {k: sum(v) / len(v) for k, v in metrics_collection.items()}
                avg_metrics_collection[key] = avg_metrics
        
        if phase_3_configs is not None:
            aux_cfg = self.cfg.copy()
            for key, value in phase_3_configs.items():
                metrics_collection = {}

                classifier_config = self.handler.get_sweeps_config_phase_3(
                    model=model_name,
                    classifier_config=aux_cfg['classifier_definition'],
                    sweep_config=value.get('parameters'),
                )
                
                aux_cfg['classifier_definition'] = classifier_config

                for seed in aux_cfg['classifier_definition']['random_seeds']:

                    metrics, _ = self._launch_classifier_launcher(aux_cfg, seed)
                    for metric, value in metrics.items():
                        if metric not in metrics_collection:
                            metrics_collection[metric] = []
                        metrics_collection[metric].append(value)

                avg_metrics = {k: sum(v) / len(v) for k, v in metrics_collection.items()}
                avg_metrics_collection[key] = avg_metrics

        print("🔍 Final evaluation metrics:")
        self._store_metrics_csv(avg_metrics_collection, model_name, save_metrics_path)

    def evaluate_protein_prediction_agreement(self, model_name: str, save_model_folder: str, save_metrics_path: str = None, best_configs_list: list = None):
        """
        Evaluate protein-level prediction agreement for the best configurations.

        :param model_name: Name of the model
        :param save_model_folder: Folder to save models
        :param save_metrics_path: Path to save metrics CSV
        :param best_configs_list: List of best configuration keys to evaluate
        """
        print("🔍 Preparing configuration for final evaluation")
    
        phase_1_configs, phase_2_configs, phase_3_configs = self.handler.get_final_evaluation_config(model=model_name, classifier_config=self.cfg['classifier_definition'])
        
        results_collection = []

        if phase_1_configs is not None:
            aux_cfg = self.cfg.copy()
            model_config = phase_1_configs['model_config']

            for key, value in phase_1_configs['embedding_config'].items():
                if best_configs_list is not None and key in best_configs_list:

                    classifier_config = self.handler._build_classifier_config(
                        classifier_config=aux_cfg['classifier_definition'],
                        model=model_name,
                        model_config=model_config,
                        embedding_config=value,
                    )
                    print(f"Classifier config: \n", classifier_config)

                    aux_cfg['classifier_definition'] = classifier_config

                    self._launch_classifier_storing_zero_shot_results(
                        aux_cfg, model_name, key, results_collection
                    )

        if phase_2_configs is not None:
            aux_cfg = self.cfg.copy()
            embedding_config = phase_2_configs['embedding_config']
        
            for key, value in phase_2_configs['model_configs'].items():
                if best_configs_list is not None and key in best_configs_list:

                    classifier_config = self.handler._build_classifier_config(
                        classifier_config=aux_cfg['classifier_definition'],
                        model=model_name,
                        model_config=value.get('parameters'),
                        embedding_config=embedding_config,
                    )

                    aux_cfg['classifier_definition'] = classifier_config
                
                    self._launch_classifier_storing_zero_shot_results(
                            aux_cfg, model_name, key, results_collection
                        )
        
        if phase_3_configs is not None:
            aux_cfg = self.cfg.copy()
            for key, value in phase_3_configs.items():
                if best_configs_list is not None and key in best_configs_list:
                        
                    classifier_config = self.handler.get_sweeps_config_phase_3(
                        model=model_name,
                        classifier_config=aux_cfg['classifier_definition'],
                        sweep_config=value.get('parameters'),
                    )
                    
                    aux_cfg['classifier_definition'] = classifier_config

                    self._launch_classifier_storing_zero_shot_results(
                            aux_cfg, model_name, key, results_collection
                        )

        print("🔍 Final evaluation metrics:")
        self._store_metrics_csv(results_collection, model_name, file_name="zero_shot_prediction_agreement", save_as_T=False)

    def _launch_classifier_storing_zero_shot_results(self, cfg, model_name, key, results_collection):
        """
        Launch classifier and store zero-shot results for each seed.

        :param cfg: Configuration dictionary
        :param model_name: Name of the model
        :param key: Configuration key
        :param results_collection: List to append results to
        """
        print(f"🔍 Running config for model {model_name} with key {key}:")

        for seed in cfg['classifier_definition']['random_seeds']:
            _, zero_shot_results = self._launch_classifier_launcher(cfg, seed)

            for result in zero_shot_results:
                result.update({
                    "model_type": model_name,
                    "model_config": key,
                    "seed": seed
                })
                results_collection.append(result)

    def _launch_classifier_launcher(self, cfg, seed=42):
        """
        Launch the classifier for a given configuration and seed.

        :param cfg: Configuration dictionary
        :param seed: Random seed
        :return: Metrics and results from the classifier launcher
        """
        print(f"🔍 Running config with seed {seed}:")
        
        config_reader = DatasetConfigReader(cfg)
        dataset_handler = DatasetHandler(config_reader, build_zero_shot_dataset=True)
        launcher = ClassifierLauncher(config_reader, dataset_handler, zero_shot_test=True, random_seed=seed)
        return launcher.run()

    def _calculate_seed_from_config(self, config, phase_num=1, seed_offset=42):
        """
        Calculate a deterministic seed from a config dictionary.

        :param config: Configuration dictionary
        :return: Integer seed
        """

        if not isinstance(config, dict):
            raise ValueError("Config must be a dictionary.")
        
        if phase_num < 1 or phase_num > 3:
            raise ValueError("Phase number must be between 1 and 3.")
        
        if phase_num == 1:
            # For phase 1, use the config index and training parameters
            config = {
                'config_index': config.get('config_index'),
                'training_cv_folds': config.get('training_cv_folds'),
                'training_cross_val_balance': config.get('training_cross_val_balance')
            }
            return seed_offset + config['config_index'] * config['training_cv_folds'] * (2 if config['training_cross_val_balance'] == 'organism' else 1)

        elif phase_num == 2:

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

        elif phase_num == 3:

            return seed_offset

    def _validate_phase_3_sweep_config(self, config):
        """
        Validate phase 3 sweep config (must have at least 2 features enabled).

        :param config: Configuration dictionary
        :return: True if valid, False otherwise
        """
        use_flags = [
            config.get("use_ESM"),
            config.get("use_ProtT5"),
            config.get("use_ProstT5"),
            config.get("go_embedding_config").get("use_GeOKG"),
            config.get("use_c_max_mbl"),
            config.get("use_f_max_mbl"),
            config.get("use_p_max_mbl"),
            config.get("use_seq_length"),
        ]

        if sum(bool(v) for v in use_flags) < 2:
            print("❌ Invalid config: fewer than 2 features selected. Skipping...")
            wandb.log({"skipped": True})
            wandb.finish()
            return False

        return True

    def _store_metrics_csv(self, metrics_collection, model_name, save_metrics_path=None, file_name=None, save_as_T=True):
        """
        Store metrics as a CSV file.

        :param metrics_collection: Dictionary or list of metrics
        :param model_name: Name of the model
        :param save_metrics_path: Path to save metrics CSV
        :param file_name: Optional custom file name
        :param save_as_T: If True, transpose DataFrame before saving
        """
        if file_name is not None:
            save_metrics_path = f"./results/{file_name}_{model_name}.csv"
        else:
            save_metrics_path = f"./results/final_evaluation_{model_name}.csv"
        
        import pandas as pd
        
        if save_as_T:
            df = pd.DataFrame(metrics_collection).T
            df.to_csv(save_metrics_path, index_label='Configuration')
        else:
            df = pd.DataFrame(metrics_collection)
            df.to_csv(save_metrics_path, index=False)

        print(f"✅ Metrics saved to {save_metrics_path}")