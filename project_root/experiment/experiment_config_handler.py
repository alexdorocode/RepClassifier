import json
import itertools
from typing import List, Dict
from omegaconf import OmegaConf

from project_root.experiment.fork_expander import ForkExpander
from project_root.utils.config_utils import merge_dicts


class ExperimentConfigHandler:
    def __init__(
        self,
        base_config_path: str,
        model_fork_paths: List[str],
        embedding_fork_path: str,
        feature_fork_path: str,
        training_fork_path: str,
        sampling_resolution: int = 3,

        phase_1_result_path: str = None,
        phase_2_result_path: str = None,
        phase_3_result_path: str = None,
    ):
        self.base_config_path = base_config_path
        self.model_fork_paths = model_fork_paths
        self.embedding_fork_path = embedding_fork_path
        self.feature_fork_path = feature_fork_path
        self.training_fork_path = training_fork_path
        self.expander = ForkExpander(sampling_resolution=sampling_resolution)

        self.phase_1_result_load_path = phase_1_result_path
        print("Initialized with phase 1 result path:", self.phase_1_result_load_path) if self.phase_1_result_load_path else None
        self.phase_2_result_path = phase_2_result_path
        print("Initialized with phase 2 result path:", self.phase_2_result_path) if self.phase_2_result_path else None
        self.phase_3_result_path = phase_3_result_path
        print("Initialized with phase 3 result path:", self.phase_3_result_path) if self.phase_3_result_path else None

        self.configs = []

    def _expand_and_merge(self, model_forks, embedding_fork, feature_fork, training_fork):
        base_config = OmegaConf.to_container(OmegaConf.load(self.base_config_path), resolve=True)

        expanded_models = self.expander.expand_models(model_forks)
        if not expanded_models:
            raise ValueError("No valid model configurations found after expansion.")

        expanded_embeddings = self.expander.expand_embeddings(embedding_fork)
        if not expanded_embeddings:
            raise ValueError("No valid embedding configurations found after expansion.")

        expanded_features = self.expander.expand_features(feature_fork)
        if not expanded_features:
            raise ValueError("No valid feature configurations found after expansion.")

        expanded_training = self.expander.expand_trainer(training_fork)
        if not expanded_training:
            raise ValueError("No valid training configurations found after expansion.")

        print(f"📦 Expanded: {len(expanded_models)} models × {len(expanded_embeddings)} embeddings × {len(expanded_features)} features × {len(expanded_training)} training")

        product = itertools.product(expanded_models, expanded_embeddings, expanded_features, expanded_training)
        return [merge_dicts(base_config, *combo) for combo in product]

    def expand_all(self):
        model_forks = [OmegaConf.to_container(OmegaConf.load(path), resolve=True) for path in self.model_fork_paths]
        embedding_fork = OmegaConf.to_container(OmegaConf.load(self.embedding_fork_path), resolve=True)
        feature_fork = OmegaConf.to_container(OmegaConf.load(self.feature_fork_path), resolve=True)
        training_fork = OmegaConf.to_container(OmegaConf.load(self.training_fork_path), resolve=True)

        self.configs = self._expand_and_merge(model_forks, embedding_fork, feature_fork, training_fork)

    def set_sweeps_config_phase_1(self, model) -> List[Dict]:
        # Load logistic regression fork path
        path = next(path for path in self.model_fork_paths if f"{model}_tune_params" in path)
        config = OmegaConf.to_container(OmegaConf.load(path), resolve=True)

        # Convert base_line to sweep-compatible format
        base_line_fixed = {
            f"{model}": {
                "parameters": {
                    k: {"values": [v]} for k, v in config[f"{model}"]["base_line"].items()
                }
            }
        }
        model_forks = [base_line_fixed]
        print(f"🔍 Using base line configuration for {model}: {model_forks}")

        # Load and resolve other forks
        embedding_fork = OmegaConf.to_container(OmegaConf.load(self.embedding_fork_path), resolve=True)
        feature_fork = {
            "features_tune_config": {
                "c_max_mbl": {"use": [True]},
                "f_max_mbl": {"use": [True]},
                "p_max_mbl": {"use": [True]},
                "seq_length": {"use": [True]}
            }
        }
        training_fork = OmegaConf.to_container(OmegaConf.load(self.training_fork_path), resolve=True)

        # Keep minimal trainer config
        training_fork = {
            "trainer_tune_config": {
                "cv_folds": [5],
                "cross_val_balance": [None]
            }
        }

        phase_1_configs = self._expand_and_merge(model_forks, embedding_fork, feature_fork, training_fork)

        self.configs = phase_1_configs

    def set_sweeps_config_phase_2(self, model):
        # Phase 2: test all models with fixed embedding/feature config
        if self.phase_1_result_load_path is None:
            raise ValueError("Phase 1 result path must be set before running Phase 2.")
        
        fixed_embedding = OmegaConf.to_container(OmegaConf.load(self.phase_1_result_load_path), resolve=True)["embedding_configs_phase_1"][f'{model}']
        fixed_feature = {
            "features_tune_config": {
                "c_max_mbl": {"use": [True]},
                "f_max_mbl": {"use": [True]},
                "p_max_mbl": {"use": [True]},
                "seq_length": {"use": [True]}
            }
        }
        training_fork = OmegaConf.to_container(OmegaConf.load(self.training_fork_path), resolve=True)
        model_forks = [OmegaConf.to_container(OmegaConf.load(path), resolve=True) for path in self.model_fork_paths]
        self.configs = self._expand_and_merge(model_forks, fixed_embedding, fixed_feature, training_fork)

    def set_sweeps_config_phase_3(self):
        # Phase 3: sweep training hyperparameters for top models
        top_model_paths = [path for path in self.model_fork_paths if any(x in path for x in ["mlp", "xgboost"])]
        model_forks = [OmegaConf.to_container(OmegaConf.load(path), resolve=True) for path in top_model_paths]
        embedding = OmegaConf.to_container(OmegaConf.load(self.embedding_fork_path), resolve=True)["best_embedding"]
        feature = OmegaConf.to_container(OmegaConf.load(self.feature_fork_path), resolve=True)["best_features"]
        training_fork = OmegaConf.to_container(OmegaConf.load(self.training_fork_path), resolve=True)
        self.configs = self._expand_and_merge(model_forks, [embedding], [feature], training_fork)

    def set_sweeps_config_phase_4(self):
        # Phase 4: test robustness across stratification or organism filtering
        best_model = OmegaConf.to_container(OmegaConf.load(self.model_fork_paths[0]), resolve=True)["best_model"]
        best_embedding = OmegaConf.to_container(OmegaConf.load(self.embedding_fork_path), resolve=True)["best_embedding"]
        best_feature = OmegaConf.to_container(OmegaConf.load(self.feature_fork_path), resolve=True)["best_features"]
        best_training = OmegaConf.to_container(OmegaConf.load(self.training_fork_path), resolve=True)["best_training"]
        self.configs =  self._expand_and_merge([best_model], [best_embedding], [best_feature], [best_training])

    def save_to_file(self, path: str):
        with open(path, "w") as f:
            json.dump(self.configs, f, indent=2)
        print(f"💾 Saved {len(self.configs)} configurations to {path}")

    def load_from_file(self, path: str):
        with open(path, "r") as f:
            self.configs = json.load(f)
        print(f"📂 Loaded {len(self.configs)} configurations from {path}")

    def get_configs(self) -> List[Dict]:
        return self.configs

    def get_classifier_ready_configs(self, input_cfg, overwrite_config=None, new_config=None) -> List[dict]:
        """
        Extracts full classifier config in the desired format, preserving:
        - defaults
        - classifier_definition
        - any top-level config metadata (e.g., label_col, balance_col, etc.)
        """

        print("🔍 Preparing classifier-ready configurations...")
        print("Overwrite config:", overwrite_config)
        print("New config:", new_config)

        classifier_configs = []

        for cfg in self.configs:
            config_out = {}

            # Copy classifier_definition block
            if "classifier_definition" in cfg:
                aux = cfg["classifier_definition"]
                aux["model"] = cfg.get("model", {})
                config_out["classifier_definition"] = aux

            # Preserve top-level metadata if available
            for meta_key in ["label_col", "balance_col", "features_col"]:
                if meta_key in cfg:
                    config_out[meta_key] = cfg[meta_key]

            classifier_configs.append(config_out)

        
        OmegaConf.set_struct(input_cfg, False)
        
        base_config = self._get_base_config(input_cfg)

        if overwrite_config and new_config:
            self._overwrite_config(input_cfg, overwrite_config, new_config)

        return base_config, classifier_configs

    def _get_base_config(self, config):
        """
        Cleans the configuration by removing unnecessary keys.
        """
        keys_to_remove = [
            'model_fork_paths',
            'embedding_fork_path',
            'feature_fork_path',
            'training_fork_path',
            'defaults'
        ]
        for key in keys_to_remove:
            if key in config:
                del config[key]
        
        return config

    def _overwrite_config(self, config, overwrite_config, new_config):
        """
        Overwrites specific configuration keys with new values.
        """
        if overwrite_config in config and isinstance(config[overwrite_config], dict):
            for key, value in new_config.items():
                if key in config['classifier_definition'][overwrite_config]:
                    config[overwrite_config][key] = value
                    print(f"Updated {key} in {overwrite_config} with new value: {value}")
                else:
                    print(f"Warning: {key} not found in {overwrite_config}, skipping update.")
        else:
            print(f"Warning: {overwrite_config} not found in config, skipping update.")