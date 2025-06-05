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
        sampling_resolution: int = 3
    ):
        self.base_config_path = base_config_path
        self.model_fork_paths = model_fork_paths
        self.embedding_fork_path = embedding_fork_path
        self.feature_fork_path = feature_fork_path
        self.training_fork_path = training_fork_path
        self.expander = ForkExpander(sampling_resolution=sampling_resolution)

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
                "cv_folds": 3,
                "cross_val_balance": training_fork["trainer_tune_config"].get("cross_val_balance", [None])
            }
        }

        phase_1_configs = self._expand_and_merge(model_forks, embedding_fork, feature_fork, training_fork)

        self.configs = phase_1_configs

    def set_sweeps_config_phase_2(self):
        # Phase 2: test all models with fixed embedding/feature config
        fixed_embedding = OmegaConf.to_container(OmegaConf.load(self.embedding_fork_path), resolve=True)["best_embedding"]
        fixed_feature = OmegaConf.to_container(OmegaConf.load(self.feature_fork_path), resolve=True)["best_features"]
        training_fork = OmegaConf.to_container(OmegaConf.load(self.training_fork_path), resolve=True)
        model_forks = [OmegaConf.to_container(OmegaConf.load(path), resolve=True) for path in self.model_fork_paths]
        self.configs = self._expand_and_merge(model_forks, [fixed_embedding], [fixed_feature], training_fork)

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

    def get_classifier_ready_configs(self, input_cfg) -> List[dict]:
        """
        Extracts full classifier config in the desired format, preserving:
        - defaults
        - classifier_definition
        - any top-level config metadata (e.g., label_col, balance_col, etc.)
        """
        classifier_configs = []

        for cfg in self.configs:
            config_out = {}

            # Copy classifier_definition block
            if "classifier_definition" in cfg:
                aux = cfg["classifier_definition"]
                aux["model"] = cfg.get("model", {})
                aux["training"] = {**aux.get('training', {}), **cfg.get("training", {})}
                config_out["classifier_definition"] = aux

            # Preserve top-level metadata if available
            for meta_key in ["label_col", "balance_col", "features_col"]:
                if meta_key in cfg:
                    config_out[meta_key] = cfg[meta_key]

            classifier_configs.append(config_out)

        
        OmegaConf.set_struct(input_cfg, False)
        
        for key in [
            'classifier_definition',
            'defaults',
            'model_fork_paths',
            'embedding_fork_path',
            'feature_fork_path',
            'training_fork_path'
        ]:
            if key in input_cfg:
                del input_cfg[key]

        return input_cfg, classifier_configs

