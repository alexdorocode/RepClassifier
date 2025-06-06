import json
import itertools
from typing import List, Dict
from omegaconf import OmegaConf

from project_root.experiment.fork_expander import ForkExpander
from project_root.utils.config_utils import merge_dicts  # Utility to deep merge dictionaries


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

        self.configs = []  # Will hold merged expanded configs

    def expand_all(self):
        print("🔄 Expanding configurations...")

        base_config = OmegaConf.to_container(OmegaConf.load(self.base_config_path), resolve=True)

        # Load fork dicts
        model_forks = [OmegaConf.to_container(OmegaConf.load(path), resolve=True) for path in self.model_fork_paths]
        embedding_fork = OmegaConf.to_container(OmegaConf.load(self.embedding_fork_path), resolve=True)
        feature_fork = OmegaConf.to_container(OmegaConf.load(self.feature_fork_path), resolve=True)
        training_fork = OmegaConf.to_container(OmegaConf.load(self.training_fork_path), resolve=True)

        # Expand each block
        print(f"📦 Expanding {len(model_forks)} model forks, 1 embedding fork, 1 feature fork, and 1 training fork...")
        expanded_models = self.expander.expand_models(model_forks)

        if not expanded_models:
            raise ValueError("No valid model configurations found after expansion.")
        else:
            print(f"✅ Expanded {len(expanded_models)} model configurations.")
        
        expanded_embeddings = self.expander.expand_embeddings(embedding_fork)
        
        if not expanded_embeddings:
            raise ValueError("No valid embedding configurations found after expansion.")
        else:
            print(f"✅ Expanded {len(expanded_embeddings)} embedding configurations.")
        
        expanded_features = self.expander.expand_features(feature_fork)

        if not expanded_features:
            raise ValueError("No valid feature configurations found after expansion.")
        else:
            print(f"✅ Expanded {len(expanded_features)} feature configurations.")

        expanded_training = self.expander.expand_trainer(training_fork)

        if not expanded_training:
            raise ValueError("No valid training configurations found after expansion.")
        else:
            print(f"✅ Expanded {len(expanded_training)} training configurations.")

        print(f"📦 Expanded: {len(expanded_models)} models × {len(expanded_embeddings)} embeddings × {len(expanded_features)} features × {len(expanded_training)} training")

        # Cartesian product of all expanded combinations
        product = itertools.product(expanded_models, expanded_embeddings, expanded_features, expanded_training)
        self.configs = [merge_dicts(base_config, *combo) for combo in product]
        print(f"✅ Total merged configurations: {len(self.configs)}")

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
