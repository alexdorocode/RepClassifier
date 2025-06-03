import os
import itertools
import pprint
import json
from omegaconf import OmegaConf
from project_root.experiment.classifier_launcher import ClassifierLauncher
from project_root.dataset.dataset_config import DatasetConfigReader
from project_root.experiment.validate_config import validate_config  # Assuming this is the correct import path for your validation function

class ExperimentLauncher:
    def __init__(self, base_config_path, config_forks):
        self.base_config_path = base_config_path
        self.config_forks = config_forks
        self.base_config = OmegaConf.load(base_config_path)
        self.fork_configs = [OmegaConf.load(f) for f in config_forks]

        self.display_configs()

    def generate_combinations(self):
        fork_options = [list(fork.values()) if isinstance(fork, dict) else fork for fork in self.fork_configs]
        all_combinations = list(itertools.product(*fork_options))
        print(f"🔄 Total combinations: {len(all_combinations)}")
        return all_combinations

    def merge_configs(self, combination):
        cfg = OmegaConf.create(OmegaConf.to_container(self.base_config, resolve=True))
        for override in combination:
            cfg = OmegaConf.merge(cfg, OmegaConf.create(override))
        return cfg

    def run_all(self):
        all_combinations = self.generate_combinations()
        for idx, combination in enumerate(all_combinations):
            print(f"\n🚀 Running combination {idx+1}/{len(all_combinations)}")
            combined_cfg = self.merge_configs(combination)

            # Initialize DatasetConfigReader
            config_reader = DatasetConfigReader(combined_cfg)

            # Validate model hyperparameters
            model_name = config_reader.model['type']
            model_params = config_reader.model['params']
            try:
                validate_config(model_name, model_params)

                # Initialize ClassifierLauncher with the full merged config
                launcher = ClassifierLauncher(combined_cfg)  # Or pass config_reader if ClassifierLauncher supports it
                launcher.run()
            except ValueError as e:
                print(f"❌ Skipping combination {idx+1} due to validation error: {e}")
    
    def display_configs(self):
        print("\n" + "="*40)
        print("🔧 Base Configuration:")
        print("="*40)
        # Use json.dumps for nice formatting if possible
        try:
            print(json.dumps(OmegaConf.to_container(self.base_config, resolve=True), indent=2))
        except Exception:
            pprint.pprint(self.base_config)
        print("\n" + "="*40)
        print("🔧 Fork Configurations:")
        print("="*40)
        for idx, fork in enumerate(self.fork_configs, 1):
            print(f"\n--- Fork {idx} ---")
            try:
                print(json.dumps(OmegaConf.to_container(fork, resolve=True), indent=2))
            except Exception:
                pprint.pprint(fork)
        print("="*40 + "\n")
