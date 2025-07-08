# Final commit – Master’s Thesis by Àlex Domínguez Roig

import itertools
import numpy as np # type: ignore
from copy import deepcopy
from typing import Dict, List, Any
from project_root.utils.config_utils import validate_config

class ForkExpander:
    """
    Utility class to expand parameter/configuration forks into all valid combinations
    for models, embeddings, features, and trainer settings.
    """

    def __init__(self, sampling_resolution: int = 5):
        """
        Initialize the ForkExpander.

        :param sampling_resolution: Number of discrete samples for continuous distributions (min/max)
        """
        self.sampling_resolution = sampling_resolution

    def expand_models(self, forks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Expand a list of forked model parameter spaces into all valid configurations.

        :param forks: List of model fork dictionaries
        :return: List of valid model configuration dictionaries
        """
        all_combinations = []

        for fork in forks:
            model_name = list(fork.keys())[0]  # e.g., 'svm', 'mlp_protein_classifier'
            model_block = fork[model_name]

            print(f"Expanding fork for model: {model_name}")

            if 'parameters' not in model_block:
                continue  # skip if no parameters to tune
            
            param_grid = self._expand_parameter_block(model_block['parameters'])

            for param_set in param_grid:
                config = {
                    "model": {
                        "type": model_name,
                        "params": param_set
                    }
                }
                is_valid = self._is_valid(model_name, param_set)
                print(f"Config for {model_name}: {config} - Valid: {is_valid}")
                if is_valid:
                    all_combinations.append(config)

        print(f"Total configurations across all models: {len(all_combinations)}")
        return all_combinations

    def expand_embeddings(self, embedding_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Expand embedding configuration into all valid combinations.

        :param embedding_config: Embedding configuration dictionary
        :return: List of valid embedding configuration dictionaries
        """
        configs = []

        seq_cfg = embedding_config.get("embedding_tune_config", {}).get("sequence_embeddings", {})
        go_cfg = embedding_config.get("embedding_tune_config", {}).get("go_embeddings", {})

        # Sequence embeddings
        seq_keys = list(seq_cfg.keys())
        seq_opts = []
        for key in seq_keys:
            use = seq_cfg[key].get("use", [True])
            dim = seq_cfg[key]["dim"]["values"]
            seq_opts.append([(key, {"use_autoencoder": False, "target_dim": d}) for u in use if u for d in dim])

        # GO embeddings
        go_use = go_cfg.get("use", [True])
        auto = go_cfg.get("autoencoded_go_embeddings", {})
        input_dims = auto.get("input_dim", {}).get("values", [500])
        emb_dims = auto.get("emb_dim", {}).get("values", [128])
        agg_methods = auto.get("aggregated_dim", {}).get("strategies", ["mean_pooling"])
        go_cats = auto.get("go_categories", {}).get("values", ["C_F"])

        print("---" * 20)
        print(f"Auto values for GO embeddings: {auto}")
        print("---" * 20)
        
        # Sequence embedding combinations
        for combination in itertools.product(*seq_opts):
            seq_dict = {name: cfg for name, cfg in combination}

            # Add GO embedding variants
            for use_go in go_use:
                if not use_go:
                    go_dict = {}
                else:
                    for use_autoencoded in [True, False]:
                        for input_dim in input_dims:
                            for emb_dim in emb_dims:
                                for agg in agg_methods:
                                    for cat in go_cats:
                                        config = {
                                            "classifier_definition": {
                                                "sequence_embeddings": seq_dict,
                                                "go_embeddings": {
                                                    "GeOKG": {
                                                        "autoencoded_embeddings": use_autoencoded,
                                                        "input_dim": input_dim,
                                                        "emb_dim": emb_dim,
                                                        "aggregated_dim": emb_dim * 10 if agg == "padding" else emb_dim,
                                                        "aggregation_strategy": agg,
                                                        "go_categories": cat
                                                    }
                                                }
                                            }
                                        }
                                        if self._is_valid_embedding(config["classifier_definition"]["go_embeddings"]):
                                            configs.append(config)
                if not use_go:
                    configs.append({
                        "classifier_definition": {
                            "sequence_embeddings": seq_dict,
                            "go_embeddings": {}
                        }
                    })
        return configs

    def expand_features(self, feature_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Expand feature configuration into all valid feature column combinations.

        :param feature_config: Feature configuration dictionary
        :return: List of feature configuration dictionaries
        """
        features = feature_config.get("features_tune_config", {})
        print(f"Expanding features with {features} settings.")
        options = {
            feature: settings["use"]
            for feature, settings in features.items()
        }

        combinations = list(itertools.product(*options.values()))
        keys = list(options.keys())

        expanded = []
        for combo in combinations:
            selected = [k for k, v in zip(keys, combo) if v]
            expanded.append({
                "classifier_definition": {
                    "features_col": selected
                }
            })
        return expanded

    def expand_trainer(self, trainer_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Expand trainer configuration into all valid combinations.

        :param trainer_config: Trainer configuration dictionary
        :return: List of trainer configuration dictionaries
        """
        params = trainer_config.get("trainer_tune_config", {})
        discrete_grid = {}

        for k, v in params.items():
            if isinstance(v, dict):
                if "values" in v:
                    discrete_grid[k] = v["values"]
                elif "min" in v and "max" in v:
                    dist = v.get("distribution", "uniform")
                    if dist == "int_uniform":
                        discrete_grid[k] = list(range(int(v["min"]), int(v["max"]) + 1))
                    elif dist == "uniform":
                        discrete_grid[k] = np.round(np.linspace(v["min"], v["max"], self.sampling_resolution), 5).tolist()
                    elif dist == "log_uniform":
                        discrete_grid[k] = np.round(np.logspace(np.log10(v["min"]), np.log10(v["max"]), self.sampling_resolution), 5).tolist()
            else:
                discrete_grid[k] = [v]

        keys, values = zip(*discrete_grid.items())
        combinations = [dict(zip(keys, vals)) for vals in itertools.product(*values)]

        return [{"training": c} for c in combinations]

    def _expand_parameter_block(self, param_dict: Dict[str, Any], debug: bool = False) -> List[Dict[str, Any]]:
        """
        Expand a single model's parameter space into all combinations.

        :param param_dict: Dictionary of parameter settings
        :param debug: If True, print debug info
        :return: List of parameter combination dictionaries
        """
        param_lists = {}
        print(f"[DEBUG] Expanding parameter block: {param_dict}") if debug else None
        for param, settings in param_dict.items():
            if debug:
                print(f"[DEBUG] Processing param: {param} | settings: {settings}")
            if 'values' in settings:
                param_lists[param] = settings['values']
                if debug:
                    print(f"[DEBUG] -> Using explicit values: {param_lists[param]}")
            elif 'min' in settings and 'max' in settings:
                dist = settings.get('distribution', 'uniform')
                if debug:
                    print(f"[DEBUG] -> Detected distribution: {dist}")
                if dist == 'int_uniform':
                    param_lists[param] = list(range(int(settings['min']), int(settings['max']) + 1))
                    if debug:
                        print(f"[DEBUG] -> int_uniform range: {param_lists[param]}")
                elif dist == 'uniform':
                    param_lists[param] = np.round(np.linspace(settings['min'], settings['max'], self.sampling_resolution), 5).tolist()
                    if debug:
                        print(f"[DEBUG] -> uniform linspace: {param_lists[param]}")
                elif dist == 'log_uniform':
                    param_lists[param] = np.round(np.logspace(np.log10(settings['min']), np.log10(settings['max']), self.sampling_resolution), 5).tolist()
                    if debug:
                        print(f"[DEBUG] -> log_uniform logspace: {param_lists[param]}")
                else:
                    print(f"[DEBUG] -> Unsupported distribution: {dist}")
                    raise ValueError(f"Unsupported distribution: {dist}")
            else:
                print(f"[DEBUG] -> Invalid parameter format for {param}: {settings}")
                raise ValueError(f"Invalid parameter format for {param}: {settings}")
    
        if debug:
            print(f"[DEBUG] param_lists: {param_lists}")
    
        keys, values = zip(*param_lists.items())
        if debug:
            print(f"[DEBUG] keys: {keys}")
            print(f"[DEBUG] values: {values}")
        combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
        if debug:
            print(f"[DEBUG] Generated {len(combinations)} combinations.")
        return combinations

    def _is_valid(self, model_name: str, param_config: Dict[str, Any]) -> bool:
        """
        Validate a model parameter configuration using external validation.

        :param model_name: Name of the model
        :param param_config: Parameter configuration dictionary
        :return: True if valid, False otherwise
        """
        try:
            return validate_config(model_name, param_config)
        except ValueError as e:
            # print(f"❌ Skipping invalid config for {model_name}: {e}")
            return False

    def _is_valid_embedding(self, embedding_config: Dict[str, Any]) -> bool:
        """
        Validate if the embedding configuration is valid.

        :param embedding_config: Embedding configuration dictionary
        :return: True if valid, False otherwise
        """
        if not embedding_config:
            return False
        
        if "GeOKG" not in embedding_config:
            print("❌ The system only supports GeOKG embeddings.")
            print("If you want to use other embeddings, please modify the code in `fork_expander.py`.")
            return False

        geokg_config = embedding_config["GeOKG"]
        
        auto = geokg_config.get("autoencoded_embeddings", {})
        input_dim = geokg_config.get("input_dim")
        emb_dim = geokg_config.get("emb_dim")
        agg = geokg_config.get("aggregation_strategy")
        cat = geokg_config.get("go_categories")

        if input_dim is None or emb_dim is None or agg is None or cat is None:
            return False

        if agg not in ["mean_pooling", "padding"]:
            return False

        if cat not in ["C_F", "C_F_P"]:
            return False

        if not auto:
            if input_dim == emb_dim and input_dim in [50, 100, 200, 500, 1000]:
                return True
        elif input_dim in [200, 500, 1000]:
            if input_dim == 1000:
                if emb_dim in [32, 128]:
                    return True
            elif input_dim in [200, 500]:
                if emb_dim in [32, 64, 128]:
                    return True

        return False