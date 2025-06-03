# XGBoost parameters url https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/ https://www.ibm.com/es-es/think/topics/xgboost
# Logic regresion url https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html


def validate_config(model_name, config):
    """
    Validate hyperparameter configurations for Logistic Regression, XGBoost, SVM, KNN, Random Forest, and MLPProteinClassifier.
    :param model_name: str, one of 'logistic_regression', 'xgboost', 'svm', 'knn', 'random_forest', 'mlp_protein_classifier'
    :param config: dict, hyperparameter values
    :return: True if valid, raises ValueError if invalid
    """
    if model_name == "logistic_regression":
        penalty = config.get("penalty")
        solver = config.get("solver")
        dual = config.get("dual")
        l1_ratio = config.get("l1_ratio")
        multi_class = config.get("multi_class")

        if penalty == 'l1' and solver not in ['liblinear', 'saga']:
            raise ValueError("LogisticRegression: penalty='l1' only works with solvers 'liblinear' or 'saga'.")
        if penalty == 'elasticnet' and solver != 'saga':
            raise ValueError("LogisticRegression: penalty='elasticnet' requires solver='saga'.")
        if penalty is None and solver == 'liblinear':
            raise ValueError("LogisticRegression: solver='liblinear' does not support penalty=None.")
        if dual and (solver != 'liblinear' or penalty != 'l2'):
            raise ValueError("LogisticRegression: dual=True requires solver='liblinear' and penalty='l2'.")
        if penalty != 'elasticnet' and l1_ratio is not None:
            raise ValueError("LogisticRegression: l1_ratio only applies with penalty='elasticnet'.")
        if multi_class == 'multinomial' and solver in ['liblinear', 'newton-cholesky']:
            raise ValueError("LogisticRegression: solver does not support multi_class='multinomial'.")

    elif model_name == "xgboost":
        booster = config.get("booster", "gbtree")
        objective = config.get("objective", "binary:logistic")
        scale_pos_weight = config.get("scale_pos_weight", 1)

        if booster != "gbtree":
            raise ValueError("XGBoost: Use 'gbtree' booster for classification.")
        if objective != "binary:logistic":
            raise ValueError("XGBoost: Use objective='binary:logistic' for binary classification.")
        if scale_pos_weight < 1:
            raise ValueError("XGBoost: scale_pos_weight should be >= 1 for binary tasks.")

    elif model_name == "svm":
        kernel = config.get("kernel", "rbf")
        degree = config.get("degree")
        coef0 = config.get("coef0")
        gamma = config.get("gamma")

        if kernel not in ['linear', 'poly', 'rbf', 'sigmoid']:
            raise ValueError("SVM: Invalid kernel.")
        if kernel != 'poly' and degree is not None:
            raise ValueError("SVM: degree applies only to kernel='poly'.")
        if kernel not in ['poly', 'sigmoid'] and coef0 is not None:
            raise ValueError("SVM: coef0 applies only to kernel='poly' or 'sigmoid'.")
        if kernel == 'linear' and gamma not in [None, 'scale', 'auto']:
            raise ValueError("SVM: gamma is ignored with kernel='linear'.")

    elif model_name == "knn":
        n_neighbors = config.get("n_neighbors")
        weights = config.get("weights")
        algorithm = config.get("algorithm")
        leaf_size = config.get("leaf_size")
        p = config.get("p")
        metric = config.get("metric")
        n_jobs = config.get("n_jobs")

        if n_neighbors is not None and (n_neighbors <= 0 or not isinstance(n_neighbors, int)):
            raise ValueError("KNN: n_neighbors must be a positive integer.")
        if weights not in [None, 'uniform', 'distance']:
            raise ValueError("KNN: weights must be 'uniform', 'distance', or None.")
        if algorithm not in [None, 'auto', 'ball_tree', 'kd_tree', 'brute']:
            raise ValueError("KNN: algorithm must be one of ['auto', 'ball_tree', 'kd_tree', 'brute'].")
        if leaf_size is not None and (leaf_size <= 0 or not isinstance(leaf_size, int)):
            raise ValueError("KNN: leaf_size must be a positive integer.")
        if p not in [1, 2, None]:
            raise ValueError("KNN: p must be 1 (Manhattan) or 2 (Euclidean) for Minkowski distance.")
        if metric not in [None, 'minkowski', 'euclidean', 'manhattan']:
            raise ValueError("KNN: metric must be 'minkowski', 'euclidean', 'manhattan', or None.")
        if n_jobs is not None and not isinstance(n_jobs, int):
            raise ValueError("KNN: n_jobs must be an integer.")

    elif model_name == "random_forest":
        n_estimators = config.get("n_estimators")
        criterion = config.get("criterion")
        max_depth = config.get("max_depth")
        min_samples_split = config.get("min_samples_split")
        min_samples_leaf = config.get("min_samples_leaf")
        max_features = config.get("max_features")
        bootstrap = config.get("bootstrap")
        class_weight = config.get("class_weight")
        max_samples = config.get("max_samples")
        n_jobs = config.get("n_jobs")

        if n_estimators is not None and (n_estimators <= 0 or not isinstance(n_estimators, int)):
            raise ValueError("RandomForest: n_estimators must be a positive integer.")
        if criterion not in [None, 'gini', 'entropy', 'log_loss']:
            raise ValueError("RandomForest: criterion must be 'gini', 'entropy', or 'log_loss'.")
        if max_depth is not None and (max_depth <= 0 or not isinstance(max_depth, int)):
            raise ValueError("RandomForest: max_depth must be a positive integer or None.")
        if min_samples_split is not None and (min_samples_split < 2 or not isinstance(min_samples_split, int)):
            raise ValueError("RandomForest: min_samples_split must be an integer >= 2.")
        if min_samples_leaf is not None and (min_samples_leaf < 1 or not isinstance(min_samples_leaf, int)):
            raise ValueError("RandomForest: min_samples_leaf must be an integer >= 1.")
        if max_features not in [None, 'auto', 'sqrt', 'log2']:
            raise ValueError("RandomForest: max_features must be 'auto', 'sqrt', 'log2', or None.")
        if bootstrap not in [None, True, False]:
            raise ValueError("RandomForest: bootstrap must be True, False, or None.")
        if class_weight not in [None, 'balanced', 'balanced_subsample']:
            raise ValueError("RandomForest: class_weight must be None, 'balanced', or 'balanced_subsample'.")
        if max_samples is not None and (not (0.0 < max_samples <= 1.0)):
            raise ValueError("RandomForest: max_samples must be between 0 and 1 if bootstrap=True.")
        if n_jobs is not None and not isinstance(n_jobs, int):
            raise ValueError("RandomForest: n_jobs must be an integer.")

    elif model_name == "mlp_protein_classifier":
        num_hidden_layers = config.get("num_hidden_layers")
        dropout_rate = config.get("dropout_rate")
        hidden_layers_mode = config.get("hidden_layers_mode")
        custom_hidden_layers = config.get("custom_hidden_layers")
        activation_function = config.get("activation_function")
        use_batch_norm = config.get("use_batch_norm")
        output_activation = config.get("output_activation")
        initialization = config.get("initialization")

        if num_hidden_layers is not None and (num_hidden_layers <= 0 or not isinstance(num_hidden_layers, int)):
            raise ValueError("MLP: num_hidden_layers must be a positive integer.")
        if dropout_rate is not None and (not (0.0 <= dropout_rate <= 1.0)):
            raise ValueError("MLP: dropout_rate must be between 0 and 1.")
        if hidden_layers_mode not in ["quadratic_increase", "custom"]:
            raise ValueError("MLP: hidden_layers_mode must be 'quadratic_increase' or 'custom'.")
        if hidden_layers_mode == "custom" and not isinstance(custom_hidden_layers, list):
            raise ValueError("MLP: custom_hidden_layers must be provided as a list when hidden_layers_mode='custom'.")
        valid_activations = ["ReLU", "LeakyReLU", "ELU", "GELU"]
        if activation_function is not None and activation_function not in valid_activations:
            raise ValueError(f"MLP: activation_function must be one of {valid_activations}.")
        if output_activation is not None and output_activation not in [None, "Sigmoid", "Tanh"]:
            raise ValueError("MLP: output_activation must be 'Sigmoid', 'Tanh', or None.")
        if initialization is not None and initialization not in [None, "xavier", "kaiming", "normal"]:
            raise ValueError("MLP: initialization must be 'xavier', 'kaiming', 'normal', or None.")
        if use_batch_norm not in [None, True, False]:
            raise ValueError("MLP: use_batch_norm must be True, False, or None.")

    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return True


'''
# Example usage
try:
    config_svm = {
        "kernel": "linear",
        "degree": 3,  # Should trigger a warning
        "coef0": 0.5,  # Should trigger a warning
        "gamma": "scale",
        "probability": True
    }
    validate_config("svm", config_svm)
except ValueError as e:
    print(f"SVM config error: {e}")

try:
    config_lr = {
        "penalty": "elasticnet",
        "solver": "liblinear",
        "dual": False
    }
    validate_config("logistic_regression", config_lr)
except ValueError as e:
    print(f"LogisticRegression config error: {e}")



# Example usage:
try:
    config_lr = {
        "penalty": "l1",
        "solver": "lbfgs",  # This will trigger an error
        "dual": False,
        "l1_ratio": None,
        "multi_class": "auto"
    }
    validate_config("logistic_regression", config_lr)
except ValueError as e:
    print(f"LogisticRegression config error: {e}")

try:
    config_xgb = {
        "booster": "gbtree",
        "objective": "binary:logistic",
        "scale_pos_weight": 0.5  # This will trigger an error
    }
    validate_config("xgboost", config_xgb)
except ValueError as e:
    print(f"XGBoost config error: {e}")
'''