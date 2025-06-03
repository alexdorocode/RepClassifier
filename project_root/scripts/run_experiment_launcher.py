from project_root.experiment.experiment_launcher import ExperimentLauncher

if __name__ == "__main__":
    base_config = "project_root/config/config_classifier_base.yaml"
    config_forks = [
        "project_root/config/tunable_params/model_tunable_params/knn_tune_params.yaml",
        "project_root/config/tunable_params/model_tunable_params/svm_tune_params.yaml",
        "project_root/config/tunable_params/model_tunable_params/mlp_tune_params.yaml",
        "project_root/config/tunable_params/model_tunable_params/rf_tune_params.yaml",
        "project_root/config/tunable_params/model_tunable_params/xgb_tune_params.yaml",
        "project_root/config/tunable_params/model_tunable_params/lr_tune_params.yaml",
        "project_root/config/tunable_params/embeddings_tunable_params.yaml",
        "project_root/config/tunable_params/features_tunable_params.yaml",
        "project_root/config/tunable_params/training_tunable_params.yaml"
    ]

    launcher = ExperimentLauncher(base_config, config_forks)
    launcher.run_all()
