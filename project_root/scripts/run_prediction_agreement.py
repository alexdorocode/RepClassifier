# Final commit – Master’s Thesis by Àlex Domínguez Roig

import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
from project_root.launchers.experiment_launcher import ExperimentLauncher
SAVE_MODEL_FOLDER = "./project_root/model_weights/final_eval/"
PATH_RESULTS_PHASE_1 = "./project_root/config/phase_results/for_final_eval_phase_1_configs.yaml"
PATH_RESULTS_PHASE_2 = "./project_root/config/phase_results/for_final_eval_phase_2_configs.yaml"
PATH_RESULTS_PHASE_3 = "./project_root/config/phase_results/for_final_eval_phase_3_configs.yaml"
BEST_CONFIGS_PATH = {
    'lr':['best_config_lr_1_10',
            'best_config_lr_2_1', 'best_config_lr_2_2', 'best_config_lr_2_3', 
          'best_config_lr_2_4', 'best_config_lr_2_6', 'best_config_lr_2_7', 
          'best_config_lr_2_8', 'best_config_lr_2_9', 'best_config_lr_2_10'],
    'svm':['best_config_svm_3_2', 'best_config_svm_3_3',
           'best_config_svm_3_4', 'best_config_svm_3_5', 'best_config_svm_3_6', 
          'best_config_svm_3_7', 'best_config_svm_3_8', 'best_config_svm_3_9', 
          'best_config_svm_3_1', 'best_config_svm_3_11'],
    'rf':['best_config_rf_3_1', 'best_config_rf_3_2', 
          'best_config_rf_3_3', 'best_config_rf_3_4', 'best_config_rf_3_5', 
          'best_config_rf_2_2', 'best_config_rf_2_7', 'best_config_rf_2_8', 
          'best_config_rf_2_9', 'best_config_rf_2_10'],
    'xgb':['best_config_xgb_1_1', 'best_config_xgb_1_2',
           'best_config_xgb_1_3', 'best_config_xgb_1_4', 'best_config_xgb_1_5', 
          'best_config_xgb_1_6', 'best_config_xgb_1_7', 'best_config_xgb_1_8', 
          'best_config_xgb_1_9', 'best_config_xgb_1_10', 
          'best_config_xgb_2_3', 'best_config_xgb_3_1'],
    'knn':['best_config_knn_3_1', 'best_config_knn_3_2',
           'best_config_knn_3_3', 'best_config_knn_3_4', 'best_config_knn_3_5', 
          'best_config_knn_3_6', 'best_config_knn_3_7', 'best_config_knn_3_8']
}

@hydra.main(config_path="/Users/alexdominguez/Documents/GitHub/TFM/RepClassifier/project_root/config/", config_name="config_experiment_base", version_base="1.3")
def main(cfg: DictConfig):
    print("🔍 Initializing ExperimentConfigHandler...")

    launcher = ExperimentLauncher(cfg, 
                                  phase_1_result_path=PATH_RESULTS_PHASE_1,
                                  phase_2_result_path=PATH_RESULTS_PHASE_2,
                                  phase_3_result_path=PATH_RESULTS_PHASE_3)
    
    print("🔍 Running Phase 3 sweep for final evaluation...")
    for model_name in ['lr', 'svm', 'rf', 'xgb', 'knn']:
        print(f"🔍 Running sweep for model: {model_name}")
        launcher.evaluate_protein_prediction_agreement(model_name=model_name, 
                                       save_model_folder=SAVE_MODEL_FOLDER,
                                       best_configs_list=BEST_CONFIGS_PATH[model_name])

if __name__ == "__main__":
    main()