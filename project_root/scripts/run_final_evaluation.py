# Final commit – Master’s Thesis by Àlex Domínguez Roig

import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
from project_root.launchers.experiment_launcher import ExperimentLauncher
SAVE_MODEL_FOLDER = "./project_root/model_weights/final_eval/"
PATH_RESULTS_PHASE_1 = "./project_root/config/phase_results/for_final_eval_phase_1_configs.yaml"
PATH_RESULTS_PHASE_2 = "./project_root/config/phase_results/for_final_eval_phase_2_configs.yaml"
PATH_RESULTS_PHASE_3 = "./project_root/config/phase_results/for_final_eval_phase_3_configs.yaml"


@hydra.main(config_path="/Users/alexdominguez/Documents/GitHub/TFM/RepClassifier/project_root/config/", config_name="config_experiment_base", version_base="1.3")
def main(cfg: DictConfig):
    print("🔍 Initializing ExperimentConfigHandler...")

    launcher = ExperimentLauncher(cfg, 
                                  phase_1_result_path=PATH_RESULTS_PHASE_1,
                                  phase_2_result_path=PATH_RESULTS_PHASE_2,
                                  phase_3_result_path=PATH_RESULTS_PHASE_3)
    
    print("🔍 Running Phase 3 sweep for final evaluation...")
    for model_name in ['knn']:# ['lr', 'svm', 'rf', 'xgb', 'knn']:
        print(f"🔍 Running sweep for model: {model_name}")
        launcher.run_final_evaluation(model_name=model_name, 
                                       save_model_folder=SAVE_MODEL_FOLDER)

if __name__ == "__main__":
    main()