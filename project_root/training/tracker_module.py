# project_root/tracking/tracker_module.py

import wandb

class TrackerModule:
    def __init__(self, project_name="ProteinClassification", run_name=None, config=None, offline=False):
        """
        Args:
            project_name: W&B project name
            run_name: Optional custom run name
            config: Optional dict with hyperparameters/config
            offline: If True, runs W&B in offline mode
        """
        self.run = None
        if offline:
            wandb.init(mode="offline", project=project_name, name=run_name, config=config)
        else:
            wandb.init(project=project_name, name=run_name, config=config)
        self.run = wandb.run

    def log_metric(self, name, value, step=None):
        wandb.log({name: value}, step=step)

    def log_artifact(self, artifact_path, artifact_name, artifact_type="file"):
        artifact = wandb.Artifact(artifact_name, type=artifact_type)
        artifact.add_file(artifact_path)
        wandb.log_artifact(artifact)

    def finish(self):
        wandb.finish()
