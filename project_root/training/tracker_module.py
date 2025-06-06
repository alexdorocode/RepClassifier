import wandb

class TrackerModule:
    def __init__(self, project_name="ProteinClassification", run_name=None, config=None, offline=False, tags=None, random_seed=None):
        """
        Args:
            project_name: W&B project name
            run_name: Optional custom run name
            config: Optional dict with hyperparameters/config
            offline: If True, runs W&B in offline mode
            tags: Optional list of tags to attach to the run
        """
        self.run = wandb.run

        # Use existing W&B run (sweeps already started it)
        if self.run is None:
            config = config.get_as_dict() if hasattr(config, "get_as_dict") else config or {}
            mode = "offline" if offline else "online"
            self.run = wandb.init(
                project=project_name,
                name=run_name,
                config=config,
                mode=mode,
                tags=tags or []
            )
        else:
            # Even when run is pre-existing, log tags or configs if provided
            if config:
                wandb.config.update(config, allow_val_change=True)
            if tags:
                current_tags = list(self.run.tags) if self.run.tags is not None else []
                current_tags += [tag for tag in tags if tag not in current_tags]
                self.run.tags = current_tags

    def log_metric(self, name, value, step=None):
        wandb.log({name: value}, step=step)

    def log_artifact(self, artifact_path, artifact_name, artifact_type="file"):
        artifact = wandb.Artifact(artifact_name, type=artifact_type)
        artifact.add_file(artifact_path)
        wandb.log_artifact(artifact)

    def finish(self):
        if self.run:
            wandb.finish()
