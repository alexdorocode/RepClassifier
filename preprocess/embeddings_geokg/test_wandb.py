import wandb

# Initialize a test run
wandb.init(project="test-wandb-setup")

# Log a simple metric
wandb.log({"test_metric": 1.0})

# Finish the run
wandb.finish()