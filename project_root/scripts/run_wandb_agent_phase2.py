# run_wandb_agent_phase1.py

import wandb
from project_root.scripts.run_sweeps_phase_2 import main  # this is your hydra @main function

def sweep_run():
    run = wandb.init()
    main()
    wandb.finish()

if __name__ == "__main__":
    sweep_id = "alexdoro_code/RepClassifier/2f73hkdr"
    wandb.agent(
        sweep_id=sweep_id,
        function=sweep_run,
        entity="alexdoro_code",         # Your W&B username or team
        project="RepClassifier",        # Name of the project (W&B will auto-create if it doesn't exist)
        count=500
        )


