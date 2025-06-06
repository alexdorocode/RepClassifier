# run_wandb_agent_phase1.py

import wandb
from project_root.scripts.run_sweeps_phase_1 import main  # this is your hydra @main function

def sweep_run():
    run = wandb.init()
    main()
    wandb.finish()

if __name__ == "__main__":
    sweep_id = "alexdoro_code/RepClassifier/s0eehaax"
    wandb.agent(sweep_id, function=sweep_run, count=200)  # Adjust count as needed
