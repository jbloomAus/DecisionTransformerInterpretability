# Tutorial

## Overview

## The basics (running the scripts)

### Training a PPO
**insert description of script and what it does**
To train a PPO you will need to run the run_ppo.sh script. You will then be prompted to create an account with Weight & Biases (W&B).
`scripts/run_ppo.sh`
**insert more about W&B**
Once your account has been created and you have entered your API key, the script will begin training your PPO.
**insert image**
When the script is complete you will be able to view the project data in W&B via the link. It will be similar to https://wand.ai/username/PPO-MiniGrid. In addition, you will also see a path to the trajectories generated through training which will be required in the next step, Running the decision transformer.

### Running the decision Transformer
Copy and paste the trajectory file path (see previous step) into the run_decision_transformer.sh script as show below.
`--trajectory_path trajectories/0efb210c-6f04-4478-8ae1-e2c4ab147e1d.gz \`
Run the script
`scripts/run_decision_transformer.sh`
**insert info and about what is happening**

Once the script has finished, navigate to the W&B dashboard for the Decision Transformer (link provided towards end of script output).
Head to the artifacts (**insert name of symbol**) and download the model file. Add this file to your local repository.

### Calibrating your model
Calibration can be done through the run_calibration.sh script. Copy and paste the path of the model file you downloaded in the previous step as shown below.

`--model_path "artifacts/MiniGrid-Dynamic-Obstacles-8x8-v0__MiniGrid-Dynamic-Obstacles-8x8-v0__1__1675306594:v0/MiniGrid-Dynamic-Obstacles-8x8-v0__MiniGrid-Dynamic-Obstacles-8x8-v0__1__1675306594.pt" \`

Run the file.
`scripts/run_calibration.sh`
