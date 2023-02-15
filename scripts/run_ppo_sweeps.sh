#

# echo the python path
which python

# Current bug, you should run the export command before running the script
echo 'Current bug, you should run the export command before running the script'
export EXP_NAME="MiniGrid-RedBlueDoors-8x8-v0"
EXP_NAME="MiniGrid-RedBlueDoors-8x8-v0" wandb sweep --project ${EXP_NAME}-Sweep \
    /Users/josephbloom/GithubRepositories/DecisionTransformerInterpretability/sweeps/ppo_sweep_template.yml
