#

# echo the python path
which python

EXP_NAME="MiniGrid-SimpleCrossingS9N3-v0" wandb sweep --project $EXP_NAME \
     /Users/josephbloom/GithubRepositories/DecisionTransformerInterpretability/sweeps/ppo_sweep_template.yml
