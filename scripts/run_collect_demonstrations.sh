#!/bin/bash

# In order to run this script you first need to have a trained model checkpoint.
# You can train a model by running the ppo training code with an environment 
# of your choice as well as an architecture capable of solving it.
# This script gives you the ability to sample from an agent (which is hopefully)
# very performant but degrade the performance at varying levels using sampling 
# methods.
# This is useful when training DecisionTransformers because as long as the RTG
# is labelled correctly all of the trajectories are "valid". This helps you
# use a lot more varied data to train the agent, which likely results in
# more generalizable features/circuits which are easier to study. 
# Talk to me (Joseph if you want to know how to think about this better
# as there's more I can write here that I don't have time to right now. 

# Set default values for arguments. This agent is very highly performant
# but it is a little goodharted on the entropy bonus. There's a card for solving this. 
CHECKPOINT_PATH="artifacts/Test-PPO-LSTM_checkpoints:v16/Test-PPO-LSTM_11.pt"
NUM_ENVS=16 # I have 16 CPUs. 

# Name output well. Possibly automated this later.
TRAJECTORY_PATH="trajectories/MiniGrid-MemoryS7FixedStart-v0-Checkpoint11-VariedSamplingStrategies.gz"


# When deciding how many steps of a given trajectory, it's worth considering trajectory lengths
# which are mediate by decision quality. For example, at higher performance, in the memory task,
# we'll get short trajectories of < 10 steps. For low performance at high entropy, you're looking
# at 50 steps and a truncation event. For mid level entropy, you might fail early failure modes
# like hitting the wrong goal. Thus we need less steps for "high performance, low entropy config".
# and more for low performance, high entropy config.

# look at this image to calibrate on temperature -> decision prob effects:
# https://shivammehta25.github.io/posts/temperature-in-language-models-open-ai-whisper-probabilistic-machine-learning/different_temperatures.jpeg


# Sampling PPO agent/low entropy trajectoriess -> Contribute examples of high performance.

# Collect many demonstrations of the PPO roughly optimally
BASIC_STEPS=30000
# Very low temperature is essentially greedy behavior, a bit of this might be good
# to increase the probability of "prototypical" trajectories.
# TEMPERATURE_1=0.001
# TEMP_STEPS_1=10000

# if using top K, avoid the bottom action k actions increases the prob
# of the second best action being picked. This might be valuable since
# there are a number of binary decision points. 
TOPK_STEPS=15000
TOPK_VALUE=2

# However, mostly it would be good to have a good chunk of the trajectories be kinda warm but not
# too warm to ensure they still perform fairly well.
TEMPERATURE_3=5
TEMP_STEPS_3=20000

# Sampling high entroy -> Mostly contributing to highly variable failures.

# very high temperature is essentially random behavior, a bit of this might be good
# to increase the probability of sampling very confused but unbiased agents
TEMPERATURE_2=100
TEMP_STEPS_2=20000

# if using bottom K, avoid the top action is probably only import thing.
# if you avoid many more you probably just get nonsense trajectories.
BOTTOMK_STEPS=15000
BOTTOMK_VALUE=6 # 7 actions in the Memory env, I want to sample a strong number of consistently failing agents.

# Run the Python script with the default argument values
python -m src.collect_demonstrations_runner \
    --checkpoint "$CHECKPOINT_PATH" \
    --num_envs $NUM_ENVS \
    --trajectory_path "$TRAJECTORY_PATH" \
    --basic $BASIC_STEPS \
    --temp $TEMP_STEPS_2 $TEMPERATURE_2 \
    --temp $TEMP_STEPS_3 $TEMPERATURE_3 \
    --topk $TOPK_STEPS $TOPK_VALUE \
    --bottomk $BOTTOMK_STEPS $BOTTOMK_VALUE
    # --temp $TEMP_STEPS_1 $TEMPERATURE_1 \

# can evalute the results for this by
# 1. Looking at distribution of rewards/time to finish. I'm hoping for much flatter distribution in both. 
# 2. Once we build a feature for it, record trajectories. Possibly we should connect this up to wandb and 
# have a dashboard for it.
