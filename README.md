# DecisionTransformerInterpretability

https://arxiv.org/pdf/2106.01345.pdf

Initial Goals: produce a decision transformer and train it successfully on a task from minigrid. 

In steps:
1. Implement a minigrid environment [using gym-minigrid](https://github.com/Farama-Foundation/Minigrid) (done!)
2. Implement a decision transformer [using transformerlens](https://github.com/neelnanda-io/TransformerLens) 
3. Train an Agent such as PPO or DQN on an environment. Done when performance is good. 
4. Write a wrapper to save episodes and store as offline trajectories. Done when episodes are saved. Might look a lot like [this](https://github.com/Farama-Foundation/D4RL/blob/master/scripts/generation/generate_minigrid_fourroom_data.py)
5. Write a training function to [train](https://github.com/kzl/decision-transformer/blob/master/gym/decision_transformer/training/act_trainer.py) a decision transformer on the offline trajectories. Done when decision transformer is trained. 
6. Evaluate the decision transformer on minigrid by visually inspecting the performance.
7. Use interpretability methods to examine performance.

# Setting up the environment

I haven't been too careful about this yet. Using python 3.9.15 with the requirements.txt and requirements_dev.txt files. We're using the V2 branch of transformer lens and Minigrid 2.1.0.

The docker file should work and we can make use of it more when the project is further ahead/if we are alternativing developers frequently and have any differential behavior. 

```bash
./scripts/build_docker.sh
./scripts/run_docker.sh
```

Then you can ssh into the docker and a good ide will bring credentials etc.

# Development so far

I'm writing this midways since I'm going on holiday for a week and others might want to do some development. 

The key components so far are:
- PPO. The ppo subpackage runs a PPO agent on a minigrid (or other gymnasium environment).
- Trajectory Writer/Reader Utils. These are working but probably can only be signed off on once we've doing offline training successfully for behavioral cloning and decision transformers. 
- Decision Transformer. The decision transformer is implemented in the decision_transformer module. It currently expects an RGB image but we can fix that easily (and should probably let it handle either based on keyword arguments).

In the dev notebook, I'm working on getting a dataset writer which reads stored trajectories and parcels them out as batched padded episodes with Reward to go. 

# Next Steps

My main goals for this project go something like:
- Get a decision transformer working on minigrid.
    - Get a dataset module working
    - Writing a training loop. 
    - Making sure we understand how to configure the decision transformer for minigrid.

After this we have several directions worth pursuing:
    - interpretability analysis (obviously)
    - making a transformer (just behavioral clonning) -> this will also require configuring the dataset module appropriately. 

# Tests:


Ensure that the run_tests.sh script is executable:
```bash
chmod a+x ./scripts/run_tests.sh
```

Run the tests:
```bash
./scripts/run_tests.sh
```

You should see something like this after the tests run. This is the coverage report. Ideally this is 100% but we're not there yet. Furthermore, it will be 100% long before we have enough tests. But if it's 100% and we have performant code with agents training and stuff otherwise working, that's pretty good.

```bash

---------- coverage: platform darwin, python 3.9.15-final-0 ----------
Name                                Stmts   Miss  Cover   Missing
-----------------------------------------------------------------
src/__init__.py                         0      0   100%
src/decision_transformer.py           132      8    94%   39, 145, 151, 156-157, 221, 246, 249
src/ppo.py                             20     20     0%   2-28
src/ppo/__init__.py                     0      0   100%
src/ppo/agent.py                      109     10    91%   41, 45, 112, 151-157
src/ppo/compute_adv_vectorized.py      30     30     0%   1-65
src/ppo/memory.py                      88     11    88%   61-64, 119-123, 147-148
src/ppo/my_probe_envs.py               99      9    91%   38, 42-44, 74, 99, 108, 137, 168
src/ppo/train.py                       69      6    91%   58, 74, 94, 98, 109, 113
src/ppo/utils.py                      146     54    63%   41-42, 61-63, 69, 75, 92-96, 110-115, 177-206, 217-235
src/utils.py                           40     17    58%   33-38, 42-65, 73, 76-79
src/visualization.py                   25     25     0%   1-34
-----------------------------------------------------------------
TOTAL                                 758    190    75%
```