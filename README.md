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

