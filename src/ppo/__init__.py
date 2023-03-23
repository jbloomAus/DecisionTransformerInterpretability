"""The PPO (Proximal Policy Optimization) subpackage contains modules and classes for training an agent using the PPO algorithm on an OpenAI gym (now gymnasium) environment.

Modules:
    - agent: Contains the Agent class, which encapsulates the neural networks used to represent the policy and value function, as well as the methods used for interacting with the environment and updating the network weights.
    - memory: Contains the Memory class, which stores experiences collected during the rollout phase and provides methods for computing advantages and generating minibatches for training.
    - utils: Contains utility functions used throughout the PPO subpackage.

Classes:
    - PPOArgs: A dataclass that stores hyperparameters and other configuration options for PPO training.
    - Minibatch: A dataclass that represents a minibatch of experiences, containing observations, actions, log probabilities, advantages, values, and returns.

Functions:
    - get_obs_preprocessor: A function that takes an OpenAI gym observation space and returns a function that preprocesses observations for use in neural networks.
    - check_and_upload_new_video: A function that checks for new video files in a specified directory and logs them to a WandB run if they are found.
    - train_ppo: The main function for training an agent using the PPO algorithm. Calls the Agent and Memory classes, and uses an optimizer and learning rate scheduler to update the network weights.

Dependencies:
    - gym (gymnasium): For interacting with OpenAI gym environments.
    - numpy: For numerical operations and array manipulation.
    - torch: For building and training neural networks.
    - einops: For efficient manipulation of tensor dimensions.
    - wandb: For logging and visualizing training metrics and video.
"""
