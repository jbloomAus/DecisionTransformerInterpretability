# Installation

## Package Overview

Currently the package has 3 important components:
- The ppo subpackage. This package enables users to train a PPO agent on a gym environment and store the trajectories generated *during training*.
- The decision_transformer subpackage. This package contains the decision transformer implementation.
- the streamlit app. This app (currently in development) enables researchers to play minigrid games whilst observing the decision transformer's predictions/activations.

## Running the scripts

Example bash scripts are provided in the scripts folder. They make use of argparse interfaces in the package.
