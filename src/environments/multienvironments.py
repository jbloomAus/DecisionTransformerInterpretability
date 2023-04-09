import gymnasium as gym
import numpy as np


class MultiEnvSampler(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 10,
    }

    def __init__(self, envs, p=None, render_mode="rgb_array"):
        if len(envs) < 2:
            raise ValueError(
                "MultiEnvSampler requires at least two environments"
            )
        self.envs = envs
        # don't call it num_envs because this interacts badly with RecordEpisodeStatistics wrapper. Solve later.
        self.n_envs = len(envs)
        self.env_names = [env.unwrapped.__class__.__name__ for env in envs]
        self.p = p
        if self.p is None:
            self.p = np.ones(self.n_envs) / self.n_envs
        elif len(self.p) != self.n_envs:
            raise ValueError(
                "The length of p must be equal to the number of environments"
            )

        obs_space = self.envs[0].observation_space
        action_space = self.envs[0].action_space

        self.env_id = 0

        self._homogenize_mission_spaces()
        for env in self.envs[1:]:
            if not env.observation_space == obs_space:
                raise ValueError(
                    f"All environments must have the same observation space\n{env.observation_space} != {obs_space}"
                )

            if not env.action_space == action_space:
                raise ValueError(
                    f"All environments must have the same action space\n{env.action_space} != {action_space}"
                )

        self.render_mode = render_mode
        self.observation_space = obs_space
        self.action_space = action_space

    def reset(self, seed=None, all_envs=False, options=None):
        np.random.seed(seed)
        self.env_id = np.random.choice(self.n_envs, p=self.p)
        if all_envs:
            return [env.reset() for env in self.envs]
        return self.envs[self.env_id].reset()

    def step(self, action):
        obs, reward, done, info, truncated = self.envs[self.env_id].step(
            action
        )
        return obs, reward, done, info, truncated

    def render(self):
        return self.envs[self.env_id].render()

    def close(self):
        for env in self.envs:
            env.close()

    def get_current_env_name(self):
        return self.env_names[self.env_id]

    def _sample_env_id(self):
        env_id = np.random.choice(self.n_envs, p=self.p)
        return env_id

    def _homogenize_mission_spaces(self):
        """resets all mission spaces to be equal to the first env"""
        # set mission space to the first env with a mission space's mission space
        if self.envs[0].observation_space["mission"] is not None:
            mission_space = self.envs[0].observation_space["mission"]
        else:
            mission_space = None
        for env in self.envs[1:]:
            if env.observation_space["mission"] is not None:
                pass
            env.observation_space["mission"] = mission_space
