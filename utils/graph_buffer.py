import torch
import numpy as np


class GReplayBuffer(object):
    def __init__(
        self,
        config,
        local_obs_shape,
        node_obs_shape,
        share_obs_space,
        action_space,
        adj_obs_space,
    ) -> None:
        self.max_len = config.episode_length + 1
        self.pointer = 0
        self.threads = config.n_rollout_threads
        self.n_agents = config.num_agents

        self.local_obs = np.zeros(
            (self.max_len, self.threads, self.n_agents, *local_obs_shape)
        )

        self.node_obs = np.zeros(
            (self.max_len, self.threads, self.n_agents, *node_obs_shape)
        )

        # self.local_obs[0] = local_obs.copy()

        # >         env.step(actions_env)
        # (array([[[-5.00000000e-001,  0.00000000e+000,  3.50972806e-001,
        #           9.79886752e-001,  9.95864451e-002, -1.43387722e+000],
        #         [ 1.74484650e-140, -5.00000000e-001, -2.17071090e-001,
        #          -7.34066096e-002,  7.20554762e-002, -6.99667440e-001],
        #         [-1.03436039e-273,  5.00000000e-001, -9.78143180e-001,
        #          -2.28807363e-001,  1.38946968e+000,  6.61274687e-001]]]), array([[[0],
        #         [1],
        #         [2]]])

        pass
