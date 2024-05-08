import torch
import numpy as np


class GReplayBuffer(object):
    def __init__(
        self,
        config,
        local_obs_shape,
        node_obs_shape,
        share_obs_shape,
        action_space,
        adj_obs_shape,
    ) -> None:
        self.max_len = config.episode_length + 1
        self.pointer = 0
        self.threads = config.n_rollout_threads
        self.n_agents = config.num_agents
        self.n_recurrent = config.recurrent_N
        self.hidden_size = config.hidden_size
        self.gamma = config.gamma

        self.local_obs = np.zeros(
            (self.max_len, self.threads, self.n_agents, *local_obs_shape)
        )

        self.node_obs = np.zeros(
            (self.max_len, self.threads, self.n_agents, *node_obs_shape)
        )

        self.share_obs = np.zeros(
            (self.max_len, self.threads, self.n_agents, *share_obs_shape)
        )

        self.adj_obs = np.zeros(
            (self.max_len, self.threads, self.n_agents, *adj_obs_shape)
        )

        self.actions = np.zeros(
            (self.max_len, self.threads, self.n_agents, action_space)
        )

        self.rewards = np.zeros((self.max_len, self.threads, self.n_agents, 1))

        self.values = np.zeros((self.max_len, self.threads, self.n_agents, 1))

        self.cumulative_rewards = np.zeros(
            (self.max_len, self.threads, self.n_agents, 1)
        )

        self.done_masks = np.ones((self.max_len, self.threads, self.n_agents, 1))

        self.rnn_states_actor = np.zeros(
            (
                self.max_len,
                self.threads,
                self.n_agents,
                self.n_recurrent,
                self.hidden_size,
            )
        )

        self.rnn_states_critic = np.zeros(
            (
                self.max_len,
                self.threads,
                self.n_agents,
                self.n_recurrent,
                self.hidden_size,
            )
        )

    def insert(
        self,
        local_obs,
        node_obs,
        share_obs,
        adj_obs,
        rewards,
        actions,
        # values,
        dones,
        rnn_states_actor,
        rnn_states_critic,
    ):
        self.done_masks[self.pointer + 1] = np.ones((self.threads, self.n_agents, 1))
        self.done_masks[self.pointer + 1][dones] = np.zeros(((dones).sum(), 1))

        if not ((rnn_states_actor is None) or (rnn_states_critic is None)):
            rnn_states_actor[dones] = np.zeros(
                ((dones).sum(), self.n_recurrent, self.hidden_size)
            )
            rnn_states_critic[dones] = np.zeros(
                ((dones).sum(), *self.rnn_states_critic.shape[3:])
            )
            self.rnn_states_actor[self.pointer + 1] = rnn_states_actor.copy()
            self.rnn_states_critic[self.pointer + 1] = rnn_states_critic.copy()

        self.local_obs[self.pointer + 1] = local_obs.copy()
        self.node_obs[self.pointer + 1] = node_obs.copy()
        self.share_obs[self.pointer + 1] = share_obs.copy()
        self.adj_obs[self.pointer + 1] = adj_obs.copy()
        self.rewards[self.pointer + 1] = rewards.copy()
        self.actions[self.pointer + 1] = actions.copy()
        # self.values[self.pointer + 1] = values.copy()
        # self.done_masks[self.pointer + 1] = done_masks

        self.pointer = (self.pointer + 1) % (self.max_len - 1)

    def cum_reward(self, next_val_pred):
        self.cumulative_rewards[-1] = next_val_pred
        for step in range(self.rewards.shape[0])[::-1]:
            self.cumulative_rewards[step] = (
                self.cumulative_rewards[step + 1]
                * self.gamma
                * self.done_masks[step + 1]
                + self.rewards[step]
            )

    def naive_recurrent_generator(self, advantages, num_mini_batch):
        pass