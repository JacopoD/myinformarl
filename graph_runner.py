import time
import numpy as np
from numpy import ndarray as arr
from typing import Tuple
import torch
from base_runner import Runner
import imageio
from utils.utils import _t2n, _flatten


class GMPERunner(Runner):
    """
    Runner class to perform training, evaluation and data
    collection for the MPEs. See parent class for details
    """

    dt = 0.1

    def __init__(self, config, envs, eval_envs, device, run_dir):
        super(GMPERunner, self).__init__(config, envs, eval_envs, device, run_dir)

    def run(self):
        self.warmup()

        episodes = (
            int(self.num_env_steps) // self.episode_length // self.n_rollout_threads
        )

        # This is where the episodes are actually run.
        for episode in range(episodes):
            for step in range(self.episode_length):
                # Sample actions
                # (
                #     values,
                #     actions,
                #     action_log_probs,
                #     rnn_states,
                #     rnn_states_critic,
                #     actions_env,
                # ) = self.collect(step)

                actions = [s.sample() for s in self.envs.action_space] + [
                    s.sample() for s in self.envs.action_space
                ]
                actions = np.expand_dims(actions, 1)
                actions = np.array(np.split(actions, self.n_rollout_threads))
                actions_env = self.one_hot_encode_actions(actions)
                values = None
                action_log_probs = None
                rnn_states_actor = None
                rnn_states_critic = None

                # Obs reward and next obs
                obs, agent_ids, node_obs, adj, rewards, dones, infos = self.envs.step(
                    actions_env
                )

                if self.use_centralized_V:
                    share_obs = obs.reshape(self.n_rollout_threads, -1)
                    share_obs = np.expand_dims(share_obs, 1).repeat(
                        self.num_agents, axis=1
                    )
                else:
                    share_obs = obs

                self.buffer.insert(
                    local_obs=obs,
                    node_obs=node_obs,
                    share_obs=share_obs,
                    adj_obs=adj,
                    rewards=rewards,
                    actions=actions,
                    # values=values,
                    dones=dones,
                    rnn_states_actor=rnn_states_actor,
                    rnn_states_critic=rnn_states_critic,
                    agent_ids=agent_ids
                )


            # compute return and update network
            self.compute()
            train_infos = self.train()
            
            self.fake_reset()
            raise NotImplementedError

            # post process
            total_num_steps = (
                (episode + 1) * self.episode_length * self.n_rollout_threads
            )

            # save model
            if episode % self.save_interval == 0 or episode == episodes - 1:
                self.save()

            # log information
            if episode % self.log_interval == 0:
                env_infos = self.process_infos(infos)

                avg_ep_rew = np.mean(self.buffer.rewards) * self.episode_length
                train_infos["average_episode_rewards"] = avg_ep_rew
                print(
                    f"Average episode rewards is {avg_ep_rew:.3f} \t"
                    f"Total timesteps: {total_num_steps} \t "
                    f"Percentage complete {total_num_steps / self.num_env_steps * 100:.3f}"
                )
                self.log_train(train_infos, total_num_steps)
                self.log_env(env_infos, total_num_steps)

            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)

    def one_hot_encode_actions(self, actions):
        return np.squeeze(np.eye(self.envs.action_space[0].n)[actions], 2)

    def fake_reset(self):
        self.buffer.share_obs[0] = self.buffer.share_obs[-1].copy()
        self.buffer.local_obs[0] = self.buffer.local_obs[-1].copy()
        self.buffer.node_obs[0] = self.buffer.node_obs[-1].copy()
        self.buffer.adj_obs[0] = self.buffer.adj_obs[-1].copy()
        self.buffer.agent_ids[0] = self.buffer.agent_ids[-1].copy()
        # self.buffer.share_agent_id[0] = self.buffer.share_agent_id[-1].copy()
        self.buffer.rnn_states_actor[0] = self.buffer.rnn_states_actor[-1].copy()
        self.buffer.rnn_states_critic[0] = self.buffer.rnn_states_critic[-1].copy()
        self.buffer.done_masks[0] = self.buffer.done_masks[-1].copy()

    def warmup(self):
        obs, agent_ids, node_obs, adj = self.envs.reset()

        if self.use_centralized_V:
            # (n_rollout_threads, n_agents, feats) -> (n_rollout_threads, n_agents*feats)
            share_obs = obs.reshape(self.n_rollout_threads, -1)
            # (n_rollout_threads, n_agents*feats) -> (n_rollout_threads, n_agents, n_agents*feats)
            share_obs = np.expand_dims(share_obs, 1).repeat(self.num_agents, axis=1)

            # (n_rollout_threads, n_agents, 1) -> (n_rollout_threads, n_agents*1)
            # share_agent_id = agent_id.reshape(self.n_rollout_threads, -1)
            # (n_rollout_threads, n_agents*1) -> (n_rollout_threads, n_agents, n_agents*1)
            # share_agent_id = np.expand_dims(share_agent_id, 1).repeat(
            #     self.num_agents, axis=1
            # )
        else:
            share_obs = obs
            # share_agent_id = agent_id

        self.buffer.local_obs[0] = obs.copy()
        self.buffer.node_obs[0] = node_obs.copy()
        self.buffer.share_obs[0] = share_obs.copy()
        self.buffer.adj_obs[0] = adj.copy()
        self.buffer.agent_ids[0] = agent_ids.copy()
        # self.buffer.share_agent_id[0] = share_agent_id.copy()

    @torch.no_grad()
    def collect(self, step: int) -> Tuple[arr, arr, arr, arr, arr, arr]:
        self.trainer.prep_rollout()
        (
            value,
            action,
            action_log_prob,
            rnn_states,
            rnn_states_critic,
        ) = self.trainer.policy.get_actions(
            _flatten(self.n_rollout_threads, self.num_agents, self.buffer.share_obs[step]),
            _flatten(self.n_rollout_threads, self.num_agents, self.buffer.local_obs[step]),
            _flatten(self.n_rollout_threads, self.num_agents, self.buffer.node_obs[step]),
            _flatten(self.n_rollout_threads, self.num_agents, self.buffer.adj_obs[step]),
            _flatten(self.n_rollout_threads, self.num_agents, self.buffer.rnn_states_actor[step]),
            _flatten(self.n_rollout_threads, self.num_agents, self.buffer.rnn_states_critic[step]),
            _flatten(self.n_rollout_threads, self.num_agents, self.buffer.done_masks[step]),
            _flatten(self.n_rollout_threads, self.num_agents, self.buffer.agent_ids[step])
            
        )
        # [self.envs, agents, dim]
        values = np.array(np.split(_t2n(value), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
        action_log_probs = np.array(
            np.split(_t2n(action_log_prob), self.n_rollout_threads)
        )
        rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))
        rnn_states_critic = np.array(
            np.split(_t2n(rnn_states_critic), self.n_rollout_threads)
        )
        # rearrange action
        # if self.envs.action_space[0].__class__.__name__ == "MultiDiscrete":
        #     for i in range(self.envs.action_space[0].shape):
        #         uc_actions_env = np.eye(self.envs.action_space[0].high[i] + 1)[
        #             actions[:, :, i]
        #         ]
        #         if i == 0:
        #             actions_env = uc_actions_env
        #         else:
        #             actions_env = np.concatenate((actions_env, uc_actions_env), axis=2)
        # if self.envs.action_space[0].__class__.__name__ == "Discrete":
        #     actions_env = np.squeeze(np.eye(self.envs.action_space[0].n)[actions], 2)
        # else:
        #     raise NotImplementedError
        if self.envs.action_space[0].__class__.__name__ == "Discrete":
            actions_env = self.one_hot_encode_actions(actions)
        else:
            raise NotImplementedError

        return (
            values,
            actions,
            action_log_probs,
            rnn_states,
            rnn_states_critic,
            actions_env,
        )

    @torch.no_grad()
    def compute(self):
        """Calculate returns for the collected data."""
        self.trainer.prep_rollout()
        
        next_val_pred = self.trainer.policy.get_values(
            # drop one dimention for batch computation
            # from (threads, n_agents, *data_shape) -> (threads * n_agents, *data_shape)
            # _flatten(self.n_rollout_threads, self.num_agents, self.buffer.share_obs[-1]),
            _flatten(self.n_rollout_threads, self.num_agents, self.buffer.node_obs[-1]),
            _flatten(self.n_rollout_threads, self.num_agents, self.buffer.adj_obs[-1]),
            _flatten(self.n_rollout_threads, self.num_agents, self.buffer.rnn_states_critic[-1]),
            _flatten(self.n_rollout_threads, self.num_agents, self.buffer.done_masks[-1]),
            # _flatten(self.n_rollout_threads, self.num_agents, self.buffer.agent_ids[-1])
        )
        # put back thread dimension
        next_val_pred = np.array(np.split(_t2n(next_val_pred), self.n_rollout_threads))
        # self.buffer.compute_reward(next_values, self.trainer.value_normalizer)
        self.buffer.cum_reward(next_val_pred)

    @torch.no_grad()
    def eval(self, total_num_steps: int):
        eval_episode_rewards = []
        eval_obs, eval_agent_id, eval_node_obs, eval_adj = self.eval_envs.reset()

        eval_rnn_states = np.zeros(
            (self.n_eval_rollout_threads, *self.buffer.rnn_states.shape[2:]),
            dtype=np.float32,
        )
        eval_masks = np.ones(
            (self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32
        )

        for eval_step in range(self.episode_length):
            self.trainer.prep_rollout()
            eval_action, eval_rnn_states = self.trainer.policy.act(
                np.concatenate(eval_obs),
                np.concatenate(eval_node_obs),
                np.concatenate(eval_adj),
                np.concatenate(eval_agent_id),
                np.concatenate(eval_rnn_states),
                np.concatenate(eval_masks),
                deterministic=True,
            )
            eval_actions = np.array(
                np.split(_t2n(eval_action), self.n_eval_rollout_threads)
            )
            eval_rnn_states = np.array(
                np.split(_t2n(eval_rnn_states), self.n_eval_rollout_threads)
            )

            if self.eval_envs.action_space[0].__class__.__name__ == "MultiDiscrete":
                for i in range(self.eval_envs.action_space[0].shape):
                    eval_uc_actions_env = np.eye(
                        self.eval_envs.action_space[0].high[i] + 1
                    )[eval_actions[:, :, i]]
                    if i == 0:
                        eval_actions_env = eval_uc_actions_env
                    else:
                        eval_actions_env = np.concatenate(
                            (eval_actions_env, eval_uc_actions_env), axis=2
                        )
            elif self.eval_envs.action_space[0].__class__.__name__ == "Discrete":
                eval_actions_env = np.squeeze(
                    np.eye(self.eval_envs.action_space[0].n)[eval_actions], 2
                )
            else:
                raise NotImplementedError

            # Obser reward and next obs
            (
                eval_obs,
                eval_agent_id,
                eval_node_obs,
                eval_adj,
                eval_rewards,
                eval_dones,
                eval_infos,
            ) = self.eval_envs.step(eval_actions_env)
            eval_episode_rewards.append(eval_rewards)

            eval_rnn_states[eval_dones == True] = np.zeros(
                ((eval_dones == True).sum(), self.recurrent_N, self.hidden_size),
                dtype=np.float32,
            )
            eval_masks = np.ones(
                (self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32
            )
            eval_masks[eval_dones == True] = np.zeros(
                ((eval_dones == True).sum(), 1), dtype=np.float32
            )

        eval_episode_rewards = np.array(eval_episode_rewards)
        eval_env_infos = {}
        eval_env_infos["eval_average_episode_rewards"] = np.sum(
            np.array(eval_episode_rewards), axis=0
        )
        eval_average_episode_rewards = np.mean(
            eval_env_infos["eval_average_episode_rewards"]
        )
        print(
            "eval average episode rewards of agent: "
            + str(eval_average_episode_rewards)
        )
        self.log_env(eval_env_infos, total_num_steps)

    @torch.no_grad()
    def render(self, get_metrics: bool = False):
        """
        Visualize the env.
        get_metrics: bool (default=False)
            if True, just return the metrics of the env and don't render.
        """
        envs = self.envs

        all_frames = []
        rewards_arr, success_rates_arr, num_collisions_arr, frac_episode_arr = (
            [],
            [],
            [],
            [],
        )

        for episode in range(self.all_args.render_episodes):
            obs, agent_id, node_obs, adj = envs.reset()
            if not get_metrics:
                if self.all_args.save_gifs:
                    image = envs.render("rgb_array")[0][0]
                    all_frames.append(image)
                else:
                    envs.render("human")

            rnn_states = np.zeros(
                (
                    self.n_rollout_threads,
                    self.num_agents,
                    self.recurrent_N,
                    self.hidden_size,
                ),
                dtype=np.float32,
            )
            masks = np.ones(
                (self.n_rollout_threads, self.num_agents, 1), dtype=np.float32
            )

            episode_rewards = []

            for step in range(self.episode_length):
                calc_start = time.time()

                self.trainer.prep_rollout()
                action, rnn_states = self.trainer.policy.act(
                    np.concatenate(obs),
                    np.concatenate(node_obs),
                    np.concatenate(adj),
                    np.concatenate(agent_id),
                    np.concatenate(rnn_states),
                    np.concatenate(masks),
                    deterministic=True,
                )
                actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
                rnn_states = np.array(
                    np.split(_t2n(rnn_states), self.n_rollout_threads)
                )

                if envs.action_space[0].__class__.__name__ == "MultiDiscrete":
                    for i in range(envs.action_space[0].shape):
                        uc_actions_env = np.eye(envs.action_space[0].high[i] + 1)[
                            actions[:, :, i]
                        ]
                        if i == 0:
                            actions_env = uc_actions_env
                        else:
                            actions_env = np.concatenate(
                                (actions_env, uc_actions_env), axis=2
                            )
                elif envs.action_space[0].__class__.__name__ == "Discrete":
                    actions_env = np.squeeze(np.eye(envs.action_space[0].n)[actions], 2)
                else:
                    raise NotImplementedError

                # Obser reward and next obs
                obs, agent_id, node_obs, adj, rewards, dones, infos = envs.step(
                    actions_env
                )
                episode_rewards.append(rewards)

                rnn_states[dones == True] = np.zeros(
                    ((dones == True).sum(), self.recurrent_N, self.hidden_size),
                    dtype=np.float32,
                )
                masks = np.ones(
                    (self.n_rollout_threads, self.num_agents, 1), dtype=np.float32
                )
                masks[dones == True] = np.zeros(
                    ((dones == True).sum(), 1), dtype=np.float32
                )

                if not get_metrics:
                    if self.all_args.save_gifs:
                        image = envs.render("rgb_array")[0][0]
                        all_frames.append(image)
                        calc_end = time.time()
                        elapsed = calc_end - calc_start
                        if elapsed < self.all_args.ifi:
                            time.sleep(self.all_args.ifi - elapsed)
                    else:
                        envs.render("human")

            env_infos = self.process_infos(infos)
            # print('_'*50)
            num_collisions = self.get_collisions(env_infos)
            frac, success = self.get_fraction_episodes(env_infos)
            rewards_arr.append(np.mean(np.sum(np.array(episode_rewards), axis=0)))
            frac_episode_arr.append(np.mean(frac))
            success_rates_arr.append(success)
            num_collisions_arr.append(num_collisions)
            # print(np.mean(frac), success)
            # print("Average episode rewards is: " +
            # str(np.mean(np.sum(np.array(episode_rewards), axis=0))))

        print(rewards_arr)
        print(frac_episode_arr)
        print(success_rates_arr)
        print(num_collisions_arr)

        if not get_metrics:
            if self.all_args.save_gifs:
                imageio.mimsave(
                    str(self.gif_dir) + "/render.gif",
                    all_frames,
                    duration=self.all_args.ifi,
                )
