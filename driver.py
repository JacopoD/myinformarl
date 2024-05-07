from runner import MPERunner
import torch
import numpy as np
import config as local_config

from envs.env_wrappers import (
    SubprocVecEnv,
    DummyVecEnv,
    GraphSubprocVecEnv,
    GraphDummyVecEnv,
)

from multiagent.MPE_env import MPEEnv

def main():
    config = local_config
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    np.random.seed(config.seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    envs = make_train_env(config)
    eval_envs = make_eval_env(config) if config.use_eval else None

    runner = MPERunner(config, envs, eval_envs, device)

    runner.run()


    envs.close()
    if eval_envs is not None:
        eval_envs.close()
    


def make_train_env(config):
    def get_env_fn(rank: int):
        def init_env():
            if config.env_name == "MPE":
                env = MPEEnv(config)
            else:
                print(f"Can not support the {config.env_name} environment")
                raise NotImplementedError
            env.seed(config.seed + rank * 1000)
            return env

        return init_env

    if config.n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(config.n_rollout_threads)])


def make_eval_env(config):
    def get_env_fn(rank: int):
        def init_env():
            if config.env_name == "MPE":
                env = MPEEnv(config)
            else:
                print(f"Can not support the {config.env_name} environment")
                raise NotImplementedError
            env.seed(config.seed * 50000 + rank * 10000)
            return env

        return init_env

    if config.n_eval_rollout_threads == 1:
        if config.env_name == "GraphMPE":
            return GraphDummyVecEnv([get_env_fn(0)])
        return DummyVecEnv([get_env_fn(0)])
    else:
        if config.env_name == "GraphMPE":
            return GraphSubprocVecEnv(
                [get_env_fn(i) for i in range(config.n_rollout_threads)]
            )
        return SubprocVecEnv(
            [get_env_fn(i) for i in range(config.n_eval_rollout_threads)]
        )

if __name__ == "__main__":
    main()