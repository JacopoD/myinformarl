from runner import MPERunner
import torch
import numpy as np
import config as local_config
from pathlib import Path
import os
import wandb

from envs.env_wrappers import (
    SubprocVecEnv,
    DummyVecEnv,
    GraphSubprocVecEnv,
    GraphDummyVecEnv,
)

from multiagent.MPE_env import MPEEnv

def main():
    config = local_config

    run_dir = setup_dirs(config)

    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    np.random.seed(config.seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    envs = make_train_env(config)
    eval_envs = make_eval_env(config) if config.use_eval else None

    runner = MPERunner(config, envs, eval_envs, device, run_dir)

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
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv(
            [get_env_fn(i) for i in range(config.n_eval_rollout_threads)]
        )
    
def setup_dirs(config):
    
    # print_args(config)

    # setup file to output tensorboard, hyperparameters, and saved models
    run_dir = (
        Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0] + "/results")
        / config.env_name
        / config.scenario_name
        / config.algorithm_name
        / config.experiment_name
    )

    if not run_dir.exists():
        os.makedirs(str(run_dir))

    if config.use_wandb:
        # init wandb
        print("_" * 50)
        print("Creating wandboard...")
        print("_" * 50)
        run = wandb.init(
            config=config,
            project=config.project_name,
            # project=config.env_name,
            name=str(config.algorithm_name)
            + "_"
            + str(config.experiment_name)
            + "_seed"
            + str(config.seed),
            # group=config.scenario_name,
            dir=str(run_dir),
        )
    else:
        if not run_dir.exists():
            curr_run = "run1"
        else:
            exst_run_nums = [
                int(str(folder.name).split("run")[1])
                for folder in run_dir.iterdir()
                if str(folder.name).startswith("run")
            ]
            if len(exst_run_nums) == 0:
                curr_run = "run1"
            else:
                curr_run = "run%i" % (max(exst_run_nums) + 1)
        run_dir = run_dir / curr_run
        if not run_dir.exists():
            os.makedirs(str(run_dir))

    return run_dir

if __name__ == "__main__":
    main()