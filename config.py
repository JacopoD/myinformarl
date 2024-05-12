seed = 97
env_name = "MPE"
scenario_name = "navigation_graph"
algorithm_name = "rmappo"
experiment_name = "aaa"
project_name = "informarl"
use_centralized_V = True
use_obs_instead_of_state = False
num_env_steps = 2000000
n_rollout_threads = 2
n_eval_rollout_threads = 2
n_render_rollout_threads = 5
use_linear_lr_decay = False
hidden_size = 64
use_wandb = True
use_render = False
recurrent_N = 1
lr = 7e-4
critic_lr = 7e-4
opti_eps = 1e-5
weight_decay = 0
gain = 0.001
use_policy_active_masks = True
use_value_active_masks = True
use_huber_loss = True
huber_delta = 10.0
use_proper_time_limits = False

gae_lambda = 0.95
use_gae = True
gamma = 0.99

use_grad_norm = True
use_max_grad_norm = True
max_grad_norm = 10.0


value_loss_coef = 1.0
entropy_coef = 0.01

num_mini_batch = 1

clip_param = 0.2
use_clipped_value_loss = True

ppo_epoch = 15

data_chunk_length = 10

split_batch = False

use_feature_normalization = True

use_valuenorm = True

layer_N = 1

use_stacked_frames = False
stacked_frames = 1

mask_agent_to_other = False


# env
world_size = 2
num_agents = 3
num_scripted_agents = 0
num_obstacles = 3
collaborative = True
max_speed = 2
collision_rew = 5.0
goal_rew = 5.0
min_dist_thresh = 0.05
use_dones = False
episode_length = 25
max_edge_dist = 1
obs_type = "local"
num_nbd_entities = 3
use_comm = False

# interval
save_interval = 1000
use_eval = True
eval_interval = 25
log_interval = 10

# dir
model_dir = None

graph_feat_type = "relative"
# actor_graph_aggr = "node"
# critic_graph_aggr = "global"