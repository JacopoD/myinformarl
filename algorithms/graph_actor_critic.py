import argparse
from typing import Tuple, List

import gym
import torch
from torch import Tensor
import torch.nn as nn
from algorithms.utils.util import init, check
from algorithms.utils.mlp import MLPBase
from algorithms.utils.rnn import RNNLayer
from algorithms.utils.act import ACTLayer
from algorithms.utils.popart import PopArt
from algorithms.utils.gnn import GNN
from utils.util import get_shape_from_obs_space


class GraphActor(nn.Module):
    """
    Actor network class for MAPPO. Outputs actions given observations.
    """

    def __init__(self, config) -> None:
        super(GraphActor, self).__init__()

        gnn = GNN(...)

        mlp_input_dim = gnn.out_dim + observation.shape

        self.mlp = MLPBase(args=config, input_dim=mlp_input_dim)

        self.rnn = RNNLayer(
            self.hidden_size,
            self.hidden_size,
            self._recurrent_N,
            self._use_orthogonal,
        )

        self.act = ACTLayer(
            "Discrete", self.hidden_size, self._use_orthogonal, self._gain
        )
        pass

    def forward(self, local_obs, node_obs, adj, ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Compute actions from the given inputs.
        """

        neighbour_features = self.gnn(node_obs, adj, n_agents, threads???)

        actor_features = torch.cat([local_obs, nbd_features], dim=1)

        actor_features = self.mlp(actor_features)

        actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        actions, action_log_probs = self.act(actor_features)

        return (actions, action_log_probs, rnn_states)

    def evaluate_actions(self) -> Tuple[Tensor, Tensor]:
        """
        Compute log probability and entropy of given actions.
        """

        nbd_features = self.gnn_base(node_obs, adj, agent_id)
        actor_features = torch.cat([obs, nbd_features], dim=1)
        actor_features = self.base(actor_features)

        actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        action_log_probs, dist_entropy = self.act.evaluate_actions(
            actor_features,
            action
        )

        return (action_log_probs, dist_entropy)


class GraphCritic(nn.Module):
    """
    Critic network class for MAPPO. Outputs value function predictions
    given centralized input (MAPPO) or local observations (IPPO).
    """

    def __init__(self) -> None:
        super(GraphCritic, self).__init__()
        pass

    def forward(self) -> Tuple[Tensor, Tensor]:
        """
        Compute actions from the given inputs.
        """
        pass
