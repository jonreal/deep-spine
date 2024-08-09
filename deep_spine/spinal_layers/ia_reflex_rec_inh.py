from typing import Callable, Tuple
from gymnasium import spaces
import torch
from torch import nn
import numpy as np
from functools import partial

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.distributions import (
    Distribution,
    DiagGaussianDistribution
)
from stable_baselines3.common.type_aliases import PyTorchObs, Schedule

from deep_spine.spinal_layers.spinal_layers import *


class IaReflexRecInh(nn.Module):
    def __init__(
            self,
            num_motor_units,
            num_proprioceptive_features,
            num_encoding_features,
            rec_inh_recurrent_network_structure='aa',
            rec_inh_projection_network_structure='aa',
        ):
        super().__init__()
        self.nu = num_motor_units
        self.np = num_proprioceptive_features
        self.ne = num_encoding_features

        #self.spindle = Spindle(self.np, self.nu)
        #self.decoder = SupraspinalDecoder(self.ne, self.nu)

        self.spindleDecoder = SpindleAndDecoder(self.np+self.ne, self.nu)
        self.recinh = RecInh(
            self.nu, network_structure=rec_inh_recurrent_network_structure
        )
        self.reflex = IaReflex(self.nu)

        if rec_inh_projection_network_structure == 'aa':
            self.A = torch.nn.Parameter(torch.randn(self.nu))
        elif rec_inh_projection_network_structure == 'fc':
            self.A = torch.nn.Parameter(torch.randn(self.nu, self.nu))
        else:
            raise ValueError('Invalid network structure')

    def forward(self, features):
        A = (
            torch.sigmoid(self.A)
            if self.A.dim() == 2
            else create_agonist_antgonist_weight_matrix(torch.sigmoid(self.A))
        )

        length = features[:,:self.nu]
        velocity = features[:,self.nu:self.nu*2]
        posd = features[:,-self.ne:-self.ne+2]
        sigma = features[:,-self.ne+2:]

        proprioceptive_features = torch.cat((length, velocity), dim=1)
        encoding = torch.cat((posd, sigma), dim=1)
        #Ia_afferent = self.spindle(proprioceptive_features)
        #prediction = self.decoder(encoding)
        efferent_feature = self.spindleDecoder(
                torch.cat((proprioceptive_features, encoding), dim=1)
        )
        Ia_afferent = efferent_feature[:,:self.nu]
        prediction = efferent_feature[:,self.nu:]
        modulation = torch.matmul(self.recinh(Ia_afferent, prediction), A)
        activation = self.reflex(Ia_afferent, prediction, modulation)
        return activation

        #proprioceptive_features = features[:,:-self.ne]
        #encoding = features[:,-self.ne:]
        #Ia_afferent = self.spindle(proprioceptive_features)
        #prediction = self.decoder(encoding)
        #modulation = torch.matmul(self.recinh(Ia_afferent, prediction), A)
        #activation = self.reflex(Ia_afferent, prediction, modulation)
        #return activation

class IaReflexRecInhActorCritic(nn.Module):
    def __init__(
        self,
        num_motor_units: int = 6,
        num_proprioceptive_features: int = 12,
        num_encoding_features: int = 4,
        rec_inh_recurrent_network_structure: str = 'aa',
        rec_inh_projection_network_structure: str = 'aa',
        nn_size_critic: int = 32
    ):
        super().__init__()

        self.latent_dim_pi = num_motor_units
        self.latent_dim_vf = nn_size_critic

        self.policy_net = IaReflexRecInh(
            num_motor_units=num_motor_units,
            num_proprioceptive_features=num_proprioceptive_features,
            num_encoding_features=num_encoding_features,
            rec_inh_recurrent_network_structure=(
                rec_inh_recurrent_network_structure
            ),
            rec_inh_projection_network_structure=(
                rec_inh_projection_network_structure
            )
        )
        self.value_net = nn.Sequential(
            nn.Linear(
                num_proprioceptive_features + num_encoding_features,
                nn_size_critic
            ),
            nn.Tanh(),
            nn.Linear(nn_size_critic, nn_size_critic),
            nn.Tanh()
        )

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        return self.policy_net(features)

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        return self.value_net(features)


class IaReflexRecInhPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        num_motor_units: int = 6,
        num_proprioceptive_features: int = 12,
        num_encoding_features: int = 4,
        nn_size_critic: int = 32,
        rec_inh_recurrent_network_structure: str = 'aa',
        rec_inh_projection_network_structure: str = 'aa',
        *args,
        **kwargs
    ):

        self.num_motor_units = num_motor_units
        self.num_proprioceptive_features = num_proprioceptive_features
        self.num_encoding_features = num_encoding_features
        self.nn_size_critic = nn_size_critic
        self.rec_inh_recurrent_network_structure = (
            rec_inh_recurrent_network_structure
        )
        self.rec_inh_projection_network_structure = (
            rec_inh_projection_network_structure
        )

        # Disable orthogonal initialization
        kwargs["ortho_init"] = False
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = IaReflexRecInhActorCritic(
            num_motor_units=self.num_motor_units,
            num_proprioceptive_features=self.num_proprioceptive_features,
            num_encoding_features=self.num_encoding_features,
            nn_size_critic=self.nn_size_critic,
            rec_inh_recurrent_network_structure=(
                self.rec_inh_recurrent_network_structure
            ),
            rec_inh_projection_network_structure=(
                self.rec_inh_projection_network_structure
            )
        )
        self.mlp_extractor.latent_dim_vf = self.nn_size_critic


    def _build(self, lr_schedule: Schedule) -> None:
        """
        Create the networks and the optimizer.

        :param lr_schedule: Learning rate schedule
            lr_schedule(1) is the initial learning rate
        """
        self._build_mlp_extractor()

        # Custom action mapping
        # IaRecInh outputs mean actions, so last layer is not needed
        if isinstance(self.action_dist, DiagGaussianDistribution):
            self.action_net = nn.Identity()
            self.log_std = nn.Parameter(
                torch.ones(self.mlp_extractor.latent_dim_pi)
                * self.log_std_init,
                requires_grad=True
            )
        else:
            raise NotImplementedError(
                    f"Unsupported distribution '{self.action_dist}'."
            )

        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)
        # Init weights: use orthogonal initialization
        # with small initial weight for the output
        if self.ortho_init:
            # TODO: check for features_extractor
            # Values from stable-baselines.
            # features_extractor/mlp values are
            # originally from openai/baselines (default gains/init_scales).
            module_gains = {
                self.features_extractor: np.sqrt(2),
                self.mlp_extractor: np.sqrt(2),
                self.action_net: 0.01,
                self.value_net: 1,
            }
            if not self.share_features_extractor:
                # Note(antonin): this is to keep SB3 results
                # consistent, see GH#1148
                del module_gains[self.features_extractor]
                module_gains[self.pi_features_extractor] = np.sqrt(2)
                module_gains[self.vf_features_extractor] = np.sqrt(2)

            for module, gain in module_gains.items():
                module.apply(partial(self.init_weights, gain=gain))

        # Setup optimizer with initial learning rate
        self.optimizer = self.optimizer_class(
            self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs
        )  # type: ignore[call-arg]

    def _get_action_dist_from_latent(
            self, latent_pi: torch.Tensor
    ) -> Distribution:
        """
        Retrieve action distribution given the latent codes.

        :param latent_pi: Latent code for the actor
        :return: Action distribution
        """
        mean_actions = latent_pi

        if isinstance(self.action_dist, DiagGaussianDistribution):
            return self.action_dist.proba_distribution(mean_actions, self.log_std)
        else:
            raise ValueError("Invalid action distribution")

    def forward(
        self,
        obs: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """

        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        values = self.value_net(latent_vf)

        # Get the action distribution
        distribution = self._get_action_dist_from_latent(latent_pi)
        if deterministic:
            actions = distribution.mean
        else:
            actions = distribution.sample()
        log_prob = distribution.log_prob(actions).sum(dim=-1)
        actions = actions.reshape((-1, *self.action_space.shape))
        return actions, values, log_prob
