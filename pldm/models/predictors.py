from typing import Optional, Union

import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions.normal import Normal
import numpy as np

from pldm.models.misc import build_mlp
from pldm import models
from .utils import *
from pldm.models.enums import PredictorConfig, PredictorOutput


class SequencePredictor(torch.nn.Module):
    def __init__(
        self,
        config,
        repr_dim,
        z_dim: Optional[int] = None,
        z_min_std: Optional[float] = None,
        z_discrete: Optional[bool] = None,
        z_discrete_dists: Optional[int] = None,
        z_discrete_dim: Optional[int] = None,
        posterior_drop_p: Optional[float] = None,
        predictor_ln: Optional[bool] = False,
        prior_arch: Optional[str] = None,
        posterior_arch: Optional[str] = None,
        posterior_input_type: Optional[str] = None,
        posterior_input_dim: Optional[str] = None,
        action_dim: Optional[int] = None,
        pred_propio_dim: Optional[Union[int, tuple]] = 0,
        pred_obs_dim: Optional[Union[int, tuple]] = 0,
        backbone_ln: Optional[torch.nn.Module] = None,
    ):
        super().__init__()
        self.config = config
        self.repr_dim = repr_dim  # may need to flatten for prior and posterior...
        self.posterior_drop_p = posterior_drop_p
        self.posterior_input_type = posterior_input_type
        self.z_discrete = z_discrete
        self.action_dim = action_dim
        self.pred_propio_dim = pred_propio_dim
        self.pred_obs_dim = pred_obs_dim

        if config.tie_backbone_ln:
            self.final_ln = backbone_ln
        elif config.predictor_ln:
            self.final_ln = nn.LayerNorm(repr_dim)
        else:
            self.final_ln = nn.Identity()

        if z_dim is not None and z_dim > 0:
            if z_discrete:
                self.prior_model = models.DiscreteNet(
                    input_dim=repr_dim,
                    arch=prior_arch,
                    z_discrete_dim=z_discrete_dim,
                    z_discrete_dists=z_discrete_dists,
                    min_std=z_min_std,
                )
                self.posterior_model = models.DiscreteNet(
                    input_dim=posterior_input_dim,
                    arch=posterior_arch,
                    z_discrete_dim=z_discrete_dim,
                    z_discrete_dists=z_discrete_dists,
                    min_std=z_min_std,
                )
                self.latent_merger = nn.Linear(z_discrete_dim * z_discrete_dists, z_dim)

            else:
                self.prior_model = models.PriorContinuous(
                    input_dim=repr_dim,
                    arch=prior_arch,
                    z_dim=z_dim,
                    min_std=z_min_std,
                )
                self.posterior_model = models.PosteriroContinuous(
                    input_dim=posterior_input_dim,
                    arch=posterior_arch,
                    z_dim=z_dim,
                    min_std=z_min_std,
                )
        else:
            self.prior_model = None
            self.posterior_model = None

    def _is_rnn(self):
        return self.__class__ == RNNPredictorV2

    def forward_multiple(
        self,
        state_encs,
        actions,
        T,
        latents=None,
        flatten_output=False,
        compute_posterior=False,
    ):
        """
        This does multiple steps
        Parameters:
            state_encs: (t, BS, input_dim)
            actions: (t-1, BS, action_dim)
            T: timesteps to propagate forward
        Output:
            state_predictions: (T, BS, hidden_dim)
            rnn_states: (T, BS, hidden_dim)
        """
        bs = state_encs.shape[1]
        current_state = state_encs[0]
        state_predictions = [current_state]
        prior_mus = []
        prior_vars = []
        prior_logits = []
        priors = []
        posterior_mus = []
        posterior_vars = []
        posterior_logits = []
        posteriors = []

        for i in range(T):
            predictor_input = []
            if self.prior_model is not None:
                prior_stats = self.prior_model(flatten_conv_output(current_state))
                # z is of shape BxD

                if latents is not None:
                    prior = latents[i]
                else:
                    prior = self.prior_model.sample(prior_stats)

                if self.z_discrete:
                    prior = self.latent_merger(prior)
                    prior_logits.append(prior_stats)
                else:
                    mu, var = prior_stats
                    prior_mus.append(mu)
                    prior_vars.append(var)

                priors.append(prior)

                if compute_posterior:
                    # compute posterior
                    if self.posterior_input_type == "term_states":
                        posterior_input = torch.cat(
                            [
                                flatten_conv_output(current_state),
                                flatten_conv_output(state_encs[i + 1]),
                            ],
                            dim=-1,
                        )
                    elif self.posterior_input_type == "actions":
                        posterior_input = actions[i]

                    posterior_stats = self.posterior_model(posterior_input)
                    posterior = self.posterior_model.sample(posterior_stats)

                    if self.z_discrete:
                        posterior = self.latent_merger(posterior)
                        posterior_logits.append(posterior_stats)
                    else:
                        posterior_mu, posterior_var = posterior_stats
                        posterior_mus.append(posterior_mu)
                        posterior_vars.append(posterior_var)

                    posteriors.append(posterior)

                    z_input = posterior

                    if (
                        self.posterior_drop_p
                        and np.random.random() < self.posterior_drop_p
                    ):
                        z_input = prior
                        # TODO check this. seems like a bug. not supposed to replace all posterior with prior
                    predictor_input.append(z_input)
                else:
                    predictor_input.append(prior)
            else:
                prior = None
                predictor_input.append(actions[i])

            assert len(predictor_input) > 0

            if self._is_rnn():
                if i == 0:
                    current_state = (
                        current_state.unsqueeze(0)
                        .repeat(self.num_layers, 1, 1)
                        .contiguous()
                    )

                next_state, next_hidden_state = self.forward(
                    rnn_state=current_state,
                    rnn_input=torch.cat(predictor_input, dim=-1),
                )
                current_state = next_hidden_state

            else:
                next_state = self.forward(
                    current_state, torch.cat(predictor_input, dim=-1)
                )
                current_state = next_state

            state_predictions.append(next_state)

        t = len(state_predictions)
        state_predictions = torch.stack(state_predictions)
        if flatten_output:
            state_predictions = state_predictions.view(t, bs, -1)

        prior_mus = torch.stack(prior_mus) if prior_mus else None
        prior_vars = torch.stack(prior_vars) if prior_vars else None
        prior_logits = torch.stack(prior_logits) if prior_logits else None
        priors = torch.stack(priors) if priors else None
        posterior_mus = torch.stack(posterior_mus) if posterior_mus else None
        posterior_vars = torch.stack(posterior_vars) if posterior_vars else None
        posterior_logits = torch.stack(posterior_logits) if posterior_logits else None
        posteriors = torch.stack(posteriors) if posteriors else None

        if self.pred_propio_dim:
            if isinstance(self.pred_propio_dim, int):
                obs_component = state_predictions[:, :, : -self.pred_propio_dim]
                propio_component = state_predictions[:, :, -self.pred_propio_dim :]
            else:
                pred_propio_channels = self.pred_propio_dim[0]
                obs_component = state_predictions[:, :, :-pred_propio_channels]
                propio_component = state_predictions[:, :, -pred_propio_channels:]
        else:
            obs_component = state_predictions
            propio_component = None

        output = PredictorOutput(
            predictions=state_predictions,
            obs_component=obs_component,
            propio_component=propio_component,
            prior_mus=prior_mus,
            prior_vars=prior_vars,
            prior_logits=prior_logits,
            priors=priors,
            posterior_mus=posterior_mus,
            posterior_vars=posterior_vars,
            posterior_logits=posterior_logits,
            posteriors=posteriors,
        )

        return output


class MLPPredictor(SequencePredictor):
    def __init__(
        self,
        # parent inputs
        config,
        repr_dim,
        action_dim=2,  # action + z_dim
        pred_propio_dim=0,
        pred_obs_dim=0,
        backbone_ln: Optional[torch.nn.Module] = None,
    ):
        super().__init__(
            config=config,
            repr_dim=repr_dim,
            action_dim=action_dim,
            pred_propio_dim=pred_propio_dim,
            pred_obs_dim=pred_obs_dim,
            backbone_ln=backbone_ln,
        )

        self.fc = build_mlp(
            layers_dims=config.predictor_subclass,
            input_dim=repr_dim + action_dim,
            output_shape=repr_dim,
            norm="layer_norm" if config.predictor_ln else None,
            activation="mish",
        )

    def forward(self, current_state, curr_action):
        inp = torch.cat([current_state, curr_action], dim=-1)
        out = self.fc(inp)
        out = self.final_ln(out)
        return out


class RNNPredictor(torch.nn.Module):
    def __init__(
        self,
        hidden_size: int = 512,
        num_layers: int = 1,
        action_dim: int = 2,
        z_dim: Optional[int] = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.z_dim = z_dim

        input_size = action_dim
        if z_dim is not None:
            input_size += z_dim

        self.rnn = torch.nn.GRU(
            input_size=input_size,  # action + optionally z_dim
            hidden_size=self.hidden_size,
            num_layers=num_layers,
        )

    def burn_in(self, *args, **kwargs):
        return None

    def predict_sequence(
        self,
        enc: torch.Tensor,
        h: torch.Tensor,
        actions: torch.Tensor,
        zs: Optional[torch.Tensor] = None,
    ):
        # in this version, encoding is directly used as h, and the passed h is ignored.
        # since h is obtained from burn_in, it's actually None.
        h = enc
        if zs is None:
            inputs = actions
        else:
            inputs = torch.cat([actions, zs], dim=-1)
        return self.rnn(inputs, h.unsqueeze(0).repeat(self.num_layers, 1, 1))[0]


class RNNPredictorV2(SequencePredictor):
    def __init__(
        self,
        # parent inputs
        config,
        hidden_size: int = 512,
        z_dim: Optional[int] = None,
        z_min_std: Optional[float] = None,
        posterior_drop_p: Optional[float] = None,
        prior_arch: Optional[str] = None,
        posterior_arch: Optional[str] = None,
        posterior_input_type: Optional[str] = None,
        posterior_input_dim: Optional[str] = None,
        action_dim: Optional[int] = None,
        pred_propio_dim=0,
        pred_obs_dim=0,
        # child inputs
        predictor_ln: Optional[bool] = False,
        num_layers: int = 1,
        input_size: int = 2,
        backbone_ln: Optional[torch.nn.Module] = None,
    ):
        super().__init__(
            config=config,
            repr_dim=hidden_size,
            z_dim=z_dim,
            z_min_std=z_min_std,
            posterior_drop_p=posterior_drop_p,
            prior_arch=prior_arch,
            posterior_arch=posterior_arch,
            posterior_input_type=posterior_input_type,
            posterior_input_dim=posterior_input_dim,
            action_dim=action_dim,
            pred_propio_dim=pred_propio_dim,
            pred_obs_dim=pred_obs_dim,
            backbone_ln=backbone_ln,
        )

        self.num_layers = num_layers
        self.input_size = input_size

        self.rnn = torch.nn.GRU(
            input_size=self.input_size,
            hidden_size=self.repr_dim,
            num_layers=num_layers,
        )

    def forward(self, rnn_state, rnn_input):
        """
        Propagate one step forward
        Parameters:
            rnn_state: (num_layers, bs, dim)
            rnn_input: (bs, a_dim)
        Output:
            output: next_state (bs, dim), next_hidden_state (num_layers, bs, dim)
        """
        # This only does one step

        next_state, next_hidden_state = self.rnn(rnn_input.unsqueeze(0), rnn_state)

        next_state = self.final_ln(next_state)
        next_hidden_state = self.final_ln(next_hidden_state)

        return next_state[0], next_hidden_state


ConvPredictorConfig = {
    "a": [(18, 32, 3, 1, 1), (32, 32, 3, 1, 1), (32, 16, 3, 1, 1)],
    "b": [(18, 32, 5, 1, 2), (32, 32, 5, 1, 2), (32, 16, 5, 1, 2)],
    "c": [(18, 32, 7, 1, 3), (32, 32, 7, 1, 3), (32, 16, 7, 1, 3)],
    "a_propio": [(20, 32, 3, 1, 1), (32, 32, 3, 1, 1), (32, 18, 3, 1, 1)],
    "d4rl_b_p": [(20, 32, 3, 1, 1), (32, 32, 3, 1, 1), (32, 18, 3, 1, 1)],
    "d4rl_c_p": [(36, 32, 3, 1, 1), (32, 32, 3, 1, 1), (32, 34, 3, 1, 1)],
}


class ConvPredictor(SequencePredictor):
    def __init__(
        self,
        # parent inputs
        config,
        repr_dim,
        z_dim: Optional[int] = None,
        z_min_std: Optional[float] = None,
        z_discrete: Optional[bool] = None,
        z_discrete_dists: Optional[int] = None,
        z_discrete_dim: Optional[int] = None,
        posterior_drop_p: Optional[float] = None,
        prior_arch: Optional[str] = None,
        posterior_arch: Optional[str] = None,
        posterior_input_type: Optional[str] = None,
        posterior_input_dim: Optional[str] = None,
        action_dim=2,  # action + z_dim
        pred_propio_dim=0,
        pred_obs_dim=0,
        # child inputs
        predictor_subclass="a",
        num_groups=4,
    ):
        super(ConvPredictor, self).__init__(
            config=config,
            repr_dim=repr_dim,
            z_dim=z_dim,
            z_min_std=z_min_std,
            z_discrete=z_discrete,
            z_discrete_dists=z_discrete_dists,
            z_discrete_dim=z_discrete_dim,
            posterior_drop_p=posterior_drop_p,
            prior_arch=prior_arch,
            posterior_arch=posterior_arch,
            posterior_input_type=posterior_input_type,
            posterior_input_dim=posterior_input_dim,
            action_dim=action_dim,
            pred_propio_dim=pred_propio_dim,
            pred_obs_dim=pred_obs_dim,
        )
        self.num_groups = num_groups

        # Define convolutional layers
        layers = []
        layers_config = ConvPredictorConfig[predictor_subclass]
        for i in range(len(layers_config) - 1):
            in_channels, out_channels, kernel_size, stride, padding = layers_config[i]

            if i == 0:
                in_channels = repr_dim[0] + action_dim

            layers.append(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
            )
            layers.append(nn.GroupNorm(4, out_channels))
            layers.append(nn.ReLU())

        # last layer
        in_channels, out_channels, kernel_size, stride, padding = layers_config[-1]
        layers.append(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        )

        self.layers = nn.Sequential(*layers)

        # Action encoder
        if self.config.action_encoder_arch and self.config.action_encoder_arch != "id":
            layer_dims = [int(x) for x in self.config.action_encoder_arch.split("-")]
            layers = []
            for i in range(len(layer_dims) - 1):
                layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
                layers.append(nn.ReLU())
            # remove last ReLU
            layers.pop()

            self.action_encoder = nn.Sequential(
                *layers,
                Expander2D(w=repr_dim[-2], h=repr_dim[-1]),
            )
            # self.action_dim = layer_dims[-1]
        else:
            self.action_encoder = Expander2D(w=repr_dim[-2], h=repr_dim[-1])

    def forward(self, current_state, curr_action):
        bs, _, h, w = current_state.shape
        curr_action = self.action_encoder(curr_action)
        x = torch.cat([current_state, curr_action], dim=1)
        x = self.layers(x)
        if self.config.residual:
            x = x + current_state

        return x


class RNNPredictorV3(torch.nn.Module):
    def __init__(
        self,
        state_size: int = 10,
        hidden_size: int = 512,
        input_size: int = 2,
        arch: str = "",
    ):
        # state size is mapped with mlp to hidden size.
        super().__init__()
        self.state_size = state_size
        self.hidden_size = hidden_size
        self.input_size = input_size
        if arch != "":
            layer_dims = (
                [state_size] + list(map(int, arch.split("-"))) + [self.hidden_size]
            )
        else:
            layer_dims = [state_size, self.hidden_size]
        self.input_mlp = build_mlp(layer_dims)
        self.output_mlp = build_mlp(layer_dims[::-1])

        self.rnn = torch.nn.GRU(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=1,
        )

    def convert_state(self, state):
        return self.input_mlp(state)

    def forward(self, rnn_input, rnn_state):
        # This only does one step
        res = self.rnn(rnn_input.unsqueeze(0), rnn_state.unsqueeze(0))
        output = self.output_mlp(res[1][0])
        return res[1][0], output


class RNNPredictorBurnin(torch.nn.Module):
    def __init__(
        self,
        hidden_size: int = 512,
        output_size: int = 512,
        num_layers: int = 1,
        action_dim: int = 2,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.action_dim = action_dim

        self.rnn = torch.nn.GRU(
            input_size=action_dim + output_size,
            hidden_size=self.hidden_size,
            num_layers=num_layers,
        )
        self.output_projector = nn.Linear(hidden_size, output_size)

    def burn_in(
        self,
        encs: torch.Tensor,
        actions: torch.Tensor,
        h: Optional[torch.Tensor] = None,
    ):
        """Runs a few iterations of RNN with the provided GT encodings to obtain h0"""
        if h is None:
            h = torch.zeros(self.num_layers, actions.shape[1], self.hidden_size).to(
                actions.device
            )

        for i in range(encs.shape[0]):
            rnn_input = torch.cat([encs[i], actions[i]], dim=1).unsqueeze(0)
            _, h = self.rnn(rnn_input, h)
        return h

    def predict_sequence(
        self, enc: torch.Tensor, actions: torch.Tensor, h: Optional[torch.Tensor] = None
    ):
        """Predicts the sequence given gt encoding for the current time step"""
        outputs = []
        if h is None:
            h = torch.zeros(self.num_layers, actions.shape[1], self.hidden_size).to(
                actions.device
            )
        for i in range(actions.shape[0]):
            rnn_input = torch.cat([enc, actions[i]], dim=1).unsqueeze(0)
            _, h = self.rnn(rnn_input, h)
            outputs.append(self.output_projector(h[-1]))
            enc = outputs[-1]  # autoregressive GRU
        outputs = torch.stack(outputs)
        return outputs


class RSSMPredictor(torch.nn.Module):
    def __init__(
        self,
        rnn_state_dim: int,
        z_dim: int,
        action_dim: int = 2,
        min_var: float = 0,
        use_action_only: bool = True,
    ):
        super().__init__()
        self.rnn_state_dim = rnn_state_dim
        self.z_dim = z_dim
        self.input_dim = z_dim + action_dim
        self.action_dim = action_dim
        self.prior_mu_net = nn.Linear(self.rnn_state_dim, self.z_dim)
        self.prior_var_net = nn.Linear(self.rnn_state_dim, self.z_dim)
        self.rnn = torch.nn.GRUCell(self.input_dim, self.rnn_state_dim)
        self.min_var = min_var
        self.use_action_only = use_action_only

    def forward(self, sampled_prior, action, rnn_state):
        if action is not None and self.use_action_only:
            rnn_input = action  # torch.cat([sampled_prior, action], dim=-1)
        elif action is not None and not self.use_action_only:
            rnn_input = torch.cat([sampled_prior, action], dim=-1)
        else:
            rnn_input = sampled_prior

        rnn_state_new = self.rnn(rnn_input, rnn_state)
        prior_mu = self.prior_mu_net(rnn_state_new)
        prior_var = self.prior_var_net(rnn_state_new)
        return rnn_state_new, prior_mu, prior_var

    def burn_in(self, *args, **kwargs):
        return None

    def predict_sequence(
        self,
        enc: torch.Tensor,
        actions: torch.Tensor,
        h: torch.Tensor,
        latents: Optional[torch.Tensor] = None,
    ):
        initial_belief = enc
        batch_size = enc.shape[0]
        sampled_prior_state = torch.zeros(batch_size, self.z_dim).to(enc.device)
        sampled_prior_states = []
        beliefs = []
        rnn_belief = initial_belief
        for i in range(len(actions)):
            rnn_belief, prior_mu, prior_var = self(
                sampled_prior_state, actions[i], rnn_belief
            )
            prior_var = F.softplus(prior_var) + self.min_var
            z = Normal(prior_mu, (prior_var))
            if latents is not None:
                sampled_prior_state = latents[i]
            else:
                sampled_prior_state = z.sample()
            sampled_prior_states.append(sampled_prior_state)
            beliefs.append(rnn_belief)
        beliefs = torch.stack(beliefs, dim=0)
        return beliefs

    def predict_sequence_posterior(
        self,
        encs: torch.Tensor,
        h: torch.Tensor,
        hjepa: torch.nn.Module,
    ):
        result = []
        T = encs.shape[0] + 1
        batch_size = encs.shape[1]
        rnn_state = encs[0]
        sampled_posterior_state = torch.zeros(batch_size, self.z_dim).to(encs.device)
        for i in range(T - 1):
            rnn_state = hjepa.predictor_l2(
                sampled_prior=sampled_posterior_state, action=None, rnn_state=rnn_state
            )
            posterior_mu, posterior_var = hjepa.posterior_l2(encs[i + 1], rnn_state)
            posterior_var = F.softplus(posterior_var) + self.min_var
            sampled_posterior_state = Normal(posterior_mu, (posterior_var)).sample()
            prediction = hjepa.decoder(rnn_state, sampled_posterior_state)
            result.append(prediction)
        return result

    def predict_decode_sequence(
        self,
        enc: torch.Tensor,
        h: torch.Tensor,
        latents: torch.Tensor,
        decoder: torch.nn.Module,
    ):
        beliefs = self.predict_sequence(enc, [None] * latents.shape[0], h, latents)
        return decoder(beliefs, latents)


def build_predictor(
    config: PredictorConfig,
    repr_dim: int,
    action_dim: int,
    pred_propio_dim: Union[int, tuple],
    pred_obs_dim: Union[int, tuple],
    backbone_ln: Optional[torch.nn.Module] = None,
):
    arch = config.predictor_arch
    predictor_subclass = config.predictor_subclass
    rnn_layers = config.rnn_layers
    prior_arch = config.prior_arch
    posterior_arch = config.posterior_arch
    z_dim = config.z_dim
    z_min_std = config.z_min_std
    z_discrete = config.z_discrete
    z_discrete_dists = config.z_discrete_dists
    z_discrete_dim = config.z_discrete_dim
    posterior_drop_p = config.posterior_drop_p
    predictor_ln = config.predictor_ln
    posterior_input_type = config.posterior_input_type
    posterior_input_dim = config.posterior_input_dim

    if arch == "mlp":
        predictor = MLPPredictor(
            config=config,
            repr_dim=repr_dim,
            action_dim=action_dim,
            pred_propio_dim=pred_propio_dim,
            pred_obs_dim=pred_obs_dim,
            backbone_ln=backbone_ln,
        )
    elif arch == "conv":
        predictor = PixelPredictorConv(action_dim=action_dim)
    elif arch == "conv2":
        predictor = ConvPredictor(
            config=config,
            repr_dim=repr_dim,
            predictor_subclass=predictor_subclass,
            z_discrete=z_discrete,
            z_discrete_dists=z_discrete_dists,
            z_discrete_dim=z_discrete_dim,
            z_dim=z_dim,
            z_min_std=z_min_std,
            posterior_drop_p=posterior_drop_p,
            prior_arch=prior_arch,
            posterior_arch=posterior_arch,
            posterior_input_type=posterior_input_type,
            posterior_input_dim=posterior_input_dim,
            action_dim=action_dim,
            pred_propio_dim=pred_propio_dim,
            pred_obs_dim=pred_obs_dim,
        )
    elif arch == "rnn":
        predictor = RNNPredictor(
            hidden_size=repr_dim,
            num_layers=rnn_layers,
            action_dim=action_dim,
            z_dim=z_dim,
        )
    elif arch == "rnnV2":
        predictor = RNNPredictorV2(
            config=config,
            hidden_size=repr_dim,
            num_layers=rnn_layers,
            input_size=action_dim,
            z_dim=z_dim,
            z_min_std=z_min_std,
            posterior_drop_p=posterior_drop_p,
            predictor_ln=predictor_ln,
            prior_arch=prior_arch,
            posterior_arch=posterior_arch,
            posterior_input_type=posterior_input_type,
            posterior_input_dim=posterior_input_dim,
            pred_propio_dim=pred_propio_dim,
            pred_obs_dim=pred_obs_dim,
            backbone_ln=backbone_ln,
            action_dim=action_dim,
        )
    elif arch == "rnnV3":
        predictor = RNNPredictorV3(
            hidden_size=repr_dim,
            num_layers=rnn_layers,
            input_size=action_dim,
        )
    elif arch == "rnn_burnin":
        predictor = RNNPredictorBurnin(
            hidden_size=repr_dim,
            output_size=repr_dim,
            num_layers=rnn_layers,
            action_dim=action_dim,
            z_dim=z_dim,
        )
    elif arch == "id":
        predictor = IDPredictor()
    else:
        predictor = Predictor(arch, repr_dim, action_dim=action_dim, z_dim=z_dim)

    return predictor
