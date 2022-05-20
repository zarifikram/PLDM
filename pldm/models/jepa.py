from dataclasses import dataclass
from typing import Optional, NamedTuple

import torch

from pldm.configs import ConfigBase
from pldm.models.encoders.enums import BackboneConfig, BackboneOutput
from pldm.models.enums import PredictorOutput, PredictorConfig
from functools import reduce
import operator
from pldm.models.encoders.encoders import build_backbone
from pldm.models.predictors import build_predictor


@dataclass
class JEPAConfig(ConfigBase):
    backbone: BackboneConfig = BackboneConfig()
    predictor: PredictorConfig = PredictorConfig()

    action_dim: int = 2

    momentum: float = 0.0  # If 0, no ema

    # whether to use the trajectory goal as the latent variable.
    use_z_goal: bool = False
    encode_only: bool = False


class ForwardResult(NamedTuple):
    backbone_output: BackboneOutput
    ema_backbone_output: BackboneOutput
    pred_output: PredictorOutput
    actions: torch.Tensor


class JEPA(torch.nn.Module):
    """Joint-Embedding Predictive Architecture
    Includes an image encoder and a predictor.
    """

    def __init__(
        self,
        config: JEPAConfig,
        input_dim,
        l1_action_dim: Optional[int] = None,
        step_skip: Optional[int] = None,
        l2: bool = False,
        use_propio_pos: bool = False,
        use_propio_vel: bool = False,
    ):
        super().__init__()
        self.config = config
        self.use_propio_pos = use_propio_pos
        self.use_propio_vel = use_propio_vel
        self.backbone = build_backbone(
            config.backbone,
            input_dim=input_dim,
        )
        self.l2 = l2

        self.spatial_repr_dim = self.backbone.output_dim

        if isinstance(self.spatial_repr_dim, tuple):
            self.repr_dim = reduce(operator.mul, self.spatial_repr_dim)
        else:
            self.repr_dim = self.spatial_repr_dim

        if self.config.momentum > 0:
            self.backbone_ema, _ = build_backbone(
                config.backbone,
                input_dim=input_dim,
            )
            self.backbone_ema.load_state_dict(self.backbone.state_dict())
            for param in self.backbone_ema.parameters():
                param.requires_grad = False
        else:
            self.backbone_ema = None

        if l2:
            if config.predictor.posterior_input_type == "term_states":
                config.predictor.posterior_input_dim = self.repr_dim * 2
            elif config.predictor.posterior_input_type == "actions":
                config.predictor.posterior_input_dim = l1_action_dim * step_skip
            else:
                raise NotImplementedError

        # right now we don't support hybrid true action + latent action
        assert (self.config.action_dim or self.config.predictor.z_dim) and not (
            self.config.action_dim and self.config.predictor.z_dim
        )

        predictor_input_dim = self.config.action_dim + self.config.predictor.z_dim

        self.config.predictor.rnn_state_dim = self.repr_dim

        # predictor.tie_backbone_ln ==> backbone.final_ln
        assert not config.predictor.tie_backbone_ln or config.backbone.final_ln

        self.predictor = build_predictor(
            config.predictor,
            repr_dim=self.backbone.output_dim,
            action_dim=predictor_input_dim,
            pred_propio_dim=self.backbone.output_propio_dim,
            pred_obs_dim=self.backbone.output_obs_dim,
            backbone_ln=self.backbone.final_ln if config.backbone.final_ln else None,
        )

    def subsampling_ratio(self):
        return 1

    def forward_prior(
        self,
        input_states: torch.Tensor,
        repr_input: bool = False,
        actions: Optional[torch.Tensor] = None,
        propio_pos: Optional[torch.Tensor] = None,
        propio_vel: Optional[torch.Tensor] = None,
        latents: Optional[torch.Tensor] = None,
        goal: Optional[torch.Tensor] = None,
        T: Optional[int] = None,
    ):
        # If latents is not None, we use it instead of sampling from prior.

        # Input states are either images or encoded representations.
        # input_states is of shape BxD or BxCxHxW
        # actions is of shape TxBxA

        assert (
            actions is not None
            or T is not None
            or latents is not None
            or goal is not None
        )

        if repr_input:
            current_state = input_states
        else:
            current_state = self.backbone.forward_multiple(input_states).encodings

        if T is None:
            if actions is not None:
                T = actions.shape[0]
            elif latents is not None:
                T = latents.shape[0]
            else:
                raise ValueError("T is None but actions and latents are None")

        pred_output = self.predictor.forward_multiple(
            current_state.unsqueeze(0),
            actions,
            T,
            latents=latents,
        )

        return ForwardResult(
            backbone_output=None,
            ema_backbone_output=None,
            pred_output=pred_output,
            actions=actions,
        )

    def forward_posterior(
        self,
        input_states: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        propio_pos: Optional[torch.Tensor] = None,
        propio_vel: Optional[torch.Tensor] = None,
        chunked_locations: Optional[torch.Tensor] = None,
        chunked_propio_pos: Optional[torch.Tensor] = None,
        chunked_propio_vel: Optional[torch.Tensor] = None,
        goal: Optional[torch.Tensor] = None,
        encode_only: bool = False,
    ):
        """
        input_states:
            TxBxD or TxBxCxHxW
        actions:
            (T-1)xBxA
        """
        if input_states.shape[-1] != self.repr_dim:
            if self.config.backbone.propio_dim is not None:
                if propio_pos.numel() == 0:
                    propio_states = propio_vel
                elif propio_vel.numel() == 0:
                    propio_states = propio_pos
                else:
                    propio_states = torch.cat([propio_pos, propio_vel], dim=-1)

                backbone_output = self.backbone.forward_multiple(
                    input_states,
                    propio=propio_states,
                )
            else:
                backbone_output = self.backbone.forward_multiple(input_states)

            state_encs = backbone_output.encodings
        else:
            state_encs = input_states  # might be problematic for l2

        if self.config.momentum > 0:
            if self.config.backbone.propio_dim is not None:
                if propio_pos.numel() == 0:
                    propio_states = propio_vel
                elif propio_vel.numel() == 0:
                    propio_states = propio_pos
                else:
                    propio_states = torch.cat([propio_pos, propio_vel], dim=-1)

                ema_backbone_output = self.backbone_ema.forward_multiple(
                    input_states,
                    propio=propio_states,
                )
            else:
                ema_backbone_output = self.backbone_ema.forward_multiple(input_states)
        else:
            ema_backbone_output = None

        if self.config.encode_only or encode_only:
            return ForwardResult(
                backbone_output=backbone_output,
                ema_backbone_output=ema_backbone_output,
                pred_output=None,
                actions=actions,
            )

        T = input_states.shape[0] - 1

        pred_output = self.predictor.forward_multiple(
            state_encs, actions, T, compute_posterior=True
        )

        return ForwardResult(
            backbone_output=backbone_output,
            ema_backbone_output=ema_backbone_output,
            pred_output=pred_output,
            actions=actions,
        )

    def update_ema(self):
        if self.config.momentum > 0:
            for param, ema_param in zip(
                self.backbone.parameters(), self.backbone_ema.parameters()
            ):
                ema_param.data.mul_(self.config.momentum).add_(
                    param.data, alpha=1 - self.config.momentum
                )
