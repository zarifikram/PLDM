from pldm.configs import ConfigBase
from dataclasses import dataclass
from typing import Optional, NamedTuple
import torch


@dataclass
class PredictorConfig(ConfigBase):
    predictor_arch: str = "rnnV3"
    predictor_subclass: str = "a"
    predictor_ln: bool = False
    rnn_state_dim: int = 512
    rnn_converter_arch: str = ""
    z_discrete: bool = False
    z_discrete_dim: int = 16
    z_discrete_dists: int = 16
    z_dim: int = 0
    z_min_std: float = 0.1
    posterior_drop_p: float = 0.0
    prior_arch: str = "512"
    posterior_arch: str = "512"
    posterior_input_type: str = "term_states"
    posterior_input_dim: Optional[int] = None
    action_encoder_arch: str = ""
    residual: bool = False
    rnn_layers: int = 1
    tie_backbone_ln: bool = False


class PredictorOutput:
    def __init__(
        self,
        predictions: torch.Tensor,
        obs_component: Optional[torch.Tensor] = None,
        propio_component: Optional[torch.Tensor] = None,
        prior_mus: Optional[torch.Tensor] = None,
        prior_vars: Optional[torch.Tensor] = None,
        prior_logits: Optional[torch.Tensor] = None,
        priors: Optional[torch.Tensor] = None,
        posterior_mus: Optional[torch.Tensor] = None,
        posterior_vars: Optional[torch.Tensor] = None,
        posterior_logits: Optional[torch.Tensor] = None,
        posteriors: Optional[torch.Tensor] = None,
    ):
        self.predictions = predictions
        self._obs_component = obs_component
        self.propio_component = propio_component
        self.prior_mus = prior_mus
        self.prior_vars = prior_vars
        self.prior_logits = prior_logits
        self.priors = priors
        self.posterior_mus = posterior_mus
        self.posterior_vars = posterior_vars
        self.posterior_logits = posterior_logits
        self.posteriors = posteriors

    @property
    def obs_component(self):
        return (
            self._obs_component if self._obs_component is not None else self.predictions
        )

    @obs_component.setter
    def obs_component(self, value: Optional[torch.Tensor]):
        self._obs_component = value


class PredictorOutput(NamedTuple):
    predictions: torch.Tensor
    obs_component: Optional[torch.Tensor] = None
    propio_component: Optional[torch.Tensor] = None
    prior_mus: Optional[torch.Tensor] = None
    prior_vars: Optional[torch.Tensor] = None
    prior_logits: Optional[torch.Tensor] = None
    priors: Optional[torch.Tensor] = None
    posterior_mus: Optional[torch.Tensor] = None
    posterior_vars: Optional[torch.Tensor] = None
    posterior_logits: Optional[torch.Tensor] = None
    posteriors: Optional[torch.Tensor] = None
