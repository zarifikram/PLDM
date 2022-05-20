from pldm.configs import ConfigBase
from dataclasses import dataclass
from typing import Optional
import torch


@dataclass
class BackboneConfig(ConfigBase):
    arch: str = "menet5"
    backbone_subclass: str = "a"
    backbone_width_factor: int = 1
    backbone_mlp: Optional[str] = None  # mlp to slap on top of backbone
    backbone_norm: str = "batch_norm"
    backbone_pool: str = "avg_pool"
    backbone_final_fc: bool = True
    channels: int = 1
    input_dim: Optional[int] = None  # if it's none, we assume it's image.
    propio_dim: Optional[int] = None
    propio_encoder_arch: Optional[str] = None
    fc_output_dim: Optional[int] = None  # if it's none, it will be a spatial output
    final_ln: bool = False


class BackboneOutput:
    def __init__(
        self,
        encodings: torch.Tensor,
        obs_component: Optional[torch.Tensor] = None,
        propio_component: Optional[torch.Tensor] = None,
    ):
        self.encodings = encodings
        self._obs_component = obs_component
        self.propio_component = propio_component

    @property
    def obs_component(self):
        return (
            self._obs_component if self._obs_component is not None else self.encodings
        )

    @obs_component.setter
    def obs_component(self, value: Optional[torch.Tensor]):
        self._obs_component = value
