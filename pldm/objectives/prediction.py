from dataclasses import dataclass
from typing import NamedTuple, List

import torch

from pldm.configs import ConfigBase
from pldm.models.jepa import ForwardResult


class PredictionLossInfo(NamedTuple):
    total_loss: torch.Tensor
    pred_loss: torch.Tensor
    loss_name: str = "prediction"
    name_prefix: str = ""

    def build_log_dict(self):
        return {
            f"{self.name_prefix}/{self.loss_name}_total_loss": self.total_loss.item(),
        }


@dataclass
class PredictionObjectiveConfig(ConfigBase):
    global_coeff: float = 1.0


class PredictionObjective(torch.nn.Module):
    def __init__(
        self,
        config: PredictionObjectiveConfig,
        repr_dim: int,
        pred_attr: str = "state",
        name_prefix: str = "",
    ):
        super().__init__()
        self.repr_dim = repr_dim
        self.config = config
        self.pred_attr = pred_attr
        self.name_prefix = name_prefix

    def __call__(self, _batch, result: List[ForwardResult]) -> PredictionLossInfo:
        result = result[-1]  # Prediction objective only uses the highest level result

        if self.pred_attr == "state":
            encodings = result.backbone_output.encodings[1:]
            predictions = result.pred_output.predictions[1:]
        elif self.pred_attr == "obs":
            encodings = result.backbone_output.obs_component[1:]
            predictions = result.pred_output.obs_component[1:]
        elif self.pred_attr == "propio":
            encodings = result.backbone_output.propio_component[1:]
            predictions = result.pred_output.propio_component[1:]
        else:
            raise NotImplementedError

        if result.ema_backbone_output is not None:
            if self.pred_attr == "state":
                encodings = result.ema_backbone_output.encodings[1:]
            elif self.pred_attr == "obs":
                encodings = result.ema_backbone_output.obs_component[1:]
            elif self.pred_attr == "propio":
                encodings = result.ema_backbone_output.propio_component[1:]
            else:
                raise NotImplementedError

        pred_loss = (encodings - predictions).pow(2).mean()

        return PredictionLossInfo(
            total_loss=pred_loss * self.config.global_coeff,
            pred_loss=pred_loss,
            loss_name=f"prediction_{self.pred_attr}",
        )
