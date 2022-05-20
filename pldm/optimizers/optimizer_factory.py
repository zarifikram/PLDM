import torch
from pldm.optimizers.lars import LARS, exclude_bias_and_norm
import enum


class OptimizerType(enum.Enum):
    Adam = "sgd"
    LARS = "lars"


class OptimizerFactory:
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer_type: str,
        base_lr: float,
    ):
        self.model = model
        self.optimizer_type = optimizer_type
        self.base_lr = base_lr

    def create_optimizer(self):
        if self.optimizer_type == OptimizerType.LARS:
            optimizer = LARS(
                self.model.parameters(),
                lr=0,
                weight_decay=1e-6,
                weight_decay_filter=exclude_bias_and_norm,
                lars_adaptation_filter=exclude_bias_and_norm,
            )
        elif self.optimizer_type == OptimizerType.Adam:
            params_list = [
                {
                    "params": self.model.level1.parameters(),
                    "lr": self.base_lr,
                }
            ]

            optimizer = torch.optim.Adam(
                params_list,
                weight_decay=1e-6,
            )
        else:
            raise NotImplementedError("Unknown optimizer type")

        return optimizer
