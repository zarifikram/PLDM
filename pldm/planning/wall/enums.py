from dataclasses import dataclass
from typing import Optional, NamedTuple
import torch
from pldm.planning.enums import MPCConfig


@dataclass
class WallMPCConfig(MPCConfig):
    discount: float = 1
    level: str = "medium"
    seed: Optional[int] = 42
    sample_y_min: int = 20
    sample_y_max: int = 28
    padding: int = 3
    error_threshold: float = 1


class MPCReport(NamedTuple):
    error_mean: torch.Tensor
    errors: torch.Tensor
    terminations: list
    planning_time: int
    cross_wall_rate: float
    init_plan_cross_wall_rate: float

    def build_log_dict(self, prefix: str = ""):
        return {
            f"{prefix}planning_error_mean": self.error_mean.item(),
            f"{prefix}planning_error_mean_rmse": self.error_mean.pow(0.5).item(),
            f"{prefix}success_rate": (self.errors < 1).float().mean().item(),
            f"{prefix}cross_wall_rate": self.cross_wall_rate,
            f"{prefix}init_plan_cross_wall_rate": self.init_plan_cross_wall_rate,
            f"{prefix}avg_termination_step": sum(self.terminations)
            / len(self.terminations),
        }
