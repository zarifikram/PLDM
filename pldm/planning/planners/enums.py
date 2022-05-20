import enum
from dataclasses import dataclass
from pldm.configs import ConfigBase


class PlannerType(enum.Enum):
    SGD = enum.auto()
    MPPI = enum.auto()
    BeamSearch = enum.auto()


@dataclass
class SGDConfig(ConfigBase):
    n_iters: int = 100
    lr: float = 1e-2
    l2_reg: float = 0.0
    action_change_reg: float = 0.0
    z_reg_coeff: float = 0.0


@dataclass
class MPPIConfig(ConfigBase):
    noise_sigma: float = 5
    num_samples: int = 500
    lambda_: float = 0.005
    z_reg_coeff: float = 0


@dataclass
class LFBGSConfig(ConfigBase):
    history_size: int = 10
    max_iter: int = 20


@dataclass
class PlannerConfig(ConfigBase):
    planner_type: PlannerType = PlannerType.SGD
    sgd: SGDConfig = SGDConfig()
    mppi: MPPIConfig = MPPIConfig()
    lfgbs: LFBGSConfig = LFBGSConfig()
    clamp_actions: bool = False
    min_step: float = 0.1
    max_step: float = 1.0
    repr_target: bool = False
    loss_coeff_first: float = 0.1
    loss_coeff_last: float = 1
    sum_all_diffs: bool = False
    min_plan_length: int = 1
    max_plan_length: int = 26
    depth_probe_threshold: int = 25
    probe_depth: bool = False
    sum_last_n: int = 3
    propio_cost: bool = False
    projected_cost: bool = False
