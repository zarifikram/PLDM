from abc import ABC, abstractmethod
from typing import NamedTuple
from pldm.planning import objectives_v2 as objectives
import torch


class PlanningResult(NamedTuple):
    pred_encs: torch.Tensor  # the entire predicted encoding
    pred_obs: torch.Tensor  # the observation component of the predicted encoding
    actions: torch.Tensor
    locations: torch.Tensor
    # locations that the model has planned to achieve
    losses: torch.Tensor


class Planner(ABC):
    def __init__(self):
        self.objective = None

    def set_objective(self, objective: objectives.BaseMPCObjective):
        self.objective = objective

    @abstractmethod
    def plan(self, obs: torch.Tensor, steps_left: int):
        pass

    def reset_targets(self, targets: torch.Tensor, repr_input: bool = False):
        self.objective.set_target(targets, repr_input=repr_input)
