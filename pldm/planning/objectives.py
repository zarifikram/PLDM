import torch
from pldm_envs.utils.normalizer import Normalizer


class BaseMPCObjective:
    """Base class for MPC objective.
    This is a callable that takes encodings and returns a tensor -
    objective to be optimized.
    """

    def __call__(self, encodings: torch.Tensor) -> torch.Tensor:
        pass


class SingleStepReprTargetMPCObjective(BaseMPCObjective):
    """Objective to measure the cost to target representation for one time step"""

    def __init__(self, target_enc: torch.Tensor):
        """_summary_
        Args:
            target_enc (D):
        """
        self.target_enc = target_enc

    def __call__(self, state, action):
        """encoding shape is B x D"""
        diff = (state - self.target_enc).pow(2)
        return diff.mean(dim=1)
