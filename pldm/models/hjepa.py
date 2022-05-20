from dataclasses import dataclass
from typing import NamedTuple, Optional

import torch
import random

from pldm.configs import ConfigBase
from pldm.models.jepa import JEPA, JEPAConfig, ForwardResult as JEPAForwardResult


@dataclass
class HJEPAConfig(ConfigBase):
    level1: JEPAConfig = JEPAConfig()
    step_skip: int = 4
    disable_l2: bool = False
    freeze_l1: bool = False
    train_l1: bool = False
    l1_n_steps: int = 17


class ForwardResult(NamedTuple):
    level1: JEPAForwardResult


class HJEPA(torch.nn.Module):
    def __init__(
        self,
        config: JEPAConfig,
        input_dim,
        normalizer=None,
        use_propio_pos=False,
        use_propio_vel=False,
    ):
        super().__init__()
        self.config = config
        self.level1 = JEPA(
            config.level1,
            input_dim=input_dim,
            use_propio_pos=use_propio_pos,
            use_propio_vel=use_propio_vel,
        )

        self.normalizer = normalizer

    def forward_prior(
        self,
        input_states: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        T: Optional[int] = None,
    ) -> ForwardResult:
        if not self.config.disable_l2:
            raise NotImplementedError(
                "forward_prior should be called for each level individually."
            )
        else:
            return ForwardResult(
                level1=self.level1.forward_prior(input_states, actions, T)
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
    ) -> ForwardResult:
        forward_result_l1 = None

        if self.config.train_l1:
            # sample a subsequence of length l1_n_steps
            sub_idx = random.randint(0, input_states.shape[0] - self.config.l1_n_steps)
            l1_input_states = input_states[sub_idx : sub_idx + self.config.l1_n_steps]
            l1_actions = actions[sub_idx : sub_idx + self.config.l1_n_steps - 1]

            forward_result_l1 = self.level1.forward_posterior(
                l1_input_states,
                l1_actions,
                propio_pos=propio_pos,
                propio_vel=propio_vel,
                chunked_locations=chunked_locations,
                chunked_propio_pos=chunked_propio_pos,
                chunked_propio_vel=chunked_propio_vel,
                encode_only=False,
            )

        return ForwardResult(level1=forward_result_l1)

    def update_ema(self):
        self.level1.update_ema()
