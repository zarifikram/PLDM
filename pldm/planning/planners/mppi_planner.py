from typing import Optional, Callable

import torch
from torch import nn

from .mppi_torch import MPPI
from pldm.models.utils import flatten_conv_output
from .planner import PlanningResult


class LearnedDynamics:
    def __init__(self, model, state_dim=None):
        self.model = model
        self.dump_dict = None
        self.state_dim = state_dim
        self.max_batch_size = 500

    def __call__(self, state, action, only_return_last=True, flatten_output=True):
        """
        state: [K x nx]
        action: [K x nx]
        """
        # make sure state is in correct format
        og_shape = state.shape
        n_samples = og_shape[0]

        if isinstance(self.state_dim, int):
            self.state_dim = (self.state_dim,)

        new_shape = (n_samples, *self.state_dim)
        state = state.view(new_shape)

        # introduce time dimension to action if needed
        if len(action.shape) < 3:
            action = action.unsqueeze(0)

        T = action.shape[0]

        if self.model.config.action_dim:
            pred_output = self.model.predictor.forward_multiple(
                state.unsqueeze(0),
                action.float(),
                T,
            )
        else:
            pred_output = self.model.predictor.forward_multiple(
                state.unsqueeze(0),
                actions=None,
                T=T,
                latents=action.float(),
            )

        preds = pred_output.predictions
        pred_obs = pred_output.obs_component

        if flatten_output:
            preds = flatten_conv_output(preds)  # required for 3rd party MPPI code...
            pred_obs = flatten_conv_output(pred_obs)

        if only_return_last:
            preds = preds[-1]
            pred_obs = pred_obs[-1]

        # we need to return both. preds is used to propagate the state forward. pred_obs is used to take cost
        return preds, pred_obs

    def before_planning_callback(self):
        self.orig_training_state = self.model.training
        self.model.train(False)

    def after_planning_callback(self):
        self.model.train(self.orig_training_state)


class RunningCost:
    def __init__(self, objective, idx=None, projector=None):
        self.objective = objective
        self.idx = idx
        self.projector = nn.Identity() if projector is None else projector

    def __call__(self, state, action):
        """encoding shape is B X D
        Note that B are samples for the same environment
        You want to diff against target_enc of shape (D) retrieved from objective
        """
        objective = self.objective
        target = objective.target_enc[self.idx]

        state = self.projector(state)
        target = self.projector(target)

        diff = (state - target).pow(2)

        return diff.mean(dim=1)


class MPPIPlanner:
    def __init__(
        self,
        config,
        model,
        normalizer,
        objective,
        prober: Optional[torch.nn.Module] = None,
        action_normalizer: Optional[Callable] = None,
        num_refinement_steps: int = 1,
        n_envs: int = None,
        l2: bool = False,
        projected_cost: bool = False,
    ):
        device = next(model.parameters()).device

        latent_actions = l2 and model.config.predictor.z_dim > 0
        self.model = model
        self.config = config
        self.dynamics = LearnedDynamics(
            model,
            state_dim=model.spatial_repr_dim,
        )
        self.normalizer = normalizer
        self.action_normalizer = action_normalizer
        self.prober = prober
        self.latent_actions = latent_actions

        noise_sigma = torch.diag(
            torch.tensor(
                [config.noise_sigma] * model.predictor.action_dim,
                dtype=torch.float32,
                device=device,
            )
        )
        self.objective = objective

        self.mppi_costs = [
            RunningCost(
                objective,
                idx=i,
                projector=prober if projected_cost else None,
            )
            for i in range(n_envs)
        ]

        if isinstance(model.spatial_repr_dim, int):
            nx = torch.Size((model.spatial_repr_dim,))
        else:
            nx = torch.Size(model.spatial_repr_dim)

        self.ctrls = [
            MPPI(
                self.dynamics,
                running_cost=self.mppi_costs[i],
                nx=nx,
                noise_sigma=noise_sigma,
                num_samples=config.num_samples,
                lambda_=config.lambda_,
                device=device,
                action_normalizer=action_normalizer,
                u_per_command=-1,
                latent_actions=latent_actions,
                z_reg_coeff=config.z_reg_coeff,
            )
            for i in range(n_envs)
        ]
        self.last_plan_size = None
        self.num_refinement_steps = num_refinement_steps
        self.l2 = l2

    @torch.no_grad()
    def plan(
        self,
        current_state: torch.Tensor,
        plan_size: int,
        repr_input: bool = True,
        curr_propio_pos: Optional[torch.Tensor] = None,
        curr_propio_vel: Optional[torch.Tensor] = None,
        diff_loss_idx: Optional[torch.tensor] = None,
    ):
        """_summary_
        Args:
            current_state (bs, ch, w, h): representation of current obs
            plan_size (int): how many predictions to make into the future

        Returns:
            predictions (plan_size + 1, bs, n)
            actions (bs, plan_size, 2)
            locations (plan_size + 1, bs, 2) - probed locations from the predictions
            losses - set to None for now
        """
        batch_size = current_state.shape[0]
        self.dynamics.before_planning_callback()

        if not repr_input:
            if self.model.backbone.config.propio_dim:
                if curr_propio_vel is not None and curr_propio_pos is not None:
                    curr_propio_states = torch.cat(
                        [curr_propio_pos, curr_propio_vel], dim=-1
                    )
                elif curr_propio_vel is not None:
                    curr_propio_states = curr_propio_vel
                elif curr_propio_pos is not None:
                    curr_propio_states = curr_propio_pos
                else:
                    raise ValueError("Need proprio states to plan")

                backbone_output = self.model.backbone(
                    current_state.cuda(), propio=curr_propio_states.cuda()
                )
            else:
                backbone_output = self.model.backbone(current_state.cuda())

            current_state = backbone_output.encodings

        actions = []
        for i in range(batch_size):
            if self.last_plan_size is not None and plan_size < self.last_plan_size:
                for _ in range(self.last_plan_size - plan_size):
                    self.ctrls[i].shift_nominal_trajectory()

            self.ctrls[i].change_horizon(plan_size)

            # add refinement steps?
            actions.append(
                self.ctrls[i].command(
                    current_state[i],
                    shift_nominal_trajectory=False,
                )
            )

        actions = torch.stack(actions)

        pred_encs, pred_obs = self.dynamics(
            state=current_state,
            action=actions.permute(1, 0, 2),
            only_return_last=False,
            flatten_output=False,
        )

        if self.action_normalizer is not None:
            actions = self.action_normalizer(actions)

        actions = self.normalizer.unnormalize_action(actions)

        self.dynamics.after_planning_callback()
        self.last_plan_size = plan_size

        losses = [0]

        if self.prober is not None:
            pred_locs = torch.stack([self.prober(x) for x in pred_obs])
            unnormed_locations = self.normalizer.unnormalize_location(
                pred_locs
            ).detach()
        else:
            unnormed_locations = None

        return PlanningResult(
            pred_encs=pred_encs,
            pred_obs=pred_obs,
            actions=actions,
            locations=unnormed_locations,
            losses=losses,
        )

    def reset_targets(self, targets: torch.Tensor, repr_input: bool = True):
        self.objective.set_target(targets, repr_input=repr_input)
