from pldm.planning import objectives_v2
import torch
from pldm.models.jepa import JEPA
from pldm_envs.utils.normalizer import Normalizer
from pldm.planning.planners.enums import PlannerType
from pldm.planning.planners.mppi_planner import MPPIPlanner
from pldm.planning.planners.sgd_planner import SGDPlanner
from pldm.planning.utils import normalize_actions
from abc import ABC
from pldm.planning.enums import MPCResult, PooledMPCResult
import numpy as np
from pldm.models.utils import flatten_conv_output
from tqdm import tqdm


class MPCEvaluator(ABC):
    def __init__(
        self,
        config,
        model: JEPA,
        prober: torch.nn.Module,
        normalizer: Normalizer,
        quick_debug: bool = False,
        prefix: str = "",
        pixel_mapper=None,
        image_based=True,
    ):
        self.config = config
        self.model = model
        self.prober = prober
        self.normalizer = normalizer
        self.quick_debug = quick_debug
        self.prefix = prefix
        self.pixel_mapper = pixel_mapper
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.image_based = image_based

    def close(self):
        pass

    def _infer_chunk_sizes(self):
        config = self.config

        if config.level1.planner_type == PlannerType.MPPI:
            # n_envs_batch_size = 500000 // (config.num_samples * config.max_plan_length)
            n_envs_batch_size = config.n_envs_batch_size
        else:
            n_envs_batch_size = config.n_envs

        chunk_sizes = [n_envs_batch_size] * (config.n_envs // n_envs_batch_size) + (
            [config.n_envs % n_envs_batch_size]
            if config.n_envs % n_envs_batch_size != 0
            else []
        )

        return chunk_sizes

    def _construct_planner(self, n_envs: int):
        config = self.config

        objective = objectives_v2.ReprTargetMPCObjective(
            model=self.model,
            propio_cost=config.level1.propio_cost,
            sum_all_diffs=config.level1.sum_all_diffs,
            loss_coeff_first=config.level1.loss_coeff_first,
            loss_coeff_last=config.level1.loss_coeff_last,
        )

        action_normalizer = lambda x: normalize_actions(
            x,
            min_norm=config.level1.min_step,
            max_norm=config.level1.max_step,
            xy_action=True,
            clamp_actions=config.level1.clamp_actions,
        )

        if config.level1.planner_type == PlannerType.MPPI:
            planner = MPPIPlanner(
                config.level1.mppi,
                model=self.model,
                normalizer=self.normalizer,
                objective=objective,  # [300, 13456]
                prober=self.prober,
                action_normalizer=action_normalizer,
                n_envs=n_envs,
                projected_cost=config.level1.projected_cost,
            )
        elif config.level1.planner_type == PlannerType.SGD:
            planner = SGDPlanner(
                config.level1.sgd,
                model=self.model,
                normalizer=self.normalizer,
                objective=objective,
                prober=self.prober,
                action_normalizer=action_normalizer,
            )
        else:
            raise NotImplementedError(
                f"Unknown planner type {config.level1.planner_type}"
            )

        return planner

    def _perform_mpc_in_chunks(self):
        """
        Divide it up in chunks in order to prevent OOM
        """
        chunk_sizes = self._infer_chunk_sizes()

        mpc_data = PooledMPCResult()
        chunk_offset = 0

        for chunk_size in chunk_sizes:
            planner = self._construct_planner(n_envs=chunk_size)
            envs = self.envs[chunk_offset : chunk_offset + chunk_size]

            mpc_result = self._perform_mpc(
                planner=planner,
                envs=envs,
            )

            obs_c = mpc_result.observations
            location_history_c = mpc_result.locations
            action_history_c = mpc_result.action_history
            reward_history_c = mpc_result.reward_history
            pred_locations_c = mpc_result.pred_locations
            final_preds_dist_c = mpc_result.final_preds_dist
            targets_c = mpc_result.targets
            loss_history_c = mpc_result.loss_history
            qpos_history_c = mpc_result.qpos_history
            propio_history_c = mpc_result.propio_history

            mpc_data.observations.append(obs_c)
            mpc_data.locations.append(location_history_c)
            mpc_data.action_history.append(action_history_c)
            mpc_data.reward_history.append(reward_history_c)
            mpc_data.pred_locations.append(pred_locations_c)
            mpc_data.final_preds_dist.append(final_preds_dist_c)
            mpc_data.targets.append(targets_c)
            mpc_data.loss_history.append(loss_history_c)
            mpc_data.qpos_history.append(qpos_history_c)
            mpc_data.propio_history.append(propio_history_c)

            chunk_offset += chunk_size

        mpc_data.concatenate_chunks()

        return mpc_data

    def _perform_mpc(
        self,
        planner,
        envs,
    ):
        """
        Parameters:
            starts: (bs, 4)
            targets: (bs, 4)
        Outputs:
            observations: list of a_T (bs, 3, 64, 64) or (bs, 2)
            locations: list of a_T (bs, 2)
            action_history: list of a_T (bs, p_T, 2)
            reward_history: list of a_T (bs,)
            pred_locations: list of a_T (p_T, bs, 1, 2)
            targets: (bs, 4)
            loss_history: list of a_T (n_iters,)
        """

        targets = [e.get_target() for e in envs]
        targets = torch.from_numpy(np.stack(targets))

        targets_t = torch.stack([e.get_target_obs() for e in envs]).to(self.device)

        # encode target obs
        if self.model.config.backbone.propio_dim is not None:
            # for target we don't care about the proprioceptive states. just make it zero.
            propio_states = torch.zeros(
                (targets_t.shape[0], self.model.config.backbone.propio_dim)
            ).to(self.device)
            targets_t = self.model.backbone(
                targets_t, propio=propio_states
            ).obs_component.detach()
        else:
            targets_t = self.model.backbone(targets_t).obs_component.detach()

        targets_t = flatten_conv_output(targets_t)
        planner.reset_targets(targets_t, repr_input=True)

        observation_history = [torch.stack([e.get_obs() for e in envs])]

        obs_t = observation_history[0]
        if self.image_based:
            obs_t = torch.cat([obs_t] * self.config.stack_states, dim=1)  # VERIFY

        action_history = []
        reward_history = []
        location_history = []
        qpos_history = []
        propio_history = []

        pred_positions_history = []
        loss_history = []
        final_preds_dist_history = []

        init_infos = [e.get_info() for e in envs]
        if "location" in init_infos[0]:
            location_history.append(np.array([info["location"] for info in init_infos]))

        if "qpos" in init_infos[0]:
            qpos_history.append(np.array([info["qpos"] for info in init_infos]))

        if "propio" in init_infos[0]:
            propio_history.append(np.array([info["propio"] for info in init_infos]))

        for i in tqdm(range(self.config.n_steps), desc="Planning steps"):
            if i % self.config.replan_every == 0:

                if planner.model.use_propio_pos:
                    curr_propio_pos = [e.get_propio_pos(normalized=True) for e in envs]
                    curr_propio_pos = torch.from_numpy(
                        np.stack(curr_propio_pos)
                    ).float()
                else:
                    curr_propio_pos = None

                if planner.model.use_propio_vel:
                    curr_propio_vel = [e.get_propio_vel(normalized=True) for e in envs]
                    curr_propio_vel = torch.from_numpy(
                        np.stack(curr_propio_vel)
                    ).float()
                else:
                    curr_propio_vel = None

                planning_result = planner.plan(
                    obs_t,
                    curr_propio_pos=curr_propio_pos,
                    curr_propio_vel=curr_propio_vel,
                    plan_size=min(
                        self.config.n_steps - i, self.config.level1.max_plan_length
                    ),
                    repr_input=False,
                )

            last_pred_obs = flatten_conv_output(planning_result.pred_obs)

            pred_dist = torch.norm(last_pred_obs - targets_t.unsqueeze(0), dim=2).cpu()
            final_preds_dist_history.append(pred_dist)

            planned_actions = (
                planning_result.actions[:, i % self.config.replan_every :]
                .detach()
                .cpu()
            )

            if self.config.random_actions:
                results = [
                    envs[j].step(envs[0].action_space.sample())
                    for j in range(len(envs))
                ]
            else:
                results = [
                    envs[j].step(
                        planned_actions[j, 0].detach().cpu().contiguous().numpy()
                    )
                    for j in range(len(envs))
                ]

            assert len(results[0]) == 5
            current_obs = torch.from_numpy(np.stack([r[0] for r in results])).float()
            rewards_t = torch.from_numpy(np.stack([r[1] for r in results])).float()
            infos = [r[4] for r in results]

            action_history.append(planned_actions.detach().cpu())
            observation_history.append(current_obs)
            reward_history.append(rewards_t)

            if "location" in infos[0]:
                location_history.append(np.array([info["location"] for info in infos]))

            if "qpos" in infos[0]:
                qpos_history.append(np.array([info["qpos"] for info in infos]))

            if "propio" in infos[0]:
                propio_history.append(np.array([info["propio"] for info in infos]))

            if planning_result.locations is not None:
                pred_locations = planning_result.locations.detach().cpu()
                pred_locations = pred_locations.squeeze(2)
                pred_positions_history.append(pred_locations)

            # stack states if necessary for next iteration
            if self.config.stack_states == 1:
                obs_t = current_obs
            else:
                obs_t = torch.cat(
                    [obs_t[:, current_obs.shape[1] :], current_obs], dim=1
                )

            loss_history.append(planning_result.losses)

        observation_history = [
            self.normalizer.unnormalize_state(o) for o in observation_history
        ]

        return MPCResult(
            observations=observation_history,
            locations=[torch.from_numpy(x) for x in location_history],
            action_history=action_history,
            reward_history=reward_history,
            pred_locations=pred_positions_history,
            final_preds_dist=final_preds_dist_history,
            targets=targets,
            loss_history=loss_history,
            qpos_history=[torch.from_numpy(x) for x in qpos_history],
            propio_history=[torch.from_numpy(x) for x in propio_history],
        )
