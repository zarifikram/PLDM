from pldm.configs import ConfigBase
import torch
from typing import Optional
from dataclasses import dataclass
import dataclasses

from pldm.probing.evaluator import ProbingConfig, ProbingEvaluator
from pldm.planning.wall.enums import WallMPCConfig
from pldm.data.enums import ProbingDatasets, DatasetType
from pldm.planning.d4rl.enums import D4RLMPCConfig
from pldm.planning.enums import LevelConfig
from pldm.planning.wall.mpc import WallMPCEvaluator
from omegaconf import MISSING

from pldm_envs.utils.normalizer import Normalizer

from pldm_envs.diverse_maze.utils import PixelMapper as D4RLPixelMapper


@dataclass
class EvalConfig(ConfigBase):
    env_name: str = MISSING
    probing: ProbingConfig = ProbingConfig()
    eval_l1: bool = True
    eval_l2: bool = False
    log_heatmap: bool = True
    disable_planning: bool = False
    wall_planning: WallMPCConfig = WallMPCConfig()
    d4rl_planning: D4RLMPCConfig = D4RLMPCConfig()

    def __post_init__(self):
        self.wall_planning.env_name = self.env_name
        self.d4rl_planning.env_name = self.env_name


class Evaluator:
    def __init__(
        self,
        config: EvalConfig,
        model: torch.nn.Module,
        quick_debug: bool,
        normalizer: Normalizer,
        epoch: int,
        probing_datasets: Optional[ProbingDatasets],
        l2_probing_datasets: Optional[ProbingDatasets],
        load_checkpoint_path: "",
        output_path: "",
        data_config=None,
    ):
        self.config = config
        self.model = model
        self.quick_debug = quick_debug
        self.normalizer = normalizer
        self.epoch = epoch
        self.output_path = output_path
        self.data_config = data_config  # wall_config

        self.probing_evaluator = ProbingEvaluator(
            model=self.model,
            config=self.config.probing,
            quick_debug=self.quick_debug,
            probing_datasets=probing_datasets,
            l2_probing_datasets=l2_probing_datasets,
            load_checkpoint_path=load_checkpoint_path,
            output_path=output_path,
        )

        self.pixel_mapper = self._create_pixel_mapper()
        self.planning_config = self._get_planning_config()

    def _get_planning_config(self):
        if "diverse" in self.config.env_name or "maze" in self.config.env_name:
            config = self.config.d4rl_planning
        elif self.config.env_name == "wall":
            config = self.config.wall_planning
        else:
            raise NotImplementedError
        return config

    def evaluate_loc_probing(self):
        probers = {}

        if self.config.eval_l1:
            probers = self.probing_evaluator.train_pred_prober(
                epoch=self.epoch,
            )

            if self.config.probing.probe_preds:
                self.probing_evaluator.evaluate_all(
                    probers=probers,
                    epoch=self.epoch,
                    pixel_mapper=self.pixel_mapper.obs_coord_to_pixel_coord,
                )

            if self.config.probing.probe_encoder:
                enc_probers = self.probing_evaluator.train_encoder_prober(
                    epoch=self.epoch,
                )

                enc_probe_loss = self.probing_evaluator.eval_probe_enc_position(
                    probers=enc_probers,
                    epoch=self.epoch,
                )

        return probers, None

    def _create_pixel_mapper(self):
        if "diverse" in self.config.env_name or "maze2d" in self.config.env_name:
            pixel_mapper = D4RLPixelMapper(env_name=self.config.env_name)
        else:

            class IdPixelMapper:
                def obs_coord_to_pixel_coord(self, x):
                    return x

                def pixel_coord_to_obs_coord(self, x):
                    return x

            pixel_mapper = IdPixelMapper()

        return pixel_mapper

    def _create_l1_planning_evaluator(
        self,
        level: str,
        level_config: LevelConfig,
    ):
        if level_config.override_config:
            max_plan_length = level_config.max_plan_length
            n_envs = level_config.n_envs
            n_steps = level_config.n_steps
            offline_T = level_config.offline_T
            plot_every = level_config.plot_every
            if self.quick_debug:
                max_plan_length = self.planning_config.level1.max_plan_length
                n_envs = self.planning_config.n_envs
                n_steps = self.planning_config.n_steps
        else:
            max_plan_length = self.planning_config.level1.max_plan_length
            n_envs = self.planning_config.n_envs
            n_steps = self.planning_config.n_steps
            offline_T = self.planning_config.offline_T
            plot_every = self.planning_config.plot_every

        planner_config = dataclasses.replace(
            self.planning_config.level1,
            max_plan_length=max_plan_length,
        )

        mpc_config = dataclasses.replace(
            self.planning_config,
            level=level,
            n_envs=n_envs,
            n_steps=n_steps,
            level1=planner_config,
            offline_T=offline_T,
            plot_every=plot_every,
        )

        if "diverse" in self.config.env_name or "maze2d" in self.config.env_name:
            from pldm.planning.d4rl.mpc import MazeMPCEvaluator

            planning_evaluator = MazeMPCEvaluator(
                config=mpc_config,
                normalizer=self.normalizer,
                jepa=self.model.level1,
                pixel_mapper=self.pixel_mapper,
                prober=self.probers["locations"],
                prefix=f"d4rl_{level}",
                quick_debug=self.quick_debug,
            )
        elif self.config.env_name == "wall":
            planning_evaluator = WallMPCEvaluator(
                config=mpc_config,
                normalizer=self.normalizer,
                jepa=self.model.level1,
                prober=self.probers["locations"],
                prefix=f"wall_{level}",
                quick_debug=self.quick_debug,
                wall_config=dataclasses.replace(self.data_config, train=False),
            )
        else:
            raise NotImplementedError

        return planning_evaluator

    def _get_planning_levels(self):
        levels = self.planning_config.levels.split(",")

        # if self.quick_debug:
        #     levels = [levels[0]]

        level_configs = [getattr(self.planning_config, level) for level in levels]

        return (levels, level_configs)

    def evaluate(self):
        """
        Evaluation consists of both probing and planning
        """

        log_dict = {}

        self.probers, self.probers_l2 = self.evaluate_loc_probing()

        # Planning
        if not self.config.disable_planning:
            if self.config.eval_l1:
                levels, level_configs = self._get_planning_levels()

                for i, level in enumerate(levels):
                    level_config = level_configs[i]

                    planning_evaluator = self._create_l1_planning_evaluator(
                        level=level,
                        level_config=level_config,
                    )

                    print(
                        f"evaluating planning level {level} for {planning_evaluator.config.n_envs} envs"
                    )

                    mpc_result, mpc_report = planning_evaluator.evaluate()

                    planning_evaluator.close()

                    log_dict.update(
                        mpc_report.build_log_dict(prefix=planning_evaluator.prefix)
                    )

                    torch.save(
                        mpc_result,
                        f"{self.output_path}/planning_l1_mpc_result_{planning_evaluator.prefix}",
                    )
                    torch.save(
                        mpc_report,
                        f"{self.output_path}/planning_l1_mpc_report_{planning_evaluator.prefix}",
                    )

        self.model.train()

        return log_dict
