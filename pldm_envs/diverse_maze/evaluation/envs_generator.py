import torch
from pldm_envs.diverse_maze.utils import load_uniform
from pldm_envs.diverse_maze.data_generation.maze_stats import RENDER_STATS
import pldm_envs.diverse_maze.ant_draw as ant_draw
from pldm_envs.utils.normalizer import Normalizer
from pldm_envs.diverse_maze.utils import sample_nearby_grid_location_v2
from pldm_envs.diverse_maze.wrappers import ActionRepeatWrapper, NormEvalWrapper

import numpy as np


def sample_location(uniform_obs, anchor=None):
    return uniform_obs[np.random.randint(0, len(uniform_obs))]


class EnvsGenerator:
    def __init__(
        self,
        env_name: str,
        n_envs: int,
        min_block_radius: int,
        max_block_radius: int,
        action_repeat: int,
        action_repeat_mode: str,
        stack_states: int = 1,
        image_obs: bool = True,
        data_path: str = None,
        trials_path: str = None,
        unique_shortest_path: bool = False,
        normalizer: Normalizer = None,
    ):
        self.env_name = env_name
        self.n_envs = n_envs
        self.min_block_radius = min_block_radius
        self.max_block_radius = max_block_radius
        self.action_repeat = action_repeat
        self.action_repeat_mode = action_repeat_mode
        self.stack_states = stack_states
        self.image_obs = image_obs
        self.data_path = data_path
        self.trials_path = trials_path
        self.unique_shortest_path = unique_shortest_path
        self.normalizer = normalizer

    def _sample_nearby_location(
        self, anchor, map_key, min_block_radius=-1, max_block_radius=3
    ):
        """
        Given the anchor coordinate in obs space, and a particualar map_key (layout),
        sample a target coordinate within {block_radius} blocks of the anchor
        """

        if "small" in self.env_name:
            obs_coord, block_dist, turns, unique_path = sample_nearby_grid_location_v2(
                anchor=anchor,
                map_key=map_key,
                min_block_radius=min_block_radius,
                max_block_radius=max_block_radius,
                obs_range_total=RENDER_STATS[self.env_name]["obs_range_total"],
                obs_min_total=RENDER_STATS[self.env_name]["obs_min_total"],
                unique_shortest_path=self.unique_shortest_path,
            )
        else:
            raise CustomError("medium setting no longer supported")

        return obs_coord, block_dist, turns

    def _make_env(
        self,
        start,
        target,
        map_idx=None,
        map_key=None,
        block_dist=None,
        turns=None,
        ood_dist=None,
        mode=None,
    ):
        if map_idx is not None:
            env_name = f"{self.env_name}_{map_idx}"
            env = ant_draw.load_environment(
                name=env_name, map_key=map_key, block_dist=block_dist, turns=turns
            )
        else:
            env_name = self.env_name
            env = ant_draw.load_environment(
                env_name, block_dist=block_dist, turns=turns
            )

        if self.action_repeat > 1:
            env = ActionRepeatWrapper(
                env,
                action_repeat=self.action_repeat,
                action_repeat_mode=self.action_repeat_mode,
            )

        env = NormEvalWrapper(env, self.normalizer, stack_states=self.stack_states)

        env.reset()
        if "ant" in env.name:
            env.set_state(qpos=start[:15], qvel=start[15:])
        else:
            env.set_state(qpos=start[:2], qvel=np.array([0, 0]))

        env.set_target(target[:2])
        env.start_xy = start[:2]
        env.ood_dist = ood_dist
        env.mode = mode

        return env

    def __call__(self):
        envs = []

        env_name = self.env_name
        n_envs = self.n_envs
        min_block_radius = self.min_block_radius
        max_block_radius = self.max_block_radius
        data_path = self.data_path
        trials_path = self.trials_path

        uniform_ds = load_uniform(env_name, data_path)

        if "diverse" in env_name:
            map_path = f"{data_path}/train_maps.pt"
            map_layouts = torch.load(map_path)
            map_keys = list(map_layouts.keys())

            if trials_path is not None and trials_path:
                trials = torch.load(trials_path)
                n_envs = min(n_envs, len(trials["map_layouts"]))

                for i in range(n_envs):
                    env = self._make_env(
                        start=trials["starts"][i],
                        target=trials["targets"][i],
                        map_idx=i,
                        map_key=trials["map_layouts"][i],
                        ood_dist=(
                            trials["ood_distance"][i]
                            if "ood_distance" in trials
                            else None
                        ),
                        mode=trials["mode"][i] if "mode" in trials else "train",
                    )
                    envs.append(env)
            else:
                trials = {
                    "starts": [],
                    "targets": [],
                    "map_layouts": [],
                    "block_dists": [],
                }

                if n_envs <= len(map_keys):
                    env_keys = map_keys[:n_envs]
                else:
                    env_keys = np.resize(map_keys, n_envs)

                for i, map_idx in enumerate(env_keys):
                    start = sample_location(uniform_ds[map_idx])

                    target, block_dist, turns = self._sample_nearby_location(
                        anchor=start[:2],
                        map_key=map_layouts[map_idx],
                        min_block_radius=min_block_radius,
                        max_block_radius=max_block_radius,
                    )

                    env = self._make_env(
                        start=start,
                        target=target,
                        map_idx=map_idx,
                        map_key=map_layouts[map_idx],
                        block_dist=block_dist,
                        turns=turns,
                    )
                    envs.append(env)

                    if block_dist >= 5 and len(trials["starts"]) < 80:
                        trials["starts"].append(start)
                        trials["targets"].append(target)
                        trials["map_layouts"].append(map_layouts[map_idx])
                        trials["block_dists"].append(block_dist)

                # if self.quick_debug:
                #     torch.save(trials, f"{self.config.data_path}/trials.pt")

        else:
            for i in range(n_envs):
                # there's just one layout. we give it 0 id by convention
                map_idx = 0

                env = self._make_env(
                    start=sample_location(uniform_ds[map_idx]),
                    target=sample_location(uniform_ds[map_idx]),
                )
                envs.append(env)

        return envs
