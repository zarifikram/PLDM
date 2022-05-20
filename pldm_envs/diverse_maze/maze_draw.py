from __future__ import annotations
from typing import *

import logging

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data
import gym
import gym.spaces

from PIL import Image

import d4rl.pointmaze
from pldm_envs.diverse_maze.transforms import select_transforms
from pldm_envs.diverse_maze.data_generation.maze_stats import RENDER_STATS
from pldm_envs.diverse_maze.utils import PixelMapper


def preprocess_maze2d_fix(
    env: d4rl.pointmaze.MazeEnv, dataset: Mapping[str, np.ndarray]
):
    ## In generation, controller is run until reached goal, which is
    ## continuously set.
    ##
    ## There, terminal is always False, and how timeout is set is unknown (not
    ## in the public script)
    ##
    ## Set timeout at time t                      (*)
    ##   iff reached goal at time t
    ##   iff goal at time t != goal at time t+1
    ##
    ## Remove terminals
    ##
    ## Add next_observations
    ##
    ## Also Maze2d *rewards* is field is off-by-one:
    ##    rewards[t] is not the reward received for performing actions[t] at observation[t].
    ## Rather, it is the reward to be received for transitioning *into* observation[t].
    ##
    ## What a mess... This fixes that too.
    ##
    ## NB that this is different from diffuser code!

    assert not np.any(dataset["terminals"])
    dataset["next_observations"] = dataset["observations"][1:]

    goal_diff = np.abs(dataset["infos/goal"][:-1] - dataset["infos/goal"][1:]).sum(
        -1
    )  # diff with next
    timeouts = goal_diff > 1e-5

    timeout_steps = np.where(timeouts)[0]
    path_lengths = timeout_steps[1:] - timeout_steps[:-1]

    logging.info(
        f"[ preprocess_maze2d_fix ] Segmented {env.name} | {len(path_lengths)} paths | "
        f"min length: {path_lengths.min()} | max length: {path_lengths.max()}"
    )

    dataset["timeouts"] = timeouts

    logging.info("[ preprocess_maze2d_fix ] Fixed terminals and timeouts")

    # Fix rewards
    assert len(env.goal_locations) == 1
    rewards = cast(
        np.ndarray,
        np.linalg.norm(
            dataset["next_observations"][:, :2] - env.get_target(),
            axis=-1,
        )
        <= 0.5,
    ).astype(dataset["rewards"].dtype)
    # check that it was wrong :/
    assert (rewards == dataset["rewards"][1:]).all()
    dataset["rewards"] = rewards
    logging.info("[ preprocess_maze2d_fix ] Fixed rewards")

    # put things back into a new dict
    dataset = dict(dataset)
    for k in dataset:
        if dataset[k].shape[0] != dataset["next_observations"].shape[0]:
            dataset[k] = dataset[k][:-1]
    return dataset


def get_d4rl_dataset(env) -> Mapping[str, np.ndarray]:
    # return preprocess_maze2d_fix(env, env.get_dataset())
    return env.get_dataset()


class D4RLMaze2DDrawer(object):
    def __init__(
        self,
        env: d4rl.pointmaze.MazeEnv,
        *,
        lookat: List[float],
        arrow_mult: float,
    ):
        self.env = env
        self.env.render(mode="rgb_array")  # initialize viewer
        self.env.viewer.cam.elevation = -90
        self.env.viewer.cam.lookat[:] = np.asarray(lookat, dtype=np.float64)

        self.arrow_mult = arrow_mult

        self.pixel_mapper = PixelMapper(env_name=env.name)

    def render_state(self, obs: Any) -> np.ndarray:
        self.env.set_state(
            *[v.numpy() for v in torch.as_tensor(obs).data.cpu()[:4].chunk(2, dim=-1)]
        )
        return self.env.render("rgb_array")

    @torch.no_grad()
    def __call__(self, obs, ax: Union[None, plt.Axes] = None) -> plt.Axes:
        obs = torch.as_tensor(obs).data.cpu()[:4]
        rgb = self.render_state(obs)
        if ax is None:
            ax = plt.gca()
        ax.imshow(rgb)
        ax.autoscale(False)
        arrow = ax.arrow(
            *self.pixel_mapper.obs_coord_to_pixel_coord(obs[:2]),
            *obs[2:].mul(torch.tensor([1, -1])).mul(self.arrow_mult),
            width=6,
            color="red",
            alpha=0.6,
        )
        return ax


def render_umaze(env, obs, set_to_obs=True, normalizer=None):
    drawer = create_drawer(env, env.name)
    og_state = env.unwrapped._get_obs()

    transforms = select_transforms(drawer.env.name)
    if len(obs.shape) == 1:
        image = Image.fromarray(np.uint8(drawer.render_state(obs)))
        image = transforms(image)
        output = torch.from_numpy(np.array(image)).permute(2, 0, 1)
    else:
        images = []
        for o in obs:
            image = Image.fromarray(np.uint8(drawer.render_state(o)))
            image = transforms(image)
            images.append(torch.from_numpy(np.array(image)).permute(2, 0, 1))
        output = torch.stack(images)

    if not set_to_obs:
        if "ant" in drawer.env.name:
            env.set_state(qpos=og_state[:15], qvel=og_state[15:])
        else:
            env.set_state(qpos=og_state[:2], qvel=og_state[2:])

    if normalizer is not None:
        output = normalizer.normalize_state(output)

    return output


def create_drawer(env, env_id) -> "D4RLMaze2DDrawer":
    if env is None:
        env = gym.make(env_id)
        env.name = env_id

    if "diverse" in env_id:
        key_word = "diverse"
        index = env_id.find(key_word)
        if index != -1:
            env_id = env_id[: index + len(key_word)]

    stats = RENDER_STATS[env_id]

    return D4RLMaze2DDrawer(env, lookat=stats["lookat"], arrow_mult=stats["arrow_mult"])
