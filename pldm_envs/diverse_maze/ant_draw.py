from __future__ import annotations
from typing import *

import os
import logging

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data
import gym
import gym.spaces

from contextlib import (
    contextmanager,
    redirect_stderr,
    redirect_stdout,
)


@contextmanager
def suppress_output():
    """
    A context manager that redirects stdout and stderr to devnull
    https://stackoverflow.com/a/52442331
    """
    with open(os.devnull, "w") as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)


import d4rl
import d4rl.locomotion

from gym.envs.registration import register
from d4rl.pointmaze import MazeEnv

# maze2d_medium_diverse = (
#     "########\\"
#     + "#OO##OO#\\"
#     + "##O#OOO#\\"
#     + "##OOO###\\"
#     + "#OO#OOO#\\"
#     + "#O#OO#O#\\"
#     + "#OOO#OG#\\"
#     + "########"
# )

# Step 3: Register the Custom Environment
# register(
#     id="maze2d_medium_diverse",
#     entry_point="data.d4rl.ant_draw:CustomMazeEnv",  # Change this to match your file name
#     max_episode_steps=600,
# )


def load_environment(
    name: str,
    map_key: str = None,
    block_dist: int = None,
    turns: int = None,
    max_episode_steps: int = 600,
):  # NB: this removes the TimeLimit wrapper
    # create custom environment if layout is provided
    if map_key is not None:
        # Define Custom Environment
        class CustomMazeEnv(MazeEnv):
            def __init__(self, **kwargs):
                # Call the __init__ method of MazeEnv with the custom maze layout
                super(CustomMazeEnv, self).__init__(maze_spec=map_key, **kwargs)

            def step(self, action):
                # Call the original step method to get next state, reward, etc.
                next_state, original_reward, done, info = super().step(action)

                # Define a binary reward based on some condition
                # Example: reward is 1 if the agent reaches the goal, otherwise 0
                if self._is_goal_reached():  # You can define your own condition here
                    reward = 1
                else:
                    reward = 0

                return next_state, reward, done, info

            def _is_goal_reached(self, goal_dist_threshold=0.5):
                # Check if the agent has reached the goal
                # You can define how you determine if the goal is reached, e.g.,
                # by comparing the agent's current position with the goal position
                current_position = self._get_obs()[:2]  # Custom method to get position
                goal_position = self.get_target()  # Custom method to get goal
                distance_to_goal = np.linalg.norm(current_position - goal_position)

                # Example condition: reward is 1 if agent is within a certain distance of the goal
                return (
                    distance_to_goal < goal_dist_threshold
                )  # Adjust threshold as needed

        register(
            id=name,
            entry_point=lambda: CustomMazeEnv(),  # Change this to match your file name
            max_episode_steps=max_episode_steps,
        )

        def convert_to_binary_array(map_key):
            # Create a 2D binary numpy array by iterating over the list of strings
            binary_array = np.array(
                [
                    [1 if char == "#" else 0 for char in string]
                    for string in map_key.split("\\")
                ]
            )
            return binary_array

        maze_map = convert_to_binary_array(map_key)
    else:
        maze_map = None

    with suppress_output():
        wrapped_env: gym.Wrapper = gym.make(name)

    env = cast(gym.Env, wrapped_env.unwrapped)
    env.max_episode_steps = wrapped_env._max_episode_steps
    env.name = name
    env.maze_map = maze_map
    env.map_key = map_key
    env.reset()
    env.step(
        env.action_space.sample()
    )  # sometimes stepping is needed to initialize internal
    env.reset()

    # custom attributes
    env.block_dist = block_dist
    env.turns = turns

    return env


def antmaze_fix_timeouts(env, dataset: Mapping[str, np.ndarray]):
    # https://gist.github.com/jannerm/d5ea90f17878b3fa198daf7dec67dfde#file-diffuser_antmaze-py-L1-L66

    logging.info("[ datasets/d4rl ] Fixing timeouts")
    N = len(dataset["observations"])
    max_episode_steps = np.where(dataset["timeouts"])[0][
        0
    ]  # usually 1000, sometimes 700

    ## 1000, 2001, 3002, 4003, ...
    timeouts = [max_episode_steps] + (
        np.arange(
            max_episode_steps + 1,
            N - max_episode_steps,
            max_episode_steps + 1,
        )
        + max_episode_steps
    ).tolist()
    timeouts = np.array(timeouts)

    timeouts_bool = np.zeros_like(dataset["timeouts"])
    timeouts_bool[timeouts] = 1

    assert np.all(timeouts_bool == dataset["timeouts"]), "sanity check"

    # dataset['timeouts'] = timeouts_bool
    dataset["terminals"][:] = 0

    fixed = {
        "observations": [],
        "actions": [],
        "next_observations": [],
        "rewards": [],
        "terminals": [],
        "timeouts": [],
        "steps": [],
    }
    step = 0
    for i in range(N - 1):
        done = dataset["terminals"][i] or dataset["timeouts"][i]

        if done:
            ## this is the last observation in its trajectory,
            ## cannot add a next_observation to this transition
            # print(i, step)
            step = 0
            continue

        for key in ["observations", "actions", "rewards", "terminals"]:
            val = dataset[key][i]
            fixed[key].append(val)

        next_observation = dataset["observations"][i + 1]
        fixed["next_observations"].append(next_observation)

        timeout = dataset["timeouts"][i + 1]
        fixed["timeouts"].append(timeout)

        fixed["steps"].append(step)

        step += 1

    fixed = {key: np.asarray(val) for key, val in fixed.items()}

    data_max_episode_steps = fixed["steps"].max() + 1
    logging.info(f"[ datasets/d4rl ] Max episode length: {max_episode_steps} (env)")
    logging.info(
        f"[ datasets/d4rl ] Max episode length: {data_max_episode_steps} (data)"
    )

    return fixed


def antmaze_scale_rewards(dataset, mult=10):
    # https://gist.github.com/jannerm/d5ea90f17878b3fa198daf7dec67dfde#file-diffuser_antmaze-py-L1-L66

    dataset["rewards"] = dataset["rewards"] * mult
    logging.info(f"[ datasets/d4rl ] Scaled rewards by {mult}")
    return dataset


def antmaze_get_dataset(env, reward_scale=1):
    dataset = env.get_dataset()
    if env.name.startswith("antmaze"):
        dataset = antmaze_fix_timeouts(env, dataset)
        dataset = antmaze_scale_rewards(dataset, reward_scale)
    return dataset


# -----------------------------------------------------------------------------#
# ---------------------------------- drawer -----------------------------------#
# -----------------------------------------------------------------------------#


class D4RLAntMazeDrawer:
    def __init__(
        self,
        env: d4rl.locomotion.ant.AntMazeEnv,
        *,
        lookat: List[float],
        distance: float,
        image_topleft_in_obs_coord: List[float],
        scale_coord_obs_to_pixel: float,
        arrow_mult: float,
    ):
        self.env = env
        self.env.render(mode="rgb_array")  # initialize viewer
        self.env.viewer.cam.elevation = -90
        self.env.viewer.cam.distance = distance
        self.env.viewer.cam.lookat[:] = np.asarray(lookat, dtype=np.float64)

        self.image_topleft_in_obs_coord = torch.as_tensor(image_topleft_in_obs_coord)
        self.scale_coord_obs_to_pixel = scale_coord_obs_to_pixel
        self.arrow_mult = arrow_mult

    def obs_coord_to_pixel_coord(self, coord) -> Tuple[int, int]:
        coord = torch.as_tensor(coord)
        jj, ii = (
            (coord - self.image_topleft_in_obs_coord)
            .mul(self.scale_coord_obs_to_pixel)
            .round()
            .long()
        )
        return (jj.item(), -ii.item())

    def pixel_coord_to_obs_coord(self, coord) -> Tuple[float, float]:
        coord = (
            torch.as_tensor(coord, dtype=torch.float32) / self.scale_coord_obs_to_pixel
        )
        coord[1].neg_()
        x, y = coord + self.image_topleft_in_obs_coord
        return (x.item(), y.item())

    def render_state(self, obs: Any) -> np.ndarray:
        obs = torch.as_tensor(obs).data.cpu().numpy()[..., :29]

        if "ant" in self.env.name:
            qpos = np.copy(self.env.physics.data.qpos)
            qvel = np.copy(self.env.physics.data.qvel)
            qpos_slice = slice(0, 15)
            qvel_slice = slice(15, None)
        else:
            qpos = np.copy(self.env.sim.data.qpos)
            qvel = np.copy(self.env.sim.data.qvel)
            qpos_slice = slice(0, 2)
            qvel_slice = slice(2, None)

        qpos[qpos_slice] = obs[qpos_slice]
        qvel[: len(obs[qvel_slice])] = obs[qvel_slice]

        self.env.set_state(qpos, qvel)

        return self.env.render("rgb_array")

    @torch.no_grad()
    def __call__(
        self, obs, ax: Union[None, plt.Axes, Tuple[plt.Artist, ...]] = None
    ) -> Tuple[plt.Axes, Tuple[plt.Artist, ...]]:
        r"""
        This can do more than just imshow `render_state`'s rgb image.

        Returns (ax, artists).

        Input `ax` can be (1) None, (2) an plt.Axes, or (3) a tuple (ax, artists).

        For case (3), update will be inplace.
        """
        obs = torch.as_tensor(obs).data.cpu()[..., :29]
        rgb = self.render_state(obs)
        if isinstance(ax, tuple):
            im, arrow = ax
            arrow.remove()
            im.set_data(rgb)
            ax = im.axes
        else:
            if ax is None:
                ax = plt.gca()
            im = ax.imshow(rgb)
        ax.autoscale(False)
        if "ant" in self.env.name:
            velocity = obs[15:17]
        else:
            velocity = obs[2:4]
        arrow = ax.arrow(
            *self.obs_coord_to_pixel_coord(obs[:2]),
            *velocity.mul(torch.tensor([1, -1])).mul(self.arrow_mult),
            width=3,
            color="red",
            alpha=0.6,
        )
        return ax, (im, arrow)


def antmaze_get_drawer(env, draw_velocity=False):
    if "ant" in env.name:
        if "umaze" in env.name:
            return D4RLAntMazeDrawer(
                env,
                lookat=[4, 4, 0],
                distance=30,
                image_topleft_in_obs_coord=[-8, 16],
                scale_coord_obs_to_pixel=500 / (8 + 16),
                arrow_mult=50,
            )
        elif "medium" in env.name:
            return D4RLAntMazeDrawer(
                env,
                lookat=[10, 10, 0],
                distance=45,
                image_topleft_in_obs_coord=[-8.25, 28.25],
                scale_coord_obs_to_pixel=500 / (8.25 + 28.25),
                arrow_mult=50,
            )
        elif "large" in env.name:
            return D4RLAntMazeDrawer(
                env,
                lookat=[18, 18, 0],
                distance=200 / 3,
                image_topleft_in_obs_coord=[-9.1, 45.1],
                scale_coord_obs_to_pixel=500 / (9.1 + 45.1),
                arrow_mult=50,
            )
    else:
        if "umaze" in env.name:
            return D4RLAntMazeDrawer(
                env,
                lookat=[3, 3, 0],
                distance=15,
                image_topleft_in_obs_coord=[-8, 16],
                scale_coord_obs_to_pixel=500 / (8 + 16),
                arrow_mult=1,
            )
        elif "medium" in env.name:
            return D4RLAntMazeDrawer(
                env,
                lookat=[10, 10, 0],
                distance=45,
                image_topleft_in_obs_coord=[-8.25, 28.25],
                scale_coord_obs_to_pixel=500 / (8.25 + 28.25),
                arrow_mult=50,
            )
        elif "large" in env.name:
            return D4RLAntMazeDrawer(
                env,
                lookat=[18, 18, 0],
                distance=200 / 3,
                image_topleft_in_obs_coord=[-9.1, 45.1],
                scale_coord_obs_to_pixel=500 / (9.1 + 45.1),
                arrow_mult=50,
            )


def set_state(env, state):
    if "ant" in env.name:
        qpos = np.copy(env.physics.data.qpos)
        qvel = np.copy(env.physics.data.qvel)
        qpos_slice = slice(0, 15)
        qvel_slice = slice(15, None)
    else:
        qpos = np.copy(env.sim.data.qpos)
        qvel = np.copy(env.sim.data.qvel)
        qpos_slice = slice(0, 2)
        qvel_slice = slice(2, None)

    qpos[qpos_slice] = state[qpos_slice]
    qvel[: len(state[qvel_slice])] = state[qvel_slice]

    env.set_state(qpos, qvel)
