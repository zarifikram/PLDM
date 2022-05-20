import os
import sys
import argparse
import yaml

from matplotlib import pyplot as plt
import numpy as np
import torch
from tqdm.auto import tqdm
import random
import math

from scipy.stats import truncnorm
from pldm_envs.diverse_maze.data_generation.map_generator import MapGenerator
from pldm_envs.diverse_maze import ant_draw
from pldm_envs.diverse_maze.data_generation.wrappers import OGBenchWrapper
from pldm_envs.diverse_maze.wrappers import ActionRepeatWrapper

from pldm_envs.utils.distributions import sample_tapered_distribution
from pldm_envs.utils.utils import sample_vector


class UniformDataGenerator:
    def __init__(
        self,
        config,
        output_path,
        map_path=None,
        exclude_map_path=None,
    ):
        self.output_path = output_path
        self.map_path = map_path
        self.exclude_map_path = exclude_map_path

        with open(config, "r") as file:
            config = yaml.safe_load(file)
        self.config = config

        self.n_episodes = self.config["n_episodes"]
        self.episode_length = self.config["episode_length"]
        self.margin = 0.5

        self.dataset_type = self.config.get("dataset_type", "explore")
        self.noise = self.config.get("noise", 0)
        self.sample_every = self.config.get("sample_every", 1)
        self.ogbench_gather = self.config.get("ogbench_gather", False)

        # point maze init qvel distribution params
        if self.config["qvel_norm_prior"]:
            qvel_lower_bd = -5.2
            qvel_upper_bd = 5.2
            qvel_mean = 0
            qvel_std = 1.6

            a = (qvel_lower_bd - qvel_mean) / qvel_std
            b = (qvel_upper_bd - qvel_mean) / qvel_std
            self.qvel_dist = truncnorm(a, b, loc=qvel_mean, scale=qvel_std)

        os.makedirs(self.output_path, exist_ok=True)
        # generate map layouts if in diverse setting
        if "diverse" in self.config["env"]:
            if self.map_path is None:
                train_maps_n = self.config["train_maps_n"]
                map_generator = MapGenerator(
                    width=self.config["num_blocks_width_in_img"] - 2,
                    height=self.config["num_blocks_width_in_img"] - 2,
                    num_maps=train_maps_n,
                    sparsity_low=self.config.get("sparsity_low", 53),
                    sparsity_high=self.config.get("sparsity_high", 88),
                    max_path_len=self.config.get("max_path_len", 13),
                    exclude_map_path=self.exclude_map_path,
                    wall_coords=self.config.get("wall_coords", []),
                    space_coords=self.config.get("space_coords", []),
                )
                self.diverse_maps = map_generator.generate_diverse_maps()
                save_map_path = f"{self.output_path}/train_maps.pt"
                torch.save(self.diverse_maps, save_map_path)
            else:
                self.diverse_maps = torch.load(self.map_path)
                n_maps = self.config["train_maps_n"]
                if len(self.diverse_maps) > n_maps:
                    self.diverse_maps = dict(list(self.diverse_maps.items())[:n_maps])
                    save_map_path = f"{self.output_path}/train_maps.pt"
                    torch.save(self.diverse_maps, save_map_path)

        # get min and max state values for open layout
        if "medium" in self.config["env"]:
            empty_map_key = "########\\#OOOOOO#\\#OOOOOO#\\#OOOOOO#\\#OOOOOO#\\#OOOOOO#\\#OOOOOO#\\########"
        elif "small" in self.config["env"]:
            empty_map_key = (
                "######\\#OOOO#\\#OOOO#\\#OOOO#\\#OOOO#\\#OOOO#\\#OOOO#\\######"
                # "######\\######\\######\\######\\######\\######\\######\\#####O"
            )

        env = ant_draw.load_environment(
            name=f"{self.config['env']}_empty",
            map_key=empty_map_key,
            max_episode_steps=self.episode_length,
        )
        min_state, max_state, _ = self.estimate_state_ranges(env, num_samples=3000)

        self.min_state = min_state
        self.max_state = max_state
        print("State ranges:", min_state, max_state)

    def estimate_state_ranges(self, env, num_samples=1000):
        """
        Estimate the minimum and maximum x, y positions in obs coords
        """
        if "ant" not in env.name:
            vals = []
            for _i in tqdm(range(num_samples), desc="Estimating state ranges"):
                vals.append(env.reset()[:2])
            vals = np.array(vals)
            return (
                np.min(vals, axis=0),
                np.max(vals, axis=0),
                vals,
            )
        else:
            # resets in ant mazes are not random, so we need to sample from the
            # dataset
            ds = ant_draw.antmaze_get_dataset(
                env
            )  # use this to get dataset, which contains some fixes for poorly formatted d4rl data
            obses = ds["observations"][..., :2]
            return (
                np.min(obses, axis=0),
                np.max(obses, axis=0),
                obses,
            )

    def pick_random_start(self, env, range_min, range_max):
        if "ant" not in env.name:
            state = env.reset()

            if self.config["qvel_norm_prior"]:
                if self.config["qvel_norm_prior_type"] == "normal":
                    qvel = self.qvel_dist.rvs(size=2)
                elif self.config["qvel_norm_prior_type"] == "uniform":
                    qvel = sample_vector(max_norm=5)

                state[2:] = qvel

            return state

        state = np.zeros(29)

        while True:
            # generate random state in the range
            state[:2] = np.random.uniform(range_min, range_max, 2)
            if not env._is_in_collision(state[:2]):
                return state

    def bound_vector_norm(self, acc, max_norm=1.0):
        """
        Bound the norm of a 2D acceleration vector to a maximum of 1.

        Parameters:
        acc (numpy array): A 2D acceleration vector of shape (2,).

        Returns:
        numpy array: The bounded 2D acceleration vector.
        """
        norm = np.linalg.norm(acc)

        if norm > max_norm:
            acc = (acc / norm) * max_norm  # Scale to the maximum norm
        return acc

    def sample_angle_tapered_distribution(self, initial_angle=180, size=1):
        angles = sample_tapered_distribution(
            a=90, L=180, sigma=30, center=180, size=size
        )
        angles = np.clip(angles, 0, 360)

        # Step 1: Calculate the rotation amount
        rotation_amount = initial_angle - 180

        # Step 2: Apply the rotation to the third angle
        rotated_angle = angles + rotation_amount

        # Step 3: Wrap the result to ensure it's within [0, 360] degrees
        rotated_angle = rotated_angle % 360

        return rotated_angle

    def sample_goal_vector(self, initial_vector, norm_range=(0, 5)):
        """
        Samples a goal vector given an initial vector.

        Parameters:
        - initial_vector: tuple (x, y), the initial vector.
        - norm_range: tuple, the range from which to uniformly sample the norm of the goal vector.

        Returns:
        - goal_vector: tuple (x_goal, y_goal), the sampled goal vector.
        """

        # Step 1: Sample the norm of the goal vector according to truncated normal dist
        # goal_norm = np.random.uniform(*norm_range)
        iv_norm = np.linalg.norm(initial_vector)
        std = 3.25
        lower_bound = norm_range[0]
        upper_bound = norm_range[1]

        # Calculate the a and b parameters for the truncated normal distribution
        a = (lower_bound - iv_norm) / std
        b = (upper_bound - iv_norm) / std

        goal_norm = truncnorm.rvs(a, b, loc=iv_norm, scale=std, size=1)[0]

        # Clip the samples to ensure they fall within the desired range (0 to 5)
        goal_norm = np.clip(goal_norm, 0, 5)

        # Step 2: Calculate the goal angle
        x_init, y_init = initial_vector
        initial_angle_degrees = np.degrees(np.arctan2(y_init, x_init))
        goal_angle_degrees = self.sample_angle_tapered_distribution(
            initial_angle_degrees
        )[0]
        goal_angle = goal_angle_degrees * np.pi / 180

        # Step 4: Convert the polar coordinates (goal_norm, goal_angle) back to Cartesian coordinates
        x_goal = goal_norm * np.cos(goal_angle)
        y_goal = goal_norm * np.sin(goal_angle)

        return np.array([x_goal, y_goal])

    def generate_acceleration_sequence(
        self, v_init, v_goal, N, sigma=0.2, noise_level=0.05
    ):
        """
        Generates a sequence of accelerations to transition from v_init to v_goal over N timesteps.

        Parameters:
        - v_init: tuple (v_init_x, v_init_y), initial velocity
        - v_goal: tuple (v_goal_x, v_goal_y), goal velocity
        - N: int, number of timesteps
        - sigma: float, controls the width of the acceleration curve
        - noise_level: float, controls the level of noise added to the acceleration

        Returns:
        - a_sequence: array of shape (N, 2), the sequence of accelerations [a_x, a_y] over N timesteps
        """

        # Calculate the required change in velocity
        delta_v = np.array(v_goal) - np.array(v_init)

        # Create a base acceleration profile using a Gaussian curve
        t = np.arange(N)
        mu = (N - 1) / 2.0

        # Gaussian curve (bell curve)
        base_curve = np.exp(-0.5 * ((t - mu) / (sigma * N)) ** 2)

        # Normalize the base curve so that the integral equals delta_v
        base_curve_integral = np.sum(base_curve)
        scaling_factor = delta_v / (base_curve_integral + 1e-8)
        a_base = np.outer(base_curve, scaling_factor)

        # Introduce stochasticity
        noise = noise_level * np.random.randn(N, 2)
        a_sequence = a_base + noise

        # Ensure the initial and final accelerations are near zero
        # a_sequence[0] = a_sequence[-1] = np.array([0, 0])

        return a_sequence

    def generate_bi_acceleration_sequence(
        self,
        v_init,
        v_goal,
        lower_N,
        upper_N,
        sigma=0.2,
        noise_level=0.05,
        same_timesteps=False,
    ):
        acc_N = random.randint(lower_N, upper_N)
        if same_timesteps:
            decc_N = acc_N
        else:
            decc_N = random.randint(lower_N, upper_N)

        single_noise_level = noise_level / math.sqrt(2)
        acc_sequence = self.generate_acceleration_sequence(
            v_init, np.array([0, 0]), acc_N, sigma=sigma, noise_level=single_noise_level
        )
        decc_sequence = self.generate_acceleration_sequence(
            np.array([0, 0]),
            v_goal,
            decc_N,
            sigma=sigma,
            noise_level=single_noise_level,
        )

        max_len = max(len(acc_sequence), len(decc_sequence))

        # Pad the sequences with zeros to make them the same length
        acc_sequence = np.pad(
            acc_sequence,
            ((0, max_len - len(acc_sequence)), (0, 0)),
            mode="constant",
            constant_values=0,
        )
        decc_sequence = np.pad(
            decc_sequence,
            ((0, max_len - len(decc_sequence)), (0, 0)),
            mode="constant",
            constant_values=0,
        )

        # Return the sum of the two sequences
        return acc_sequence + decc_sequence

    def generate_ou_trajectory(self, env):
        obs = env._get_obs()
        trajectory = [obs]
        action_history = []

        params = self.config["ou_params"]
        theta = params["ou_theta"]
        dt = params["ou_dt"]
        sigma = params["ou_sigma"]

        # Initialize action with small random values
        action = np.random.uniform(-0.1, 0.1, size=2)
        mu = np.zeros_like(action)  # Mean for OU process

        step_counter = 0
        while step_counter < self.episode_length:
            if random.random() < 1 / params["turn_frequency"]:
                curr_obs = env._get_obs()
                v_curr = curr_obs[2:]
                v_goal = self.sample_goal_vector(v_curr)

                v_diff = v_goal - v_curr
                v_diff_norm = np.linalg.norm(v_diff)
                lower_N = int(v_diff_norm * 10 / 6)
                upper_N = lower_N + 10

                acc_seq = self.generate_bi_acceleration_sequence(
                    v_curr,
                    v_goal,
                    lower_N=lower_N,
                    upper_N=upper_N,
                    sigma=params["turn_sigma"],
                    noise_level=params["turn_noise_level"],
                )

                acc_seq = np.array([self.bound_vector_norm(a) for a in acc_seq])

                for i in range(acc_seq.shape[0]):
                    # update the state
                    action = acc_seq[i]
                    obs = env.step(action)[0]

                    action_history.append(action)
                    trajectory.append(obs)
                    step_counter += 1

                    if step_counter >= self.episode_length:
                        break
            else:
                # Ornstein-Uhlenbeck process for action noise
                dx = theta * (mu - action) * dt + sigma * np.sqrt(dt) * np.random.randn(
                    2
                )
                action = action.copy() + dx
                action = self.bound_vector_norm(action)

                # Update the state
                obs = env.step(action)[0]

                action_history.append(action)
                trajectory.append(obs)
                step_counter += 1

        return trajectory, action_history

    def generate_uniform_trajectory(self, env):
        obs = env._get_obs()
        trajectory = [obs]
        action_history = []

        for i in range(self.episode_length):
            if i % self.config["resample_every"] == 0:
                a = sample_vector(max_norm=1).astype(np.float32)
            obs = env.step(a)[0]

            action_history.append(a)
            trajectory.append(obs)

        return trajectory, action_history

    def gather_dataset(self, env):
        """
        Return:
        - all_obses_stacked: np.array of shape (n_episodes, episode_length, obs_dim)
        - all_actions_stacked: np.array of shape (n_episodes, episode_length - 1, action_dim)
        """

        all_obses = []
        all_actions = []

        for _ in tqdm(range(self.n_episodes), desc="Collecting episodes"):
            _ = env.reset()
            start_state = self.pick_random_start(
                env,
                self.min_state - self.margin,
                self.max_state + self.margin,
            )

            if "ant" in env.name:
                env.set_state(qpos=start_state[:15], qvel=start_state[15:])
            else:
                env.set_state(qpos=start_state[:2], qvel=start_state[2:])

            if self.config["sampling_mode"] == "uniform":
                obs_history, action_history = self.generate_uniform_trajectory(env=env)
            else:
                obs_history, action_history = self.generate_ou_trajectory(env=env)

            all_obses.append(np.stack(obs_history))
            all_actions.append(np.stack(action_history))

        all_obses_stacked = np.stack(all_obses)
        all_actions_stacked = np.stack(all_actions)

        return all_obses_stacked, all_actions_stacked

    def generate_data_for_env(self, env, map_idx=0):
        if self.config["action_repeat"] > 1:
            env = ActionRepeatWrapper(
                env,
                action_repeat=self.config["action_repeat"],
                action_repeat_mode=self.config["action_repeat_mode"],
            )

        layout_dir = f"{self.output_path}/layout_{map_idx}"
        os.makedirs(layout_dir, exist_ok=True)

        _, _, samples = self.estimate_state_ranges(env)

        plt.figure(dpi=200)
        plt.scatter(samples[:, 0], samples[:, 1], s=1)
        plt.grid()
        plt.title(f"Locations of reset states for {env.name}")
        plt.savefig(f"{layout_dir}/reset_states_locations.png")

        obses, actions = self.gather_dataset(env)
        all_obses_x = obses.reshape(-1, obses.shape[-1])

        plt.figure(dpi=200)
        plt.scatter(all_obses_x[:, 0], all_obses_x[:, 1], s=1)
        plt.xlim(0, 6.5)
        plt.ylim(0, 6.5)
        plt.grid()
        plt.title(f"Locations of all generated trajectories states for {env.name}")
        plt.savefig(f"{layout_dir}/all_states_locations.png")

        splits = [
            {"actions": a, "observations": o, "map_idx": map_idx}
            for a, o in zip(actions, obses)
        ]

        return splits

    def generate_data(self):
        cfg = self.config
        n_episodes = self.n_episodes
        episode_length = self.episode_length

        print(f"Collecting {n_episodes} episodes for length {episode_length}")

        if "diverse" in cfg["env"]:
            splits = []
            # we are creating many envs - each with unique layout
            for map_idx, map_key in self.diverse_maps.items():
                env = ant_draw.load_environment(
                    name=f"{cfg['env']}_{map_idx}",
                    map_key=map_key,
                    max_episode_steps=episode_length,
                )
                splits += self.generate_data_for_env(env, map_idx)
        else:
            env = ant_draw.load_environment(
                cfg["env"],
                max_episode_steps=episode_length,
            )
            splits = self.generate_data_for_env(env)

        output_file_name = f"{self.output_path}/data.p"
        print("Saving data to", output_file_name)
        torch.save(splits, output_file_name)

        output_metadata_name = f"{self.output_path}/metadata.pt"
        torch.save(self.config, output_metadata_name)


def main():
    parser = argparse.ArgumentParser(description="Generating Data for Mazes")
    parser.add_argument("--config", type=str)
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--map_path", type=str, default=None)
    parser.add_argument("--exclude_map_path", type=str, default=None)

    args = parser.parse_args()

    generator = UniformDataGenerator(
        config=args.config,
        output_path=args.output_path,
        map_path=args.map_path,
        exclude_map_path=args.exclude_map_path,
    )
    generator.generate_data()


if __name__ == "__main__":
    main()
