from pathlib import Path
from PIL import Image

import torch
import numpy as np
import zarr
import torchvision.transforms as transforms

from pldm_envs.diverse_maze.enums import D4RLSample, D4RLDatasetConfig


def get_eval_env_name(env_name: str):
    split_name = env_name.split("-")
    if "ant" in env_name:
        split_name[-1] = "v0"
    split_name.insert(1, "eval")
    return "-".join(split_name)


class D4RLDataset(torch.utils.data.Dataset):
    def __init__(self, config: D4RLDatasetConfig, images_tensor=None):
        self.config = config

        if self.config.path is None:
            print("using d4rl dataset")
            self._prepare_ds()
        elif self.config.mixture_expert != 0:
            print("using mixture dataset")
            self._prepare_mixed_ds()
        else:
            self._prepare_saved_ds()

        if self.config.images_path is not None:
            print("states will contain images")
            if self.config.images_path.endswith("zarr"):
                print(f"loading zarr {self.config.images_path} into memory")
                if images_tensor is None:
                    self.images_tensor = zarr.open(self.config.images_path, "r")[:]
                else:
                    self.images_tensor = images_tensor
                print("using zarr format, shape of zarr is:", self.images_tensor.shape)
            elif self.config.images_path.endswith("npy"):
                print(f"loading {self.config.images_path}")
                if images_tensor is None:
                    self.images_tensor = np.load(self.config.images_path, mmap_mode="r")
                else:
                    self.images_tensor = images_tensor
                print("shape of images is:", self.images_tensor.shape)
            else:
                if "ant" not in self.config.env_name:
                    self.image_transform = transforms.Compose(
                        [
                            transforms.CenterCrop(200),
                            transforms.Resize(64),
                            transforms.ToTensor(),
                        ]
                    )
        else:
            print("states will contain proprioceptive info")

    def _prepare_mixed_ds(self):
        self._prepare_saved_ds()
        n_splits_to_keep = int((1 - self.config.mixture_expert) * len(self.splits))
        self.splits = self.splits[:n_splits_to_keep]

        self.env = data.ant_draw.load_environment(self.config.env_name)
        self.ds = data.ant_draw.antmaze_get_dataset(self.env)

        if "steps" not in self.ds:  # for umaze
            self.ds["steps"] = np.arange(len(self.ds["observations"]))

        idxs = (self.ds["steps"] == 0).nonzero()[0]
        lengths = idxs[1:] - idxs[:-1]
        lengths = np.append(lengths, len(self.ds["observations"]) - lengths.sum())

        splits = []
        for start, length in zip(idxs, lengths):
            new_split = {}
            for k in self.ds.keys():
                new_split[k] = self.ds[k][start : start + length]
            splits.append(new_split)

        if len(splits) == 1:
            n_expert_samples_to_keep = int(
                self.config.mixture_expert * len(splits[0]["actions"])
            )
            for k in splits[0].keys():
                splits[0][k] = splits[0][k][:n_expert_samples_to_keep]
            self.splits.append(splits[0])
        else:
            n_expert_splits_to_keep = int(self.config.mixture_expert * len(splits))
            self.splits = self.splits + splits[:n_expert_splits_to_keep]

        self.cum_lengths = np.cumsum(
            [
                len(x["observations"])
                - self.config.sample_length
                - (self.config.stack_states - 1)
                for x in self.splits
            ]
        )

    def _prepare_ds(self):
        self.env = data.ant_draw.load_environment(self.config.env_name)
        self.ds = data.ant_draw.antmaze_get_dataset(self.env)

        if "steps" not in self.ds:  # for umaze
            self.ds["steps"] = np.arange(len(self.ds["observations"]))

        idxs = (self.ds["steps"] == 0).nonzero()[0]
        lengths = idxs[1:] - idxs[:-1]
        lengths = np.append(lengths, len(self.ds["observations"]) - lengths.sum())

        splits = []
        for start, length in zip(idxs, lengths):
            new_split = {}
            for k in self.ds.keys():
                new_split[k] = self.ds[k][start : start + length]
            splits.append(new_split)

        self.splits = splits
        self.cum_lengths = np.cumsum(
            [
                len(x["observations"])
                - self.config.sample_length
                - (self.config.stack_states - 1)
                for x in splits
            ]
        )

    def _prepare_saved_ds(self):
        assert self.config.path is not None
        print("loading saved dataset from", self.config.path)
        self.splits = torch.load(self.config.path)
        self.cum_lengths = np.cumsum(
            [
                len(x["observations"])
                - self.config.sample_length
                - (self.config.stack_states - 1)
                for x in self.splits
            ]
        )

        self.cum_lengths_total = np.cumsum(
            [len(x["observations"]) for x in self.splits]
        )

    def __len__(self):
        if self.config.crop_length is not None:
            return min(self.config.crop_length, self.cum_lengths[-1])
        else:
            return self.cum_lengths[-1]

    def _load_images(self, episode_idx, start_idx, length):
        assert self.config.images_path is not None
        if self.config.images_path.endswith("npy") or self.config.images_path.endswith(
            "zarr"
        ):
            return self._load_images_tensor(episode_idx, start_idx, length)
        else:
            return self._load_images_files(episode_idx, start_idx, length)

    def _load_images_tensor(self, episode_idx, start_idx, length):
        if episode_idx == 0:
            index = start_idx
        else:
            index = self.cum_lengths_total[episode_idx - 1] + start_idx
        return (
            torch.from_numpy(self.images_tensor[index : index + length])
            .permute(0, 3, 1, 2)
            .float()
        )

    def _load_images_files(self, episode_idx, start_idx, length):
        images = []
        for i in range(start_idx, start_idx + length):
            image_path = Path(self.config.images_path) / f"{episode_idx}_{i}.png"
            image = self.image_transform(Image.open(image_path).convert("RGB"))
            images.append(image)
        return torch.stack(images, dim=0)

    def sample_location(self):
        idx = np.random.randint(0, len(self.splits))
        idx_2 = np.random.randint(0, len(self.splits[idx]["observations"]))
        return self.splits[idx]["observations"][idx_2]

    def __getitem__(self, idx):
        episode_idx = np.searchsorted(self.cum_lengths, idx, side="right")
        start_idx = idx - self.cum_lengths[episode_idx - 1] if episode_idx > 0 else idx
        end_idx = start_idx + self.config.sample_length + self.config.stack_states - 1

        # in maze2d case, its qvel
        propio_vel = torch.from_numpy(
            self.splits[episode_idx]["observations"][start_idx:end_idx, 2:]
        ).float()

        locations = self.splits[episode_idx]["observations"][start_idx:end_idx, :2]

        if self.config.images_path is not None:
            states = self._load_images(
                episode_idx,
                start_idx,
                self.config.sample_length + self.config.stack_states - 1,
            )
        else:
            if self.config.location_only:
                states = locations
            else:
                states = self.splits[episode_idx]["observations"][start_idx:end_idx]
            states = torch.from_numpy(states).float()

        actions = torch.from_numpy(
            self.splits[episode_idx]["actions"][start_idx : end_idx - 1]
        ).float()
        # to be compatible with other datasets with the dot.

        if self.config.stack_states > 1:
            states = torch.stack(
                [
                    states[i : i + self.config.stack_states]
                    for i in range(self.config.sample_length)
                ],
                dim=0,
            )
            states = states.flatten(1, 2)  # (sample_length, stack_states * state_dim)
            locations = locations[(self.config.stack_states - 1) :]
            actions = actions[(self.config.stack_states - 1) :]
            propio_vel = propio_vel[(self.config.stack_states - 1) :]

        if self.config.random_actions:
            # uniformly sample values from -1 to 1
            actions = torch.rand_like(actions) * 2 - 1

        locations = torch.from_numpy(locations).float()

        return D4RLSample(
            states=states,
            actions=actions,
            locations=locations,
            indices=idx,
            propio_vel=propio_vel,
            propio_pos=torch.empty(0),
        )
