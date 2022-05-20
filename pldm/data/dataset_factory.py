import dataclasses

from pldm_envs.wall.data.offline_wall import OfflineWallDataset
from pldm_envs.wall.data.wall import WallDataset
from pldm_envs.wall.data.single import DotDataset
from pldm_envs.wall.data.wall_expert import (
    WrappedWallExpertDataset,
)
from pldm_envs.wall.data.wall_passing_test import WallPassingTestDataset
from pldm_envs.wall.data.border_passing_test import BorderPassingTestDataset

from pldm.data.utils import make_dataloader, make_dataloader_for_prebatched_ds


# if "AMD" not in torch.cuda.get_device_name(0):
from pldm_envs.diverse_maze.d4rl import D4RLDataset

from pldm.probing.evaluator import ProbingConfig
from pldm_envs.utils.normalizer import Normalizer
from pldm.data.enums import DataConfig, DatasetType, ProbingDatasets, Datasets


class DatasetFactory:
    def __init__(
        self,
        config: DataConfig,
        probing_cfg: ProbingConfig = ProbingConfig(),
        disable_l2: bool = True,
    ):
        self.config = config
        self.probing_cfg = probing_cfg
        self.disable_l2 = disable_l2

    def create_datasets(self):
        if self.config.dataset_type == DatasetType.Single:
            return self._create_single_datasets()
        elif self.config.dataset_type == DatasetType.Wall:
            return self._create_wall_datasets()
        elif self.config.dataset_type == DatasetType.WallExpert:
            return self._create_wall_expert_datasets()
        elif self.config.dataset_type == DatasetType.D4RL:
            return self._create_d4rl_datasets()
        elif self.config.dataset_type == DatasetType.LocoMaze:
            return self._create_locomaze_datasets()
        else:
            raise NotImplementedError

    def _create_single_datasets(self):
        ds = DotDataset(self.config.dot_config)
        val_ds = DotDataset(
            dataclasses.replace(self.config.dot_config, train=False),
            normalizer=ds.normalizer,
        )

        datasets = Datasets(ds=ds, val_ds=val_ds)

        return datasets

    def _create_wall_datasets(self):
        if self.config.offline_wall_config.use_offline:
            ds = OfflineWallDataset(config=self.config.offline_wall_config)
            ds = make_dataloader(
                ds=ds, loader_config=self.config, suffix="offline_wall"
            )
        else:
            ds = WallDataset(self.config.wall_config)
            ds = make_dataloader_for_prebatched_ds(
                probe_ds,
                loader_config=self.config,
            )

        probing_datasets = self._create_wall_probing_datasets(ds.normalizer)

        datasets = Datasets(
            ds=ds,
            val_ds=None,
            probing_datasets=probing_datasets,
        )

        return datasets

    def _create_wall_probing_datasets(self, normalizer: Normalizer):
        probe_ds = WallDataset(
            dataclasses.replace(
                self.config.wall_config,
                size=self.config.wall_config.val_size,
                train=False,
                n_steps=self.probing_cfg.l1_depth,
                fix_wall_batch_k=None,
                expert_cross_wall_rate=0,
            )
        )
        probe_ds = make_dataloader_for_prebatched_ds(
            probe_ds,
            loader_config=self.config,
            normalizer=normalizer,
        )

        probe_val_ds = WallDataset(
            dataclasses.replace(
                self.config.wall_config,
                size=self.config.wall_config.val_size,
                n_steps=self.probing_cfg.l1_depth,
                fix_wall_batch_k=None,
                train=False,
                expert_cross_wall_rate=0,
            )
        )
        probe_val_ds = make_dataloader_for_prebatched_ds(
            probe_val_ds,
            loader_config=self.config,
            normalizer=normalizer,
        )

        extra_datasets = {}

        if self.probing_cfg.probe_wall:
            wall_test_ds = WallPassingTestDataset(
                dataclasses.replace(
                    self.config.wall_config,
                    size=self.config.wall_config.val_size,
                    n_steps=self.probing_cfg.l1_depth,
                    fix_wall_batch_k=None,
                    train=False,
                )
            )
            extra_datasets["wall_test"] = make_dataloader_for_prebatched_ds(
                wall_test_ds, loader_config=self.config, normalizer=normalizer
            )
        if self.probing_cfg.probe_border:
            border_test_ds = BorderPassingTestDataset(
                dataclasses.replace(
                    self.config.wall_config,
                    size=self.config.wall_config.val_size,
                    n_steps=self.probing_cfg.l1_depth,
                    fix_wall_batch_k=None,
                    train=False,
                )
            )
            extra_datasets["border_test"] = make_dataloader_for_prebatched_ds(
                border_test_ds, loader_config=self.config, normalizer=normalizer
            )

        probing_datasets = ProbingDatasets(
            ds=probe_ds, val_ds=probe_val_ds, extra_datasets=extra_datasets
        )

        return probing_datasets

    def _create_wall_expert_datasets(self):
        ds = WallDataset(dataclasses.replace(self.config.wall_config, train=False))
        ds = WrappedWallExpertDataset(
            self.config.wall_expert_config, normalizer=ds.normalizer
        )
        val_ds = WrappedWallExpertDataset(
            dataclasses.replace(self.config.wall_expert_config, train=False),
            normalizer=ds.normalizer,
        )

        datasets = Datasets(
            ds=ds,
            val_ds=None,
        )

        return datasets

    def _create_d4rl_datasets(self):
        ds = D4RLDataset(self.config.d4rl_config)
        ds = make_dataloader(ds=ds, loader_config=self.config)

        probe_ds = D4RLDataset(
            dataclasses.replace(
                self.config.d4rl_config,
                path=self.probing_cfg.train_path,
                images_path=self.probing_cfg.train_images_path,
                sample_length=self.probing_cfg.l1_depth,
            ),
        )
        probe_ds = make_dataloader(
            ds=probe_ds,
            loader_config=self.config,
            normalizer=ds.normalizer,
            suffix="probe_train",
        )

        probe_val_ds = D4RLDataset(
            dataclasses.replace(
                self.config.d4rl_config,
                path=self.probing_cfg.val_path,
                images_path=self.probing_cfg.val_images_path,
                sample_length=self.probing_cfg.l1_depth,
                train=False,
                crop_length=50000,
                batch_size=64,
            ),
        )

        probe_val_ds = make_dataloader(
            ds=probe_val_ds,
            loader_config=self.config,
            normalizer=ds.normalizer,
            suffix="probe_val",
        )

        datasets = Datasets(
            ds=ds,
            val_ds=None,
            probing_datasets=ProbingDatasets(ds=probe_ds, val_ds=probe_val_ds),
        )

        return datasets
