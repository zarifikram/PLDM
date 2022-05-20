from typing import NamedTuple, List, Any, Optional, Dict
from dataclasses import dataclass
import itertools
import os

import torch
from tqdm.auto import tqdm
import numpy as np
import math

from pldm.models.misc import Prober
from pldm.configs import ConfigBase
from pldm.logger import Logger
import torch.nn.functional as F
from matplotlib import pyplot as plt
from pldm.data.enums import ProbingDatasets, DatasetType
from pldm.data.utils import get_optional_fields
from pldm.optimizers.schedulers import Scheduler, LRSchedule
import glob

from matplotlib import pyplot as plt
from pldm_envs.utils.normalizer import Normalizer
from pldm.models.jepa import JEPA
from pldm.models.hjepa import HJEPA


@dataclass
class ProbeTargetConfig(ConfigBase):
    arch: Optional[str] = None
    subclass: Optional[str] = None


@dataclass
class ProbingConfig(ConfigBase):
    probe_targets: str = "locations"
    l2_probe_targets: str = "locations"
    locations: ProbeTargetConfig = ProbeTargetConfig()
    propio_pos: ProbeTargetConfig = ProbeTargetConfig()
    propio_vel: ProbeTargetConfig = ProbeTargetConfig()
    full_finetune: bool = False
    lr: float = 1e-3
    epochs: int = 3
    epochs_enc: int = 3
    max_samples: Optional[int] = None
    max_samples_enc: Optional[int] = None
    schedule: LRSchedule = LRSchedule.Constant
    sample_timesteps: Optional[int] = None
    prober_arch: str = ""
    epochs_latent: int = 5
    l1_depth: int = 17
    l2_depth: int = 91
    probe_propio: bool = True
    probe_mpc: bool = False
    probe_wall: bool = True
    probe_border: bool = False
    probe_encoder: bool = True
    probe_preds: bool = True
    probe_expert: bool = False
    visualize_probing: bool = True
    load_prober: bool = False
    arch_subclass: str = "a"
    train_images_path: Optional[str] = None
    train_path: Optional[str] = None
    val_images_path: Optional[str] = None
    val_path: Optional[str] = None
    eval_contrastive: bool = False


class ProbeResult(NamedTuple):
    model: torch.nn.Module
    average_eval_loss: float
    eval_losses_per_step: List[float]
    plots: List[Any]


default_config = ProbingConfig()


def location_losses(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    assert pred.shape == target.shape
    # Pred and target are both B x T x N_DOTS x 2 or B x N_DOTS x 2.
    # we just avg the batch.
    # mse = (pred - target).pow(2).flatten(end_dim=-4).mean(dim=0)
    mse = (pred - target).pow(2).mean(dim=0)
    return mse


class ProbingEvaluator:
    def __init__(
        self,
        model: torch.nn.Module,
        probing_datasets: Optional[ProbingDatasets],
        l2_probing_datasets: Optional[ProbingDatasets],
        load_checkpoint_path: str = "",
        output_path: str = "",
        config: ProbingConfig = default_config,
        quick_debug: bool = False,
    ):
        self.config = config
        self.model = model
        self.quick_debug = quick_debug

        self.ds = probing_datasets.ds
        self.val_ds = probing_datasets.val_ds
        self.extra_val_ds = probing_datasets.extra_datasets

        self.load_checkpoint_path = load_checkpoint_path
        self.output_path = output_path

    def _context_manager(self):
        return torch.enable_grad() if self.config.full_finetune else torch.no_grad()

    def _infer_prober_path(self, probe_target, epoch, level):
        if self.load_checkpoint_path is not None and self.config.load_prober:
            root_path = "/".join(self.load_checkpoint_path.split("/")[:-1])
            prober_ckpt_paths = glob.glob(f"{root_path}/*{probe_target}*")
            assert len(prober_ckpt_paths) > 0
            # we get the most recent path (corresponding to latest epoch prober)
            prober_ckpt_path = max(prober_ckpt_paths, key=os.path.getctime)
            return prober_ckpt_path
        else:
            root_path = self.output_path

            prober_ckpt_path = (
                f"{self.output_path}/{level}_prober-{probe_target}_epoch={epoch}.pt"
            )
            return prober_ckpt_path

    def _infer_prober_input_dim_for_attr(
        self, probe_target, predictor, conv_input=False
    ):
        if predictor.pred_propio_dim == 0:
            repr_dim = predictor.repr_dim
        elif probe_target == "locations":
            repr_dim = predictor.pred_obs_dim
        elif probe_target == "propio_pos" or probe_target == "propio_vel":
            repr_dim = predictor.pred_propio_dim
        else:
            raise ValueError(f"Invalid probe target {probe_target}")

        if not conv_input and isinstance(repr_dim, tuple):
            return math.prod(repr_dim)

        return repr_dim

    def _get_pred_output_for_attr(self, pred_output, probe_target):
        if probe_target == "locations":
            if pred_output.obs_component is not None:
                return pred_output.obs_component
            else:
                return pred_output.predictions
        elif probe_target == "propio_pos" or probe_target == "propio_vel":
            if pred_output.propio_component is not None:
                return pred_output.propio_component
            else:
                return pred_output.predictions
        else:
            raise ValueError(f"Invalid probe target {probe_target}")

    def _get_enc_output_for_attr(self, enc_output, probe_target):
        if probe_target == "locations":
            if enc_output.obs_component is not None:
                return enc_output.obs_component
            else:
                return enc_output.encodings
        elif probe_target == "propio_pos" or probe_target == "propio_vel":
            if enc_output.propio_component is not None:
                return enc_output.propio_component
            else:
                return enc_output.encodings
        else:
            raise ValueError(f"Invalid probe target {probe_target}")

    def train_pred_prober(
        self,
        epoch: int,
        extra: Optional[Dict[str, Any]] = None,
    ):
        """
        Probes whether the predicted embeddings capture the future locations
        """

        level = "l1"

        plot_prefix = f"{level}_{epoch}"

        model = self.model.level1
        dataset = self.ds

        config = self.config
        epochs = config.epochs

        if self.quick_debug:
            epochs = 1
        test_batch = next(iter(dataset))

        probers = {}
        ckpt_paths = {}
        probe_targets = self.config.probe_targets.split(",")

        for probe_target in probe_targets:
            prober_output_shape = getattr(test_batch, probe_target)[0, 0].shape
            prober_output_shape = np.prod(prober_output_shape)

            probe_target_cfg = getattr(config, probe_target)

            prober_input_dim = self._infer_prober_input_dim_for_attr(
                probe_target=probe_target,
                predictor=model.predictor,
                conv_input=probe_target_cfg.arch == "conv",
            )

            prober = Prober(
                prober_input_dim,
                arch=probe_target_cfg.arch,
                output_shape=prober_output_shape,
                input_dim=prober_input_dim,  # to fix
                arch_subclass=probe_target_cfg.subclass,
            )

            probers[probe_target] = prober.cuda()

            # load prober logic
            ckpt_path = self._infer_prober_path(
                probe_target=probe_target, epoch=epoch, level=level
            )
            if config.load_prober:
                prober_ckpt = torch.load(ckpt_path)
                prober.load_state_dict(prober_ckpt["state_dict"])
                print(f"loaded {level} prober from {ckpt_path}")

            ckpt_paths[probe_target] = ckpt_path

        if config.load_prober:
            return probers

        all_parameters = []
        for probe_target, prober in probers.items():
            all_parameters += list(prober.parameters())

        if config.full_finetune:
            model.train()
            all_parameters += list(model.parameters())

        optimizer_pred_prober = torch.optim.Adam(all_parameters, config.lr)

        sample_step = 0
        step = 0

        batch_size = dataset.config.batch_size
        batch_steps = None

        if config.max_samples is not None:
            batch_steps = config.max_samples // dataset.config.batch_size
            n_epochs = max(1, batch_steps // len(dataset))
            if batch_steps < len(dataset):
                dataset = itertools.islice(dataset, batch_steps)
            epochs = n_epochs

        scheduler = Scheduler(
            schedule=self.config.schedule,
            base_lr=config.lr,
            data_loader=dataset,
            epochs=epochs,
            optimizer=optimizer_pred_prober,
            batch_steps=batch_steps,
            batch_size=batch_size,
        )

        for epoch in tqdm(range(epochs), desc=f"Probe {level} prediction epochs"):
            for batch in tqdm(dataset, desc="Probe prediction step"):
                # put time first
                states = batch.states.cuda().transpose(0, 1)
                actions = batch.actions.cuda().transpose(0, 1)
                optional_fields = get_optional_fields(batch, device=states.device)

                with self._context_manager():
                    forward_result = model.forward_posterior(
                        states, actions, **optional_fields
                    )

                pred_output = forward_result.pred_output

                losses_list = []

                for probe_target, prober in probers.items():
                    pred_encs = self._get_pred_output_for_attr(
                        pred_output,
                        probe_target,
                    )

                    n_steps = pred_encs.shape[0]
                    bs = pred_encs.shape[1]

                    if not config.full_finetune:
                        pred_encs = pred_encs.detach()

                    target = getattr(batch, probe_target).cuda()
                    target = target[:, :: model.subsampling_ratio()]

                    if (
                        config.sample_timesteps is not None
                        and config.sample_timesteps < n_steps
                    ):
                        sample_shape = (config.sample_timesteps,) + pred_encs.shape[1:]
                        # we only randomly sample n timesteps to train prober.
                        # we most likely do this to avoid OOM
                        sampled_pred_encs = torch.empty(
                            sample_shape,
                            dtype=pred_encs.dtype,
                            device=pred_encs.device,
                        )

                        sampled_target_locs = torch.empty(
                            bs, config.sample_timesteps, 1, 2
                        )

                        for i in range(bs):
                            indices = torch.randperm(n_steps)[: config.sample_timesteps]
                            sampled_pred_encs[:, i, :] = pred_encs[indices, i, :]
                            sampled_target_locs[i, :] = target[i, indices]

                        pred_encs = sampled_pred_encs
                        target = sampled_target_locs.cuda()

                    pred_locs = torch.stack([prober(x) for x in pred_encs], dim=1)

                    losses = location_losses(pred_locs, target)
                    per_probe_loss = losses.mean()

                    if self.quick_debug or step % 100 == 0:
                        log_dict = {
                            f"finetune_pred_{plot_prefix}_{probe_target}/loss": per_probe_loss.item(),
                        }
                        Logger.run().log(log_dict)

                    losses_list.append(per_probe_loss)

                optimizer_pred_prober.zero_grad()
                loss = sum(losses_list)
                loss.backward()
                optimizer_pred_prober.step()

                scheduler.adjust_learning_rate(step)

                step += 1
                sample_step += states.shape[0]

                if self.quick_debug:
                    break

        if config.full_finetune:  # we save the finetuned model as well
            model_ckpt_path = ckpt_path.replace("prober", "finetuned_model")
            torch.save({"model_state_dict": self.model.state_dict()}, model_ckpt_path)

        for probe_target, prober in probers.items():
            ckpt_path = ckpt_paths[probe_target]
            torch.save({"state_dict": prober.state_dict()}, ckpt_path)

        model.eval()

        return probers

    @torch.no_grad()
    def evaluate_all(
        self,
        probers,
        epoch,
        pixel_mapper=None,
        visualize=True,
    ):
        """
        Evaluates on all the different validation datasets
        """

        val_datasets = {"pred_probe": self.val_ds}
        val_datasets.update(self.extra_val_ds)

        for prefix, val_ds in val_datasets.items():
            self.evaluate_pred_prober(
                probers=probers,
                epoch=epoch,
                val_ds=val_ds,
                pixel_mapper=pixel_mapper,
                visualize=visualize,
            )

    @torch.no_grad()
    def evaluate_pred_prober(
        self,
        probers,
        epoch,
        val_ds: DatasetType,
        pixel_mapper=None,
        visualize=True,
    ):
        level = "l1"

        plot_prefix = f"{level}_{epoch}"

        model = self.model.level1

        quick_debug = self.quick_debug

        eval_repr_losses = []
        target_repr_losses = []

        probing_losses = {}
        for probe_target, prober in probers.items():
            prober.eval()
            probing_losses[probe_target] = []

        for idx, batch in enumerate(tqdm(val_ds, desc="Eval probe pred")):
            # put time first
            states = batch.states.cuda().transpose(0, 1)

            actions = batch.actions.cuda().transpose(0, 1)

            optional_fields = get_optional_fields(batch, device=states.device)

            forward_result = model.forward_posterior(states, actions, **optional_fields)

            pred_output = forward_result.pred_output
            enc_output = forward_result.backbone_output

            for probe_target, prober in probers.items():
                pred_encs = self._get_pred_output_for_attr(
                    pred_output,
                    probe_target,
                )

                encs = self._get_enc_output_for_attr(enc_output, probe_target)

                target = getattr(batch, probe_target).cuda()
                target = target[:, :: model.subsampling_ratio()]

                pred_locs = torch.stack([prober(x) for x in pred_encs], dim=1)

                losses = location_losses(pred_locs, target)
                probing_losses[probe_target].append(losses.cpu())

            repr_loss = F.mse_loss(encs, pred_encs, reduction="none")
            reduce_dims = tuple(range(1, encs.ndim))
            repr_loss = repr_loss.mean(dim=reduce_dims)
            eval_repr_losses.append(repr_loss.cpu())

            target_encs = encs[-1]
            target_encs = target_encs.unsqueeze(0).expand(
                encs.shape[0], *[-1] * len(target_encs.shape)
            )

            # permutation = torch.randperm(64)
            # target_encs = target_encs[:, permutation, :, :, :]
            target_repr_loss = F.mse_loss(target_encs, pred_encs, reduction="none")
            target_repr_loss = target_repr_loss.mean(dim=reduce_dims)
            target_repr_losses.append(target_repr_loss.cpu())

            if quick_debug and idx > 2:
                break

        repr_loss = torch.stack(eval_repr_losses).mean(dim=0)
        target_repr_losses = torch.stack(target_repr_losses).mean(dim=0)

        # Plot repr loss over timesteps
        Logger.run().log_line_plot(
            data=[[i, x.item()] for i, x in enumerate(repr_loss)],
            plot_name=f"finetune_pred_val_{plot_prefix}_repr_loss",
        )

        # Plot target repr loss over timesteps
        Logger.run().log_line_plot(
            data=[[i, x.item()] for i, x in enumerate(target_repr_losses)],
            plot_name=f"finetune_pred_val_{plot_prefix}_target_repr_loss",
        )

        log_dict = {}

        for probe_target, eval_losses in probing_losses.items():
            losses_t = torch.stack(eval_losses, dim=0).mean(dim=0)
            losses_t = val_ds.normalizer.unnormalize_mse(losses_t, probe_target)
            losses_t = losses_t.mean(dim=-1)
            average_eval_loss = losses_t.mean().item()
            log_dict[f"finetune_pred_val_{plot_prefix}_{probe_target}/loss_avg"] = (
                average_eval_loss
            )
            log_dict[
                f"finetune_pred_val_{plot_prefix}_{probe_target}/loss_rmse_avg"
            ] = np.sqrt(average_eval_loss)

            # Plot probbing loss over timesteps
            Logger.run().log_line_plot(
                data=[[i, x.item()] for i, x in enumerate(losses_t)],
                plot_name=f"finetune_pred_val_{plot_prefix}_{probe_target}_loss",
            )

        Logger.run().log(log_dict)

        # right now, we only visualize location predictions
        if self.config.visualize_probing and visualize:
            self.plot_prober_predictions(
                next(iter(val_ds)),
                model,
                probers["locations"],
                normalizer=val_ds.normalizer,
                name_prefix=plot_prefix,
                idxs=None if not quick_debug else list(range(10)),
                pixel_mapper=pixel_mapper,
            )

        return

    def train_encoder_prober(self, epoch: int):
        """
        Train a prober to probe whether the encoded embeddings captures the true location
        """
        plot_prefix = str(epoch)

        jepa = self.model.level1
        repr_dim = jepa.repr_dim
        dataset = self.ds
        quick_debug = self.quick_debug
        config = self.config

        test_batch = next(iter(dataset))

        probers = {}
        probe_targets = self.config.probe_targets.split(",")
        for probe_target in probe_targets:
            prober_output_shape = getattr(test_batch, probe_target)[0, 0].shape
            prober_output_shape = np.prod(prober_output_shape)

            probe_target_cfg = getattr(config, probe_target)

            prober_input_dim = self._infer_prober_input_dim_for_attr(
                probe_target=probe_target,
                predictor=jepa.predictor,
                conv_input=probe_target_cfg.arch == "conv",
            )

            prober = Prober(
                repr_dim,
                arch=probe_target_cfg.arch,
                output_shape=prober_output_shape,
                input_dim=jepa.spatial_repr_dim,
                arch_subclass=probe_target_cfg.subclass,
            )
            probers[probe_target] = prober.cuda()

        all_parameters = []
        for probe_target, prober in probers.items():
            all_parameters += list(prober.parameters())

        if config.full_finetune:
            jepa.train()
            all_parameters += list(jepa.backbone.parameters())

        optimizer = torch.optim.Adam(all_parameters, config.lr)

        if quick_debug:
            config.epochs_enc = 1

        batch_size = dataset.config.batch_size
        batch_steps = None
        if config.max_samples_enc is not None:
            batch_steps = config.max_samples_enc // dataset.config.batch_size
            n_epochs = max(1, batch_steps // len(dataset))
            if batch_steps < len(dataset):
                dataset.dataset.config.crop_length = config.max_samples
            config.epochs_enc = n_epochs

        scheduler = Scheduler(
            schedule=self.config.schedule,
            base_lr=config.lr,
            data_loader=dataset,
            epochs=config.epochs_enc,
            optimizer=optimizer,
            batch_steps=batch_steps,
            batch_size=batch_size,
        )

        step = 0

        for epoch in tqdm(range(config.epochs_enc), desc="Eval enc"):
            for batch in dataset:
                states = batch.states.cuda().transpose(0, 1)
                actions = batch.actions.cuda().transpose(0, 1)

                optional_fields = get_optional_fields(batch, device=states.device)

                with self._context_manager():
                    forward_result = jepa.forward_posterior(
                        states, actions, encode_only=True, **optional_fields
                    )

                e = forward_result.backbone_output.encodings[0]

                losses_list = []
                for probe_target, prober in probers.items():
                    target = getattr(batch, probe_target)[:, 0].cuda().float()

                    pred = prober(e)

                    loss = location_losses(pred, target)
                    losses_list.append(loss.mean())

                    if quick_debug or step % 100 == 0:
                        log_dict = {
                            f"finetune_enc_{plot_prefix}_{probe_target}/loss": loss.mean().item(),
                        }
                        Logger.run().log(log_dict)

                optimizer.zero_grad()
                total_loss = sum(losses_list)
                total_loss.backward()
                optimizer.step()

                scheduler.adjust_learning_rate(step)

                step += 1
                if quick_debug:
                    break

            if quick_debug:
                break

        jepa.eval()

        return probers

    @torch.no_grad()
    def eval_probe_enc_position(
        self,
        probers,
        epoch: int,
    ):
        plot_prefix = str(epoch)

        jepa = self.model.level1
        val_dataset = self.val_ds
        quick_debug = self.quick_debug

        probing_losses = {}
        for probe_target, prober in probers.items():
            prober.eval()
            probing_losses[probe_target] = []

        for idx, batch in enumerate(val_dataset):
            states = batch.states.cuda().transpose(0, 1)
            actions = batch.actions.cuda().transpose(0, 1)

            optional_fields = get_optional_fields(batch, device=states.device)

            forward_result = jepa.forward_posterior(
                states, actions, encode_only=True, **optional_fields
            )

            e = forward_result.backbone_output.encodings[0]

            for probe_target, prober in probers.items():
                target = getattr(batch, probe_target)[:, 0].cuda().float()
                pred = prober(e)

                losses = location_losses(pred, target)

                probing_losses[probe_target].append(losses.cpu())

            if idx > 2 and quick_debug:
                break

        log_dict = {}
        for probe_target, eval_losses in probing_losses.items():
            avg_loss = torch.stack(eval_losses, dim=0).mean(dim=0)
            unnormalized_avg_loss = (
                val_dataset.normalizer.unnormalize_mse(avg_loss, probe_target)
                .mean()
                .cpu()
            )
            log_dict = {
                f"avg_eval_enc_{plot_prefix}_{probe_target}_loss": unnormalized_avg_loss,
                f"avg_eval_enc_{plot_prefix}_{probe_target}_loss_rmse": np.sqrt(
                    unnormalized_avg_loss
                ),
            }
        Logger.run().log(log_dict)

        return unnormalized_avg_loss

    @torch.no_grad()
    def plot_prober_predictions(
        self,
        batch,
        jepa: JEPA,
        prober: torch.nn.Module,
        normalizer: Normalizer,
        name_prefix: str = "",
        idxs: Optional[List[int]] = None,
        notebook: bool = False,
        pixel_mapper=None,
    ):
        # infer
        states = batch.states.cuda().transpose(0, 1)

        actions = batch.actions.cuda().transpose(0, 1)

        optional_fields = get_optional_fields(batch, device=states.device)

        pred_output = jepa.forward_posterior(
            states, actions, **optional_fields
        ).pred_output

        if pred_output.obs_component is not None:
            pred_encs = pred_output.obs_component
        else:
            pred_encs = pred_output.predictions

        pred_locs = torch.stack([prober(x) for x in pred_encs], dim=1)

        # pred_locs is of shape (batch_size, time, 1, 2)
        if idxs is None:
            idxs = list(range(min(pred_locs.shape[0], 64)))

        gt_locations = normalizer.unnormalize_location(batch.locations).cpu()
        pred_locs = normalizer.unnormalize_location(pred_locs).cpu()

        if pixel_mapper is not None:
            gt_locations = pixel_mapper(gt_locations)
            pred_locs = pixel_mapper(pred_locs)

        # plot
        for i in tqdm(idxs, desc=f"Plotting {name_prefix}"):  # batch size
            fig = plt.figure(dpi=200)
            if hasattr(batch, "view_states") and batch.view_states.shape[-1]:
                images = batch.view_states
            else:
                images = batch.states

            x_max = images.shape[-2] - 1
            y_max = images.shape[-1] - 1

            plt.imshow(-1 * images[i, 0].sum(dim=0).cpu(), cmap="gray")

            plt.plot(
                gt_locations[i, :, 0].cpu(),
                gt_locations[i, :, 1].cpu(),
                marker="o",
                markersize=2.5,
                linewidth=1,
                c="#3777FF",
                alpha=0.8,
            )
            plt.plot(
                pred_locs[i, :, 0].cpu(),
                pred_locs[i, :, 1].cpu(),
                marker="o",
                markersize=2.5,
                linewidth=1,
                c="#D62828",
                alpha=0.8,
            )
            plt.xlim(0, x_max)
            plt.ylim(y_max, 0)

            if not notebook:
                Logger.run().log_figure(fig, f"{name_prefix}/prober_predictions_{i}")
                plt.close(fig)
