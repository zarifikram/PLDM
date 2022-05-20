from pldm.logger import Logger
from typing import List
from matplotlib import pyplot as plt
from pldm.logger import Logger
import math
from tqdm import tqdm
import numpy as np
import torch
import sys

default_plot_idxs = list(range(100))


def log_planning_plots(
    result,
    report,
    idxs: List[int] = default_plot_idxs,
    prefix: str = "wall_",
    n_steps: int = 44,
    xy_action: bool = True,
    plot_every: int = 1,
    quick_debug: bool = False,
    pixel_mapper=None,
    plot_failure_only: bool = False,
    log_pred_dist_every: int = sys.maxsize,
    mark_action: bool = True,
):
    num_plots = math.ceil(len(result.locations) / plot_every)
    grid_size = max(4, math.ceil(math.sqrt(num_plots)))

    img_size = result.observations[0][0].shape[-1]

    if pixel_mapper is not None:
        targets_pixels = torch.as_tensor(
            pixel_mapper.obs_coord_to_pixel_coord(result.targets)
        )

        locations_pixels = [
            torch.as_tensor(pixel_mapper.obs_coord_to_pixel_coord(x))
            for x in result.locations
        ]

        pred_locations_pixels = [
            torch.as_tensor(pixel_mapper.obs_coord_to_pixel_coord(x))
            for x in result.pred_locations
        ]
    else:
        targets_pixels = result.targets
        locations_pixels = result.locations
        pred_locations_pixels = result.pred_locations

    if idxs is None:
        idxs = default_plot_idxs
    for idx in idxs:
        if plot_failure_only and report.success[idx]:
            continue

        fig = plt.figure(dpi=300)
        start_location = locations_pixels[0][idx].cpu()
        subplot_idx = 0
        for i in range(len(locations_pixels)):
            if i % plot_every:
                continue

            if i > report.terminations[idx]:
                break
            plt.subplot(grid_size, grid_size, subplot_idx + 1)

            if "wall" in prefix:
                img = -1 * result.observations[i][idx].sum(dim=0).detach().cpu()
            elif result.observations[i][idx].shape[0] > 1:
                # if multiple channels, need to convert to grayscale
                img = result.observations[i][idx].detach().cpu().numpy()  # (3, 64, 64)
                img = np.transpose(img, (1, 2, 0))
                # Convert to grayscale using the weighted sum of RGB channels
                img = (
                    0.2989 * img[:, :, 0]
                    + 0.5870 * img[:, :, 1]
                    + 0.1140 * img[:, :, 2]
                )

                # Normalize the grayscale image to the range [0, 1]
                img = (img - img.min()) / (img.max() - img.min())
            else:
                img = result.observations[i][idx][0]

            plt.imshow(
                img,
                cmap="gray",
            )
            current_location = locations_pixels[i][idx].detach().cpu()
            if i != len(locations_pixels) - 1:
                # skip last one as there's no action at the last timestep
                action = result.action_history[i][idx, 0].detach()
                if not xy_action:
                    action = DotDataset.polar_to_xy(action)
                action = action * 5  # for visibility

                if mark_action:
                    plt.arrow(
                        x=current_location[0],
                        y=current_location[1],
                        dx=action[0].cpu(),
                        dy=action[1].cpu(),
                        width=0.05,
                        color="#F77F00",
                        head_width=2,
                    )

                if pred_locations_pixels is not None:
                    plt.plot(
                        pred_locations_pixels[i][:, idx, 0].detach().cpu(),
                        pred_locations_pixels[i][:, idx, 1].detach().cpu(),
                        marker="o",
                        markersize=0.1,
                        linewidth=0.1,
                        c="red",
                        alpha=1,
                    )

                    if log_pred_dist_every < 999999:
                        final_pred_dists = result.final_preds_dist[i][:, idx].tolist()
                        # skip every nth. make sure to include first and last
                        last_dist = final_pred_dists[-1]
                        final_pred_dists = final_pred_dists[::log_pred_dist_every]
                        if last_dist != final_pred_dists[-1]:
                            final_pred_dists.append(last_dist)

                        plt.text(
                            1.05,
                            0.5,
                            "\n".join([f"{dist:.2f}" for dist in final_pred_dists]),
                            transform=plt.gca().transAxes,
                            fontsize=2.5,
                            verticalalignment="center",
                            horizontalalignment="left",
                            color="blue",
                        )

            plt.scatter(
                start_location[0],
                start_location[1],
                s=0.1,
                c="blue",
                marker="o",
                alpha=1,
            )

            plt.scatter(
                targets_pixels[idx, 0].cpu(),
                targets_pixels[idx, 1].cpu(),
                s=0.1,
                c="#F77F00",
                marker="o",
                alpha=1,
            )
            plt.xlim(0, img_size - 1)
            plt.ylim(img_size - 1, 0)
            plt.gca().set_xticks([])
            plt.gca().set_yticks([])
            subplot_idx += 1

        # if quick_debug:
        #     breakpoint()
        # result.targets.shape = [15, 2]
        # result.locations[0].shape = [15, 2]

        log_name = f"mpc/{prefix}_{idx}"
        if plot_failure_only:
            start_x = result.locations[0][idx][0]
            start_y = result.locations[0][idx][1]
            target_x = result.targets[idx][0]
            target_y = result.targets[idx][1]
            log_name += (
                f"_{int(start_x)}_{int(start_y)}_{int(target_x)}_{int(target_y)}"
            )

        Logger.run().log_figure(fig, log_name)
        plt.close(fig)


def log_l1_planning_loss(result, prefix: str = "wall_"):
    logger = Logger.run()
    steps = [0, len(result.loss_history) // 2]

    for step in steps:
        losses = result.loss_history[step]
        for loss in losses:
            log_dict = {f"{prefix}_l1_step_{step}_plan_loss": loss}
            logger.log(log_dict)
        logger.log({f"{prefix}_l1_step_{step}_plan_iterations": len(losses)})
