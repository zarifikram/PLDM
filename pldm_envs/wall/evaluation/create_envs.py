import numpy as np
from pldm_envs.wall.wall import DotWall
from pldm_envs.wall.wrappers import NormEvalWrapper


def construct_eval_envs(
    seed,
    wall_config,
    n_envs: int,
    level: str,
    cross_wall: bool = True,
    start_ends=None,
    normalizer=None,
):
    if start_ends is None:
        rng = np.random.default_rng(seed)
        envs = [
            DotWall(
                rng=rng,
                border_wall_loc=wall_config.border_wall_loc,
                wall_width=wall_config.wall_width,
                door_space=wall_config.door_space,
                wall_padding=wall_config.wall_padding,
                img_size=wall_config.img_size,
                fix_wall=wall_config.fix_wall,
                cross_wall=cross_wall,
                level=level,
                n_steps=wall_config.n_steps,
                action_step_mean=wall_config.action_step_mean,
                max_step_norm=wall_config.action_upper_bd,
                fix_wall_location=wall_config.fix_wall_location,
                fix_door_location=wall_config.fix_door_location,
            )
            for _ in range(n_envs)
        ]

        [e.reset() for e in envs]

        if normalizer is not None:
            envs = [NormEvalWrapper(e, normalizer=normalizer) for e in envs]

        return envs
    else:
        raise NotImplementedError
