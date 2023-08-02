import torch


def obs_to_state_target(obs):
    birdview = obs["birdview_mpc"]
    state_ = obs["state"]

    location = state_[..., 11:13]
    speed = state_[..., 0:1]
    yaw = torch.deg2rad(state_[..., 16:17])

    target_location = state_[..., 17:19]
    target_speed = torch.zeros_like(speed)
    target_yaw = torch.deg2rad(state[..., 22:23])

    state = dict(
        ego=dict(
            location=location,
            speed=speed,
            yaw=yaw,
        ),
        world=birdview,
    )

    target = dict(
        ego=dict(
            location=target_location,
            speed=target_speed,
            yaw=target_yaw,
        ),
        world=None,
    )

    return state, target
