def obs_to_state_target(obs):
    birdview = obs["birdview_mpc"]
    state_ = obs["state"]

    location = state_[..., 7:9]
    speed = state[..., 0:1]
    yaw = state[..., 12:13]

    state = dict(
        ego=dict(
            location=location,
            speed=speed,
            yaw=yaw,
        ),
        world=birdview,
    )
    
	target = 
