import wandb


def create_initial_run(config):
    run = wandb.init(
        id=wandb.util.generate_id(),
        project=config.wandb_project,
        group=config.wandb_group,
        name=config.wandb_name,
        resume="allow",
        config=config)

    if config.wandb_id is None:
        run.config.update({"wandb_id": run.id}, allow_val_change=True)

    run.define_metric("train/step")
    run.define_metric("val/step")
    run.define_metric(name="train/*", step_metric="train/step")
    run.define_metric(name="val/*", step_metric="val/step")
    run.define_metric(name="eval/*", step_metric="val/step")

    return run


def create_resumed_run(config):

    run = wandb.init(
        project=config.wandb_project,
        group=config.wandb_group,
        id=config.wandb_id,
        resume="allow")
    return run


def create_wandb_run(config):
    # Setup the wandb
    if config.wandb:

        if not config.resume:

            run = create_initial_run(config=config)

        else:

            run = create_resumed_run(config=config)

    else:

        run = None

    return run
