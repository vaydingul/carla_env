import wandb


def create_initial_run(config):
    run = wandb.init(
        id=wandb.util.generate_id(),
        project=config["wandb"]["project"],
        group=config["wandb"]["group"],
        name=config["wandb"]["name"],
        notes=config["wandb"]["notes"],
        resume="allow",
        config=config,
    )

    if config["wandb"]["id"] is None:
        run.config["wandb"]["id"] = run.id
        # run.update()

    run.define_metric("train/step")
    run.define_metric("val/step")
    run.define_metric("eval/step")
    run.define_metric(name="train/*", step_metric="train/step")
    run.define_metric(name="val/*", step_metric="val/step")
    run.define_metric(name="eval/*", step_metric="eval/step")

    return run


def create_resumed_run(config):

    run = wandb.init(
        project=config["wandb"]["project"],
        group=config["wandb"]["group"],
        id=config["wandb"]["id"],
        resume="allow",
    )

    return run


def create_wandb_run(config):
    # Setup the wandb
    if config["wandb"]["enable"]:

        if not config["wandb"]["resume"]:

            run = create_initial_run(config=config)

        else:

            run = create_resumed_run(config=config)

    else:

        run = None

    return run
