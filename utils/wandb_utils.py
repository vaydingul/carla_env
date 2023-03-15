import wandb


class DummyConfig:
    """Dummy class for wandb when it is not enabled."""

    def __init__(self) -> None:
        pass

    def update(self, *args, **kwargs):
        pass


class DummyWandb:
    """Dummy class for wandb when it is not enabled."""

    def __init__(self, config=DummyConfig()) -> None:
        self.config = config
        pass

    def log(self, *args, **kwargs):
        pass

    def watch(self, *args, **kwargs):
        pass

    def finish(self, *args, **kwargs):
        pass

    def save(self, *args, **kwargs):
        pass


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
        config["wandb"]["id"] = run.id
        run.config.update(config, allow_val_change=True)

    run.define_metric("train/step")
    run.define_metric("val/step")
    run.define_metric("eval/step")
    run.define_metric(name="train/*", step_metric="train/step")
    run.define_metric(name="val/*", step_metric="val/step")
    run.define_metric(name="eval/*", step_metric="eval/step")

    return run


def create_resumed_run(config):

    if config["wandb"]["id"] is None:

        if config["wandb"]["link"] is not None:

            config["wandb"]["id"] = config["wandb"]["link"].split("/")[-1]

        else:

            raise ValueError(
                "wandb id or wandb link must be specified when resuming a run"
            )

    else:

        if config["wandb"]["link"] is None:

            config["wandb"][
                "link"
            ] = f"vaydingul/{config['wandb']['project']}/{config['wandb']['id']}"

        else:

            if config["wandb"]["link"].split("/")[-1] != config["wandb"]["id"]:

                raise ValueError(
                    "wandb id and wandb link do not match, please check the values"
                )

    run = wandb.init(
        project=config["wandb"]["project"],
        group=config["wandb"]["group"],
        id=config["wandb"]["id"],
        resume="allow",
    )

    run.config.update(config, allow_val_change=True)

    return run


def create_wandb_run(config=None, dummy=False):
    # Setup the wandb
    if config is not None:


        if config["wandb"]["enable"]:

            if not config["wandb"]["resume"]:

                run = create_initial_run(config=config)

            else:

                run = create_resumed_run(config=config)

            if dummy:

                run = DummyWandb(config=run.config)


    else:

        run = DummyWandb()

    return run
