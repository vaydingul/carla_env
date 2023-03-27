from carla_env.modules.noiser.noiser import Noiser


class DummyNoiser(Noiser):
    def __init__(self, *args, **kwargs):
        pass

    def reset(self):
        pass

    def callback(self, timestamp):
        pass

    def close(self):
        pass
