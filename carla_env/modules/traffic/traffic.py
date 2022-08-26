from carla_env.modules import module
import carla


class TrafficModule(module.Module):
    """Concrete implementation of Module abstract base class for traffic management"""

    def __init__(self, client, config) -> None:
        super().__init__()
        for k in config.keys():
    self.config[k] = config[k]
        self.client = client
        # TODO: Figure out how to connect vehicles

    def start(self):
        """Start the traffic manager"""

        self.traffic_manager = self.client.get_traffic_manager(
            self.config["port"])
        self.traffic_manager.set_synchronous_mode(
            self.config["synchronous_mode"])
        # TODO: Set autopilot for all vehicles
        # TODO: Set minimum stop distance
        # TODO: Set desired speed
        # TODO: Set hybrid physics mode

    def step(self):
        """Step the traffic manager"""
        # TODO : Update

        pass

    def stop(self):
        """Stop the traffic manager"""
        pass

    def reset(self):
        """Reset the traffic manager"""
        pass

    def render(self):
        """Render the traffic manager"""
        pass

    def seed(self):
        """Seed the traffic manager"""
        pass

    def get_config(self):
        """Get the config of the traffic manager"""
        return self.config
