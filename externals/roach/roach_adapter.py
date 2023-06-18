from externals.adapter import Adapter
from externals.roach.carla_gym.utils import config_utils
from externals.roach.carla_gym.core.obs_manager.obs_manager_handler import (
    ObsManagerHandler,
)
from externals.roach.carla_gym.core.task_actor.common.criteria.run_stop_sign import (
    RunStopSign,
)
from externals.roach.carla_gym.utils.traffic_light import TrafficLightHandler
from utilities.config_utils import parse_yml

from omegaconf import DictConfig, OmegaConf


class RoachAdapter(Adapter):
    """
    A Roach adapter to use Roach's agent in this environment
    """

    def __init__(self, config):
        """
        Initialize the adapter
        :param config: the configuration of the adapter
        """
        super().__init__(config)

    def build_from_config(self):
        self.driver = self.config["actors"]["hero"]["driver"]
        self.driver_config = self.config["driver"][self.driver]

    def reset(
        self,
        vehicle,
        global_plan_world_coord,
        global_plan_world_coord_downsampled,
        global_plan_gps=None,
        global_plan_gps_downsampled=None,
    ):
        """
        Reset the adapter
        :return: the initial state
        """
        self.vehicle = vehicle

        agent_class = config_utils.load_entry_point(self.driver_config["entry_point"])
        OmegaConf.save(config=self.driver_config, f="config_agent.yaml")
        self.agent = agent_class("config_agent.yaml")

        world = self.vehicle.get_world()
        self.vehicle.criteria_stop = RunStopSign(world)

        # Set various plans
        self.vehicle.route_plan = global_plan_world_coord
        
        self.vehicle.global_plan_gps = global_plan_gps


        TrafficLightHandler.reset(world)

        self.obs_manager_handler = ObsManagerHandler({"hero": self.agent.obs_configs})

        self.obs_manager_handler.reset({"hero": self.vehicle})

    def step(self, route=None):
        """
        Make a step in the environment
        :param action: the action to be made
        :return: the next state, the reward, whether the episode is finished, and additional information
        """
        if route is not None:
            if len(route) > 80:
                route = route[:80]
            else:
                # Complete to 80 with repeating last element
                route = route + [route[-1]] * (80 - len(route))

            self.vehicle.route_plan = route

        return self.agent.run_step(
            self.obs_manager_handler.get_observation(None)["hero"], None
        )

    def get_state(self):
        """
        Get the current state
        :return: the current state
        """
        pass

    def get_action(self, state):
        """
        Get the action to be made
        :param state: the current state
        :return: the action to be made
        """
        pass

    def render(self):
        return self.agent.render()

    def set_default_config(self):
        self.config = {}


if __name__ == "__main__":
    adapter = RoachAdapter(
        parse_yml("configs/mpc_with_external_agent/roach/config.yml")["external_agent"]
    )
    adapter.reset()
