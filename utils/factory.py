def ego_forward_model_factory(config):

    if (
        (config["experiment_type"] == "train_ego_forward_model")
        or (config["experiment_type"] == "test_ego_forward_model")
        or (config["experiment_type"] == "train_dfm_km_cp")
        or (config["experiment_type"] == "test_dfm_km_cp")
        or (config["experiment_type"] == "test_mpc")
    ):

        from carla_env.models.dynamic.vehicle import KinematicBicycleModel

        return KinematicBicycleModel

    else:

        return None


def world_forward_model_factory(config):

    if (
        (config["experiment_type"] == "train_world_forward_model")
        or (config["experiment_type"] == "test_world_forward_model")
        or (config["experiment_type"] == "train_dfm_km_cp")
        or (config["experiment_type"] == "test_poilcy")
        or (config["experiment_type"] == "test_mpc")
    ):

        from carla_env.models.world.world import WorldBEVModel

        return WorldBEVModel

    else:

        return None


def policy_model_factory(config):

    if (config["experiment_type"] == "train_dfm_km_cp") or (
        config["experiment_type"] == "test_dfm_km_cp"
    ):

        if config["fused"]:

            from carla_env.models.policy.policy_fused import Policy

            return Policy

        else:

            from carla_env.models.policy.policy import Policy

            return Policy

    else:

        return None


def cost_factory(config):

    if (
        (config["experiment_type"] == "train_dfm_km_cp")
        or (config["experiment_type"] == "test_dfm_km_cp")
        or (config["experiment_type"] == "test_mpc")
    ):

        if config["bev_type"] == "extended_bev":

            from carla_env.cost.masked_cost_batched_extended_bev import Cost

            return Cost

        elif config["bev_type"] == "bev":

            from carla_env.cost.masked_cost_batched_bev import Cost

            return Cost

        else:

            raise ValueError("Invalid bev_type")

    else:

        return None


def model_factory(config):

    if (config["experiment_type"] == "train_dfm_km_cp") or (
        config["experiment_type"] == "test_dfm_km_cp"
    ):

        from carla_env.models.dfm_km_cp import (
            DecoupledForwardModelKinematicsCoupledPolicy,
        )

        return DecoupledForwardModelKinematicsCoupledPolicy

    else:

        return None


def trainer_factory(config):

    if config["experiment_type"] == "train_ego_forward_model":

        from carla_env.trainer.ego_forward_model import Trainer

        return Trainer

    elif config["experiment_type"] == "train_world_forward_model":

        from carla_env.trainer.world_forward_model import Trainer

        return Trainer

    elif config["experiment_type"] == "train_dfm_km_cp":

        from carla_env.trainer.dfm_km_cp import Trainer

        return Trainer

    else:

        return None


def evaluator_factory(config):

    if config["experiment_type"] == "test_ego_forward_model":

        from carla_env.evaluator.ego_forward_model import Evaluator

        return Evaluator

    elif config["experiment_type"] == "test_world_forward_model":

        from carla_env.evaluator.world_forward_model import Evaluator

        return Evaluator

    else:

        return None


def optimizer_factory(config):

    if (
        (config["experiment_type"] == "train_dfm_km_cp")
        or (config["experiment_type"] == "train_world_forward_model")
        or (config["experiment_type"] == "train_ego_forward_model")
    ):

        if config["training"]["optimizer_type"] == "Adam":

            from torch.optim import Adam

            return Adam

        elif config["training"]["optimizer_type"] == "SGD":

            from torch.optim import SGD

            return SGD

        elif config["training"]["optimizer_type"] == "RMSprop":

            from torch.optim import RMSprop

            return RMSprop

        elif config["training"]["optimizer_type"] == "AdamW":

            from torch.optim import AdamW

            return AdamW

        else:

            raise ValueError("Invalid optimizer_type")

    else:

        return None


def loss_criterion_factory(config):

    if (
        (config["experiment_type"] == "train_dfm_km_cp")
        or (config["experiment_type"] == "train_world_forward_model")
        or (config["experiment_type"] == "train_ego_forward_model")
    ):

        if config["training"]["loss"]["criterion"] == "MSELoss":

            from torch.nn import MSELoss

            return MSELoss

        elif config["training"]["loss"]["criterion"] == "SmoothL1Loss":

            from torch.nn import SmoothL1Loss

            return SmoothL1Loss

        elif config["training"]["loss"]["criterion"] == "L1Loss":

            from torch.nn import L1Loss

            return L1Loss

        elif config["training"]["loss"]["criterion"] == "BCELoss":

            from torch.nn import BCELoss

            return BCELoss

        elif config["training"]["loss"]["criterion"] == "BCEWithLogitsLoss":

            from torch.nn import BCEWithLogitsLoss

            return BCEWithLogitsLoss

        else:

            raise ValueError("Invalid loss_criterion")

    else:

        return None


def metric_factory(config):

    if (config["experiment_type"] == "eval_world_forward_model") or (
        config["experiment_type"] == "eval_ego_forward_model"
    ):

        if config["evaluation"]["metric"]["type"] == "MSE":

            from torch.nn import MSELoss

            return MSELoss

        elif config["evaluation"]["metric"]["type"] == "SmoothL1Loss":

            from torch.nn import SmoothL1Loss

            return SmoothL1Loss

        elif config["evaluation"]["metric"]["type"] == "L1Loss":

            from torch.nn import L1Loss

            return L1Loss

        elif config["evaluation"]["metric"]["type"] == "BCELoss":

            from torch.nn import BCELoss

            return BCELoss

        elif config["evaluation"]["metric"]["type"] == "BCEWithLogitsLoss":

            from torch.nn import BCEWithLogitsLoss

            return BCEWithLogitsLoss

        elif config["evaluation"]["metric"]["type"] == "MAE":

            import torch

            def MAE(y_pred, y_true):

                return torch.mean(torch.abs(y_pred - y_true))

            return MAE

        else:

            raise ValueError("Invalid metric")

    else:

        return None


def scheduler_factory(config):

    if (
        (config["experiment_type"] == "train_dfm_km_cp")
        or (config["experiment_type"] == "train_world_forward_model")
        or (config["experiment_type"] == "train_ego_forward_model")
    ):

        if config["training"]["scheduler"]["enable"]:

            if config["training"]["scheduler"]["type"] == "StepLR":

                from torch.optim.lr_scheduler import StepLR

                return StepLR

            elif config["training"]["scheduler"]["type"] == "ReduceLROnPlateau":

                from torch.optim.lr_scheduler import ReduceLROnPlateau

                return ReduceLROnPlateau

            else:

                raise ValueError("Invalid scheduler_type")

        else:

            return None

    else:

        return None


def dataset_factory(config):

    if (
        (config["experiment_type"] == "train_ego_forward_model")
        or (config["experiment_type"] == "test_ego_forward_model")
        or (config["experiment_type"] == "train_dfm_km_cp")
        or (config["experiment_type"] == "train_world_forward_model")
        or (config["experiment_type"] == "test_world_forward_model")
    ):

        from carla_env.dataset.instance import InstanceDataset

        return InstanceDataset

    else:

        None


def writer_factory(config):

    if (config["experiment_type"] == "collect_data_random") or (
        config["experiment_type"] == "collect_data_driving"
    ):

        from carla_env.writer.writer import InstanceWriter

        return InstanceWriter

    else:

        None


def writer_key_factory(config):

    if (config["experiment_type"] == "collect_data_random") or (
        config["experiment_type"] == "collect_data_driving"
    ):
        writer_type_list = []

        from carla_env.writer.writer import InstanceWriterType

        for key_ in config["writer"]["keys"]:

            if key_["type"] == "rgb_image":

                type_ = InstanceWriterType.RGB_IMAGE

            elif key_["type"] == "bev_image":

                type_ = InstanceWriterType.BEV_IMAGE

            elif key_["type"] == "json":

                type_ = InstanceWriterType.JSON

            else:

                raise ValueError("Invalid writer_type")

            writer_type_list.append(
                {"key": key_["key"], "value": key_["value"], "type": type_}
            )

    else:

        None

    return writer_type_list


def environment_factory(config):

    if (config["experiment_type"] == "collect_data_random") or (
        config["experiment_type"] == "collect_data_driving"
    ):

        from carla_env.carla_env_data_collect import CarlaEnvironment

        return CarlaEnvironment

    elif config["experiment_type"] == "test_mpc":

        if config["bev_type"] == "extended_bev":

            from carla_env.carla_env_extended_bev_traffic import CarlaEnvironment

            return CarlaEnvironment

        elif config["bev_type"] == "bev":

            from carla_env.carla_env_bev_traffic import CarlaEnvironment

            return CarlaEnvironment

        else:

            raise ValueError("Invalid bev_type")

    else:

        return None


def sensor_factory(config):

    if (
        (config["experiment_type"] == "collect_data_random")
        or (config["experiment_type"] == "collect_data_driving")
        or (config["experiment_type"] == "test_mpc")
        or (config["experiment_type"] == "test_policy")
    ):

        sensors = config["environment"]["sensors"]
        sensor_list = []
        for sensor in sensors:

            if sensor["type"] == "RGBSensor":

                from carla_env.modules.sensor.rgb_sensor import RGBSensorModule

                cls = RGBSensorModule

            elif sensor["type"] == "CollisionSensor":

                from carla_env.modules.sensor.collision_sensor import (
                    CollisionSensorModule,
                )

                cls = CollisionSensorModule

            elif sensor["type"] == "VehicleSensor":

                from carla_env.modules.sensor.vehicle_sensor import VehicleSensorModule

                cls = VehicleSensorModule

            elif sensor["type"] == "SemanticSensor":

                from carla_env.modules.sensor.semantic_sensor import (
                    SemanticSensorModule,
                )

                cls = SemanticSensorModule

            elif sensor["type"] == "OccupancySensor":

                from carla_env.modules.sensor.occupancy_sensor import (
                    OccupancySensorModule,
                )

                cls = OccupancySensorModule

            else:

                raise ValueError("Invalid sensor_type")

            sensor_list.append({"class": cls, "args": sensor})

    return sensor_list
