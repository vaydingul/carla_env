def ego_forward_model_factory(config):

    if (
        (config["experiment_type"] == "train_ego_forward_model")
        or (config["experiment_type"] == "test_ego_forward_model")
        or (config["experiment_type"] == "train_policy_model")
        or (config["experiment_type"] == "test_dfm_km_cp")
        or (config["experiment_type"] == "test_mpc")
    ):

        if config["ego_forward_model"]["type"] == "KinematicBicycleModel":

            from carla_env.models.dynamic.vehicle import KinematicBicycleModel

            return KinematicBicycleModel

        else:

            raise ValueError("Invalid ego forward model type")

    else:

        raise ValueError("Invalid experiment type")


def world_forward_model_factory(config):

    if (
        (config["experiment_type"] == "train_world_forward_model")
        or (config["experiment_type"] == "test_world_forward_model")
        or (config["experiment_type"] == "train_policy_model")
        or (config["experiment_type"] == "test_policy_model")
        or (config["experiment_type"] == "test_mpc")
    ):
        if config["world_forward_model"]["type"] == "WorldBEVModel":

            from carla_env.models.world.world import WorldBEVModel

            return WorldBEVModel

        else:

            raise ValueError("Invalid world forward model type")

    else:

        raise ValueError("Invalid experiment type")


def policy_model_factory(config):

    if (config["experiment_type"] == "train_policy_model") or (
        config["experiment_type"] == "test_policy_model"
    ):

        if config["policy_model"]["type"] == "FusedPolicyModel":

            from carla_env.models.policy.policy_fused import Policy

            return Policy

        elif config["policy_model"]["type"] == "PolicyModel":

            from carla_env.models.policy.policy import Policy

            return Policy

        else:

            raise ValueError("Invalid policy model type")

    else:

        raise ValueError("Invalid experiment type")


def cost_factory(config):

    if (
        (config["experiment_type"] == "train_policy_model")
        or (config["experiment_type"] == "test_policy_model")
        or (config["experiment_type"] == "test_mpc")
    ):

        if config["cost"]["type"] == "extended_bev":

            from carla_env.cost.masked_cost_batched_extended_bev import Cost

            return Cost

        elif config["cost"]["type"] == "bev":

            from carla_env.cost.masked_cost_batched_bev import Cost

            return Cost

        else:

            raise ValueError("Invalid bev_type")

    else:

        raise ValueError("Invalid experiment type")


def model_factory(config):

    if (config["experiment_type"] == "train_policy_model") or (
        config["experiment_type"] == "test_dfm_km_cp"
    ):

        from carla_env.models.dfm_km_cp import (
            DecoupledForwardModelKinematicsCoupledPolicy,
        )

        return DecoupledForwardModelKinematicsCoupledPolicy

    else:

        raise ValueError("Invalid experiment type")


def trainer_factory(config):

    if config["experiment_type"] == "train_ego_forward_model":

        from carla_env.trainer.ego_forward_model import Trainer

        return Trainer

    elif config["experiment_type"] == "train_world_forward_model":

        from carla_env.trainer.world_forward_model import Trainer

        return Trainer

    elif config["experiment_type"] == "train_policy_model":

        from carla_env.trainer.policy_model import Trainer

        return Trainer

    else:

        raise ValueError("Invalid experiment type")


def evaluator_factory(config):

    if config["experiment_type"] == "eval_ego_forward_model":

        from carla_env.evaluator.ego_forward_model import Evaluator

        return Evaluator

    elif config["experiment_type"] == "eval_world_forward_model":

        from carla_env.evaluator.world_forward_model import Evaluator

        return Evaluator

    else:

        raise ValueError("Invalid experiment type")


def tester_factory(config):

    if config["experiment_type"] == "test_policy_model":

        from carla_env.tester.policy_model import Tester

        return Tester

    elif config["experiment_type"] == "test_mpc":

        from carla_env.tester.mpc import Tester

        return Tester

    else:

        raise ValueError("Invalid experiment type")


def optimizer_factory(config):

    if (
        (config["experiment_type"] == "train_policy_model")
        or (config["experiment_type"] == "train_world_forward_model")
        or (config["experiment_type"] == "train_ego_forward_model")
        or (config["experiment_type"] == "test_mpc")
    ):

        if config["training"]["optimizer"]["type"] == "Adam":

            from torch.optim import Adam

            return Adam

        elif config["training"]["optimizer"]["type"] == "SGD":

            from torch.optim import SGD

            return SGD

        elif config["training"]["optimizer"]["type"] == "RMSprop":

            from torch.optim import RMSprop

            return RMSprop

        elif config["training"]["optimizer"]["type"] == "AdamW":

            from torch.optim import AdamW

            return AdamW

        else:

            raise ValueError("Invalid optimizer_type")

    else:

        raise ValueError("Invalid experiment type")


def loss_criterion_factory(config):

    if (
        (config["experiment_type"] == "train_policy_model")
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

        raise ValueError("Invalid experiment type")


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

            def MAE(y_pred, y_true, kwargs):

                return torch.mean(torch.abs(y_pred - y_true))

            return MAE

        else:

            raise ValueError("Invalid metric")

    else:

        raise ValueError("Invalid experiment type")


def scheduler_factory(config):

    if (
        (config["experiment_type"] == "train_policy_model")
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

        raise ValueError("Invalid experiment type")


def dataset_factory(config):

    if (
        (config["experiment_type"] == "train_ego_forward_model")
        or (config["experiment_type"] == "test_ego_forward_model")
        or (config["experiment_type"] == "train_policy_model")
        or (config["experiment_type"] == "train_world_forward_model")
        or (config["experiment_type"] == "test_world_forward_model")
    ):

        from carla_env.dataset.instance import InstanceDataset

        return InstanceDataset

    else:

        raise ValueError("Invalid experiment type")


def writer_factory(config):

    if (config["experiment_type"] == "collect_data_random") or (
        config["experiment_type"] == "collect_data_driving"
    ):

        from carla_env.writer.writer import InstanceWriter

        return InstanceWriter

    else:

        raise ValueError("Invalid experiment type")


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

        return writer_type_list

    else:

        raise ValueError("Invalid experiment type")


def environment_factory(config):

    if (config["experiment_type"] == "collect_data_random") or (
        config["experiment_type"] == "collect_data_driving"
    ):

        from carla_env.carla_env_data_collect import CarlaEnvironment

        return CarlaEnvironment

    elif (config["experiment_type"] == "test_policy_model") or (
        config["experiment_type"] == "test_mpc"
    ):

        from carla_env.carla_env_testing_traffic import CarlaEnvironment

        return CarlaEnvironment

    else:

        raise ValueError("Invalid experiment type")


def sensor_factory(config):

    if (
        (config["experiment_type"] == "collect_data_random")
        or (config["experiment_type"] == "collect_data_driving")
        or (config["experiment_type"] == "test_mpc")
        or (config["experiment_type"] == "test_policy_model")
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

            sensor_list.append(
                {"class": cls, "id": sensor["id"], "config": sensor["config"]}
            )

        return sensor_list

    else:

        raise ValueError("Invalid experiment type")
