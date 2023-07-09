TEST_EXPERIMENTS = [
    "test_dfm_km_cp",
    "test_mpc",
    "test_mpc_rgb2bev",
    "test_gradcem",
    "eval_mpc_leaderboard",
    "eval_gradcem_leaderboard",
]

TRAIN_EXPERIMENTS = [
    "train_ego_forward_model",
    "train_world_forward_model",
    "train_policy_model",
]

EVAL_EXPERIMENTS = [
    "eval_ego_forward_model",
    "eval_world_forward_model",
    "eval_world_forward_model_last_frame_repeated",
    "eval_policy_model",
]

DATA_EXPERIMENTS = []


def ego_forward_model_factory(config):
    if (
        (config["experiment_type"] == "train_ego_forward_model")
        or (config["experiment_type"] == "eval_ego_forward_model")
        or (config["experiment_type"] == "train_policy_model")
        or (config["experiment_type"] == "test_dfm_km_cp")
        or (config["experiment_type"] == "test_mpc")
        or (config["experiment_type"] == "test_mpc_rgb2bev")
        or (config["experiment_type"] == "test_gradcem")
    ):
        if config["ego_forward_model"]["type"] == "KinematicBicycleModel":
            from carla_env.models.dynamic.vehicle import KinematicBicycleModel

            return KinematicBicycleModel

        elif config["ego_forward_model"]["type"] == "DynamicBicycleModel":
            from carla_env.models.dynamic.vehicle import DynamicBicycleModel

            return DynamicBicycleModel

        else:
            raise ValueError("Invalid ego forward model type")

    else:
        raise ValueError("Invalid experiment type")


def world_forward_model_factory(config):
    if (
        (config["experiment_type"] == "train_world_forward_model")
        or (config["experiment_type"] == "eval_world_forward_model")
        or (config["experiment_type"] == "eval_world_forward_model_last_frame_repeated")
        or (config["experiment_type"] == "train_policy_model")
        or (config["experiment_type"] == "test_policy_model")
        or (config["experiment_type"] == "test_mpc")
        or (config["experiment_type"] == "test_mpc_rgb2bev")
        or (config["experiment_type"] == "test_gradcem")
    ):
        if config["world_forward_model"]["type"] == "WorldBEVModel":
            from carla_env.models.world.world import WorldBEVModel

            return WorldBEVModel

        elif config["world_forward_model"]["type"] == "WorldSVGLPModel":
            from carla_env.models.world.world_svg import WorldSVGLPModel

            return WorldSVGLPModel

        else:
            raise ValueError("Invalid world forward model type")

    else:
        raise ValueError("Invalid experiment type")


def policy_model_factory(config):
    if (config["experiment_type"] == "train_policy_model") or (
        config["experiment_type"] == "test_policy_model"
    ):
        if config["policy_model"]["type"] == "CILRSPolicyModel":
            from carla_env.models.policy.policy_cilrs import Policy

            return Policy

        elif config["policy_model"]["type"] == "FusedPolicyModel":
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
        or (config["experiment_type"] == "test_mpc_rgb2bev")
        or (config["experiment_type"] == "eval_mpc_leaderboard")
        or (config["experiment_type"] == "test_gradcem")
        or (config["experiment_type"] == "eval_gradcem_leaderboard")
    ):
        if config["cost"]["type"] == "extended_bev":
            from carla_env.cost.masked_cost_batched_extended_bev import Cost

            return Cost

        elif config["cost"]["type"] == "simple_bev":
            from carla_env.cost.masked_cost_batched_simple_bev import Cost

            return Cost

        elif config["cost"]["type"] == "extended_bev_with_pedestrian":
            from carla_env.cost.masked_cost_batched_extended_bev_with_pedestrian import (
                Cost,
            )

            return Cost

        elif config["cost"]["type"] == "simple_bev_with_pedestrian":
            from carla_env.cost.masked_cost_batched_simple_bev_with_pedestrian import (
                Cost,
            )

            return Cost

        elif config["cost"]["type"] == "special_bev_with_pedestrian":
            from carla_env.cost.masked_cost_batched_special_bev_with_pedestrian import (
                Cost,
            )

            return Cost
        else:
            raise ValueError("Invalid bev_type")

    else:
        raise ValueError("Invalid experiment type")


def trainer_factory(config):
    if config["experiment_type"] == "train_ego_forward_model":
        from carla_env.trainer.ego_forward_model import Trainer

        return Trainer

    elif config["experiment_type"] == "train_world_forward_model":
        if config["world_forward_model"]["type"] == "WorldBEVModel":
            from carla_env.trainer.world_forward_model import Trainer

            return Trainer

        else:
            from carla_env.trainer.world_forward_model_svg import Trainer

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
    elif config["experiment_type"] == "eval_world_forward_model_svg":
        from carla_env.evaluator.world_forward_model_svg import Evaluator

        return Evaluator

    elif config["experiment_type"] == "eval_world_forward_model_last_frame_repeated":
        from carla_env.evaluator.world_forward_model_last_frame_repeated import (
            Evaluator,
        )

        return Evaluator

    elif config["experiment_type"] == "eval_policy_model_leaderboard":
        from carla_env.evaluator.policy_model_leaderboard import Evaluator

        return Evaluator

    elif config["experiment_type"] == "eval_mpc_leaderboard":
        from carla_env.evaluator.mpc_leaderboard import Evaluator

        return Evaluator

    elif config["experiment_type"] == "eval_gradcem_leaderboard":
        from carla_env.evaluator.gradcem_leaderboard import Evaluator

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

    elif config["experiment_type"] == "test_mpc_rgb2bev":
        from carla_env.tester.mpc_rgb2bev import Tester

        return Tester

    elif config["experiment_type"] == "test_gradcem":
        from carla_env.tester.gradcem import Tester

        return Tester

    else:
        raise ValueError("Invalid experiment type")


def optimizer_factory(config):
    if (
        (config["experiment_type"] == "train_policy_model")
        or (config["experiment_type"] == "train_world_forward_model")
        or (config["experiment_type"] == "train_ego_forward_model")
        or (config["experiment_type"] == "test_mpc")
        or (config["experiment_type"] == "test_mpc_rgb2bev")
        or (config["experiment_type"] == "eval_mpc_leaderboard")
        or (config["experiment_type"] == "test_gradcem")
        or (config["experiment_type"] == "eval_gradcem_leaderboard")
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
    if (
        (config["experiment_type"] == "eval_world_forward_model")
        or (config["experiment_type"] == "eval_world_forward_model_last_frame_repeated")
        or (config["experiment_type"] == "eval_ego_forward_model")
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


def dataset_factory(config, class_name=None):
    if class_name is not None:
        if class_name == "InstanceDataset":
            from carla_env.dataset.instance import InstanceDataset

            return InstanceDataset

        elif class_name == "InstanceDatasetRAM":
            from carla_env.dataset.instance import InstanceDatasetRAM

            return InstanceDatasetRAM

        elif class_name == "TorchDataset":
            from carla_env.dataset.torch_dataset import TorchDataset

            return TorchDataset

        else:
            raise ValueError("Invalid class_name")

    else:
        if (
            (config["experiment_type"] == "train_ego_forward_model")
            or (config["experiment_type"] == "eval_ego_forward_model")
            or (config["experiment_type"] == "train_policy_model")
            or (config["experiment_type"] == "train_world_forward_model")
            or (config["experiment_type"] == "eval_world_forward_model")
            or (
                config["experiment_type"]
                == "eval_world_forward_model_last_frame_repeated"
            )
            or (config["experiment_type"] == "dataset_summary")
            or (config["experiment_type"] == "dataset_weight")
        ):
            dataset_key = [key for key in config.keys() if "dataset" in key][0]

            if config[dataset_key]["type"] == "InstanceDataset":
                from carla_env.dataset.instance import InstanceDataset

                return InstanceDataset

            elif config[dataset_key]["type"] == "InstanceDatasetRAM":
                from carla_env.dataset.instance import InstanceDatasetRAM

                return InstanceDatasetRAM

            elif config[dataset_key]["type"] == "TorchDataset":
                from carla_env.dataset.torch_dataset import TorchDataset

                return TorchDataset

            else:
                raise ValueError("Invalid dataset_type")

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

            elif key_["type"] == "radar":
                type_ = InstanceWriterType.RADAR

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

    elif (
        (config["experiment_type"] == "test_policy_model")
        or (config["experiment_type"] == "test_mpc")
        or (config["experiment_type"] == "test_mpc_rgb2bev")
        or (config["experiment_type"] == "test_gradcem")
    ):
        from carla_env.carla_env_testing_traffic import CarlaEnvironment

        return CarlaEnvironment

    elif config["experiment_type"] == "eval_policy_model_leaderboard":
        from carla_env.carla_env_leaderboard import CarlaEnvironment

        return CarlaEnvironment

    elif config["experiment_type"] == "play_carla":
        from carla_env.carla_env_playground import CarlaEnvironment

        return CarlaEnvironment

    elif (
        (config["experiment_type"] == "eval_mpc_leaderboard")
        or (config["experiment_type"] == "eval_policy_model_leaderboard")
        or (config["experiment_type"] == "eval_gradcem_leaderboard")
    ):
        from carla_env.carla_env_leaderboard import CarlaEnvironment

        return CarlaEnvironment

    else:
        raise ValueError("Invalid experiment type")


def sensor_factory(config):
    if (
        (config["experiment_type"] == "collect_data_random")
        or (config["experiment_type"] == "collect_data_driving")
        or (config["experiment_type"] == "test_mpc")
        or (config["experiment_type"] == "test_mpc_rgb2bev")
        or (config["experiment_type"] == "test_gradcem")
        or (config["experiment_type"] == "test_policy_model")
        or (config["experiment_type"] == "play_carla")
        or (config["experiment_type"] == "eval_policy_model_leaderboard")
        or (config["experiment_type"] == "eval_mpc_leaderboard")
        or (config["experiment_type"] == "eval_gradcem_leaderboard")
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

            elif sensor["type"] == "GNSSSensor":
                from carla_env.modules.sensor.gnss_sensor import GNSSSensorModule

                cls = GNSSSensorModule

            elif sensor["type"] == "IMUSensor":
                from carla_env.modules.sensor.imu_sensor import IMUSensorModule

                cls = IMUSensorModule

            elif sensor["type"] == "RadarSensor":
                from carla_env.modules.sensor.radar_sensor import RadarSensorModule

                cls = RadarSensorModule

            elif sensor["type"] == "SituationSensor":
                from carla_env.modules.sensor.situation_sensor import (
                    SituationSensorModule,
                )

                cls = SituationSensorModule

            else:
                raise ValueError("Invalid sensor_type")

            sensor_list.append(
                {"class": cls, "id": sensor["id"], "config": sensor["config"]}
            )

        return sensor_list

    else:
        raise ValueError("Invalid experiment type")


def noiser_factory(config):
    if (
        (config["experiment_type"] == "collect_data_random")
        or (config["experiment_type"] == "collect_data_driving")
        or (config["experiment_type"] == "test_mpc")
        or (config["experiment_type"] == "test_mpc_rgb2bev")
        or (config["experiment_type"] == "test_policy_model")
        or (config["experiment_type"] == "play_carla")
        or (config["experiment_type"] == "eval_policy_model_leaderboard")
    ):
        if "noiser" in config["environment"]:
            noiser = config["environment"]["noiser"]

            if noiser["type"] == "DummyNoiser":
                from carla_env.modules.noiser.dummy import DummyNoiser

                return {"class": DummyNoiser, "config": {}}

            else:
                if noiser["type"] == "GaussianNoiser":
                    from carla_env.modules.noiser.gaussian import GaussianNoiser

                    return {"class": GaussianNoiser, "config": noiser["config"]}

                else:
                    raise ValueError("Invalid noiser_type")

        else:
            from carla_env.modules.noiser.dummy import DummyNoiser

            return {"class": DummyNoiser, "config": {}}

    else:
        raise ValueError("Invalid experiment type")


def adapter_factory(config):
    if (
        (config["experiment_type"] == "collect_data_random")
        or (config["experiment_type"] == "collect_data_driving")
        or (config["experiment_type"] == "test_mpc")
        or (config["experiment_type"] == "test_mpc_rgb2bev")
        or (config["experiment_type"] == "test_policy_model")
        or (config["experiment_type"] == "play_carla")
        or (config["experiment_type"] == "eval_policy_model_leaderboard")
        or (config["experiment_type"] == "eval_mpc_leaderboard")
        or (config["experiment_type"] == "eval_gradcem_leaderboard")
    ):
        if "adapter" in config:
            if config["adapter"]["type"] == "roach":
                from externals.roach.roach_adapter import RoachAdapter

                return RoachAdapter

            else:
                raise ValueError("Invalid adapter_type")

        else:
            return None

    else:
        raise ValueError("Invalid experiment type")