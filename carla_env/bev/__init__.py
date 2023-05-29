import carla
import logging
import numpy as np
import cv2

from enum import IntEnum, auto, Enum
from pathlib import Path
from typing import List
from filelock import FileLock

from . import actors, cache
from .actors import SegregatedActors
from .colors import RGB
from .mask import (
    PixelDimensions,
    Coord,
    CroppingRect,
    MapMaskGenerator,
    Mask,
    COLOR_ON,
    RenderingWindow,
    Dimensions,
)

LOGGER = logging.getLogger(__name__)


BirdView = np.ndarray  # [np.uint8] with shape (height, width, channel)
RgbCanvas = np.ndarray  # [np.uint8] with shape (height, width, 3)


class BirdViewCropType(Enum):
    FRONT_AND_REAR_AREA = auto()  # Freeway mode
    FRONT_AREA_ONLY = auto()  # Like in "Learning by Cheating"
    DYNAMIC = auto()


DEFAULT_HEIGHT = 336  # its 84m when density is 4px/m
DEFAULT_WIDTH = 150  # its 37.5m when density is 4px/m
DEFAULT_CROP_TYPE = BirdViewCropType.FRONT_AND_REAR_AREA


class BirdViewMasks(IntEnum):
    OFFROAD = 11
    PEDESTRIANS = 10
    RED_YELLOW_LIGHTS = 9
    GREEN_LIGHTS = 8
    AGENT = 7
    VEHICLES = 6
    LANES = 5
    ROAD_GREEN = 4
    ROAD_RED_YELLOW = 3
    ROAD_OFF = 2
    ROAD_ON = 1
    ROAD = 0

    @staticmethod
    def top_to_bottom() -> List[int]:
        return list(BirdViewMasks)

    @staticmethod
    def bottom_to_top() -> List[int]:
        return list(reversed(BirdViewMasks.top_to_bottom()))


RGB_BY_MASK = {
    BirdViewMasks.PEDESTRIANS: RGB.ORANGE,
    BirdViewMasks.LANES: RGB.WHITE,
    # BirdViewMasks.RED_LIGHTS: RGB.RED,
    # BirdViewMasks.YELLOW_LIGHTS: RGB.YELLOW,
    BirdViewMasks.RED_YELLOW_LIGHTS: RGB.DIM_RED,
    BirdViewMasks.GREEN_LIGHTS: RGB.GREEN,
    BirdViewMasks.AGENT: RGB.CHAMELEON,
    BirdViewMasks.VEHICLES: RGB.SKY_BLUE,
    BirdViewMasks.ROAD_GREEN: RGB.DIM_GREEN,
    BirdViewMasks.ROAD_RED_YELLOW: RGB.RED,
    # BirdViewMasks.ROAD_YELLOW: RGB.DIM_YELLOW,
    # BirdViewMasks.ROAD_RED: RGB.DIM_RED,
    BirdViewMasks.ROAD_ON: RGB.DIM_GRAY,
    BirdViewMasks.ROAD_OFF: RGB.DARK_GRAY,
    BirdViewMasks.ROAD: RGB.CHOCOLATE,
    BirdViewMasks.OFFROAD: RGB.DARK_VIOLET,
}

BIRDVIEW_CROP_TYPE = {
    "FRONT_AND_REAR_AREA": BirdViewCropType.FRONT_AND_REAR_AREA,
    "FRONT_AREA_ONLY": BirdViewCropType.FRONT_AREA_ONLY,
    "DYNAMIC": BirdViewCropType.DYNAMIC,
}


def rotate(image, angle, center=None, scale=1.0):
    assert image.dtype == np.uint8

    """Copy paste of imutils method but with INTER_NEAREST and BORDER_CONSTANT flags"""
    # grab the dimensions of the image
    (h, w) = image.shape[:2]

    # if the center is None, initialize it as the center of
    # the image
    if center is None:
        center = (w // 2, h // 2)

    # perform the rotation
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(
        image,
        M,
        (w, h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    # return the rotated image
    return rotated


def circle_circumscribed_around_rectangle(rect_size: Dimensions) -> float:
    """Returns radius of that circle."""
    a = rect_size.width / 2
    b = rect_size.height / 2
    return float(np.sqrt(np.power(a, 2) + np.power(b, 2)))


def square_fitting_rect_at_any_rotation(rect_size: Dimensions) -> float:
    """Preview: https://pasteboard.co/J1XK62H.png"""
    radius = circle_circumscribed_around_rectangle(rect_size)
    side_length_of_square_circumscribed_around_circle = radius * 2
    return side_length_of_square_circumscribed_around_circle


class BirdViewProducer:
    """Responsible for producing top-down view on the map, following agent's vehicle.

    About BirdView:
    - top-down view, fixed directly above the agent (including vehicle rotation), cropped to desired size
    - consists of stacked layers (masks), each filled with ones and zeros (depends on MaskMaskGenerator implementation).
        Example layers: road, vehicles, pedestrians. 0 indicates -> no presence in that pixel, 1 -> presence
    - convertible to RGB image
    - Rendering full road and lanes masks is computationally expensive, hence caching mechanism is used
    """

    def __init__(
        self,
        client: carla.Client,
        target_size: PixelDimensions,
        render_lanes_on_junctions: bool,
        pixels_per_meter: int = 4,
        crop_type: BirdViewCropType = BirdViewCropType.FRONT_AND_REAR_AREA,
        dynamic_crop_margin: float = 0.1,
        road_on_off: bool = False,
        road_light: bool = False,
        light_circle: bool = False,
        lane_marking_thickness=1,
    ) -> None:

        self.client = client
        self.target_size = target_size
        self.pixels_per_meter = pixels_per_meter
        self._crop_type = crop_type
        self.dynamic_crop_margin = dynamic_crop_margin
        self.road_on_off = road_on_off
        self.road_light = road_light
        self.light_circle = light_circle
        self.lane_marking_thickness = lane_marking_thickness
        self.is_first_frame = True
        if crop_type is BirdViewCropType.FRONT_AND_REAR_AREA:
            rendering_square_size = round(
                square_fitting_rect_at_any_rotation(self.target_size)
            )
        elif (crop_type is BirdViewCropType.FRONT_AREA_ONLY) or (
            crop_type is BirdViewCropType.DYNAMIC
        ):
            # We must keep rendering size from FRONT_AND_REAR_AREA (in order to
            # avoid rotation issues)
            enlarged_size = PixelDimensions(
                width=target_size.width, height=target_size.height * 2
            )
            rendering_square_size = round(
                square_fitting_rect_at_any_rotation(enlarged_size)
            )
        else:
            raise NotImplementedError
        self.rendering_area = PixelDimensions(
            width=rendering_square_size, height=rendering_square_size
        )
        self._world = client.get_world()
        self._map = self._world.get_map()
        self.masks_generator = MapMaskGenerator(
            client,
            pixels_per_meter=pixels_per_meter,
            render_lanes_on_junctions=render_lanes_on_junctions,
        )

        cache_path = self.parametrized_cache_path()
        with FileLock(f"{cache_path}.lock"):
            if Path(cache_path).is_file():
                LOGGER.info(f"Loading cache from {cache_path}")
                static_cache = np.load(cache_path)
                self.full_road_cache = static_cache[0]
                self.full_lanes_cache = static_cache[1]
                self.full_centerlines_cache = static_cache[2]
                LOGGER.info(f"Loaded static layers from cache file: {cache_path}")
            else:
                LOGGER.warning(
                    f"Cache file does not exist, generating cache at {cache_path}"
                )
                self.full_road_cache = self.masks_generator.road_mask()
                self.full_lanes_cache = self.masks_generator.lanes_mask(
                    self.lane_marking_thickness
                )
                self.full_centerlines_cache = self.masks_generator.centerlines_mask()
                static_cache = np.stack(
                    [
                        self.full_road_cache,
                        self.full_lanes_cache,
                        self.full_centerlines_cache,
                    ]
                )
                np.save(cache_path, static_cache, allow_pickle=False)
                LOGGER.info(f"Saved static layers to cache file: {cache_path}")

    def parametrized_cache_path(self) -> str:
        cache_dir = Path("cache")
        cache_dir.mkdir(parents=True, exist_ok=True)
        opendrive_content_hash = cache.generate_opendrive_content_hash(self._map)
        cache_filename = (
            f"{self._map.name}__"
            f"px_per_meter={self.pixels_per_meter}__"
            f"road_on_off={self.road_on_off}__"
            f"road_light={self.road_light}__"
            f"light_circle={self.light_circle}__"
            f"lane_marking_thickness={self.lane_marking_thickness}__"
            f"opendrive_hash={opendrive_content_hash}__"
            f"margin={mask.MAP_BOUNDARY_MARGIN}.npy"
        )
        return str(cache_dir / cache_filename)

    def step(self, agent_vehicle: carla.Actor, waypoint=None) -> BirdView:
        all_actors = actors.query_all(world=self._world)
        segregated_actors = actors.segregate_by_type(
            actors=all_actors, agent_vehicle=agent_vehicle
        )
        agent_vehicle_loc = agent_vehicle.get_location()
        agent_vehicle_loc_waypoint = self._map.get_waypoint(
            agent_vehicle_loc, project_to_road=True, lane_type=carla.LaneType.Driving
        )
        # Reusing already generated static masks for whole map
        self.masks_generator.disable_local_rendering_mode()

        agent_global_px_pos = self.masks_generator.location_to_pixel(agent_vehicle_loc)

        cropping_rect = CroppingRect(
            x=int(agent_global_px_pos.x - self.rendering_area.width / 2),
            y=int(agent_global_px_pos.y - self.rendering_area.height / 2),
            width=self.rendering_area.width,
            height=self.rendering_area.height,
        )

        # conservative_rect = CroppingRect(
        #     x=int(agent_global_px_pos.x - self.target_size.width / 2),
        #     y=int(agent_global_px_pos.y - self.target_size.height / 2),
        #     width=self.target_size.width,
        #     height=self.target_size.height,
        # )

        if self._crop_type is BirdViewCropType.DYNAMIC:

            if self.is_first_frame:
                self.is_first_frame = False
                self.cropping_rect = cropping_rect
                self.agent_vehicle_loc = agent_vehicle_loc
                self.agent_vehicle_angle = (
                    agent_vehicle.get_transform().rotation.yaw + 90
                )
                self.agent_global_px_pos = agent_global_px_pos
        else:

            self.cropping_rect = cropping_rect
            self.agent_vehicle_loc = agent_vehicle_loc

        masks = np.zeros(
            shape=(
                len(BirdViewMasks),
                self.rendering_area.height,
                self.rendering_area.width,
            ),
            dtype=np.uint8,
        )
        masks[BirdViewMasks.ROAD.value] = self.full_road_cache[
            self.cropping_rect.vslice, self.cropping_rect.hslice
        ]
        masks[BirdViewMasks.LANES.value] = self.full_lanes_cache[
            self.cropping_rect.vslice, self.cropping_rect.hslice
        ]
        # masks[BirdViewMasks.CENTERLINES.value] = self.full_centerlines_cache[
        #     cropping_rect.vslice, cropping_rect.hslice
        # ]

        # Dynamic masks
        rendering_window = RenderingWindow(
            origin=self.agent_vehicle_loc, area=self.rendering_area
        )

        self.masks_generator.enable_local_rendering_mode(rendering_window)

        if self.road_on_off:

            waypoint = waypoint if waypoint is not None else agent_vehicle_loc_waypoint

            road_on_mask = self.masks_generator.road_on_mask(waypoint)
            road_off_mask = self.masks_generator.road_off_mask(waypoint)
            masks[BirdViewMasks.ROAD_ON.value] = road_on_mask
            masks[BirdViewMasks.ROAD_OFF.value] = road_off_mask

        masks = self._render_actors_masks(agent_vehicle, segregated_actors, masks)

        (
            cropped_masks,
            reference_change,
        ) = self.apply_agent_following_transformation_to_masks(
            agent_vehicle,
            masks=masks,
            angle=self.agent_vehicle_angle
            if self._crop_type == BirdViewCropType.DYNAMIC
            else None,
        )

        if self._crop_type is BirdViewCropType.DYNAMIC:

            if reference_change:

                self.cropping_rect = cropping_rect
                self.agent_vehicle_loc = agent_vehicle_loc
                self.agent_vehicle_angle = (
                    agent_vehicle.get_transform().rotation.yaw + 90
                )
                self.agent_global_px_pos = agent_global_px_pos
        else:

            self.cropping_rect = cropping_rect
            self.agent_vehicle_loc = agent_vehicle_loc

        ordered_indices = [mask.value for mask in BirdViewMasks.bottom_to_top()]

        # Create offroad mask which is the where every channel is zero
        offroad_mask = np.where(cropped_masks.sum(axis=-1) == 0, 1, 0)
        cropped_masks[:, :, BirdViewMasks.OFFROAD.value] = offroad_mask

        return cropped_masks[:, :, ordered_indices]

    @staticmethod
    def as_rgb(birdview: BirdView) -> RgbCanvas:
        h, w, d = birdview.shape
        assert d == len(BirdViewMasks)
        rgb_canvas = np.zeros(shape=(h, w, 3), dtype=np.uint8)

        def nonzero_indices(arr):
            return arr == COLOR_ON

        for mask_type in BirdViewMasks.bottom_to_top():
            rgb_color = RGB_BY_MASK[mask_type]
            mask = birdview[:, :, mask_type]
            # If mask above contains 0, don't overwrite content of canvas (0
            # indicates transparency)
            rgb_canvas[nonzero_indices(mask)] = rgb_color
        return rgb_canvas

    @staticmethod
    def as_rgb_with_indices(birdview: BirdView, indices: list):
        h, w, d = birdview.shape
        assert d == len(indices)
        rgb_canvas = np.zeros(shape=(h, w, 3), dtype=np.uint8)

        def nonzero_indices(arr):
            return arr == COLOR_ON

        for (k, mask_type) in enumerate(indices):
            # print(BirdViewMasks.bottom_to_top()[mask_type])
            rgb_color = RGB_BY_MASK[BirdViewMasks.bottom_to_top()[mask_type]]
            # print(BirdViewMasks.bottom_to_top()[mask_type])
            mask = birdview[:, :, k]
            # If mask above contains 0, don't overwrite content of canvas (0
            # indicates transparency)
            rgb_canvas[nonzero_indices(mask)] = rgb_color
        return rgb_canvas

    def _render_actors_masks(
        self,
        agent_vehicle: carla.Actor,
        segregated_actors: SegregatedActors,
        masks: np.ndarray,
    ) -> np.ndarray:
        """Fill masks with ones and zeros (more precisely called as "bitmask").
        Although numpy dtype is still the same, additional semantic meaning is being added.
        """
        lights_masks = self.masks_generator.traffic_lights_masks(
            segregated_actors.traffic_lights
        )

        if self.light_circle:
            red_lights_mask, yellow_lights_mask, green_lights_mask = lights_masks
            masks[BirdViewMasks.RED_YELLOW_LIGHTS.value] = np.logical_or(
                red_lights_mask, yellow_lights_mask
            )
            # masks[BirdViewMasks.YELLOW_LIGHTS.value] = yellow_lights_mask
            masks[BirdViewMasks.GREEN_LIGHTS.value] = green_lights_mask

        if self.road_light:
            road_green_mask = self.masks_generator.road_light_mask(
                [
                    tl
                    for tl in segregated_actors.traffic_lights
                    if tl.state == carla.TrafficLightState.Green
                ]
            )
            road_yellow_mask = self.masks_generator.road_light_mask(
                [
                    tl
                    for tl in segregated_actors.traffic_lights
                    if tl.state == carla.TrafficLightState.Yellow
                ]
            )
            road_red_mask = self.masks_generator.road_light_mask(
                [
                    tl
                    for tl in segregated_actors.traffic_lights
                    if tl.state == carla.TrafficLightState.Red
                ]
            )
            masks[BirdViewMasks.ROAD_GREEN.value] = road_green_mask
            # masks[BirdViewMasks.ROAD_YELLOW.value] = road_yellow_mask
            masks[BirdViewMasks.ROAD_RED_YELLOW.value] = np.logical_or(
                road_red_mask, road_yellow_mask
            )

        masks[BirdViewMasks.AGENT.value] = self.masks_generator.agent_vehicle_mask(
            agent_vehicle
        )
        masks[BirdViewMasks.VEHICLES.value] = self.masks_generator.vehicles_mask(
            segregated_actors.vehicles
        )
        masks[BirdViewMasks.PEDESTRIANS.value] = self.masks_generator.pedestrians_mask(
            segregated_actors.pedestrians
        )

        return masks

    def apply_agent_following_transformation_to_masks(
        self, agent_vehicle: carla.Actor, masks: np.ndarray, angle=None
    ) -> np.ndarray:
        """Returns image of shape: height, width, channels"""
        agent_transform = agent_vehicle.get_transform()
        angle_ = (
            agent_transform.rotation.yaw + 90
        )  # vehicle's front will point to the top

        # Rotating around the center
        crop_with_car_in_the_center = masks
        masks_n, h, w = crop_with_car_in_the_center.shape
        rotation_center = Coord(x=w // 2, y=h // 2)

        # warpAffine from OpenCV requires the first two dimensions to be in
        # order: height, width, channels
        crop_with_centered_car = np.transpose(
            crop_with_car_in_the_center, axes=(1, 2, 0)
        )
        rotated = rotate(
            crop_with_centered_car,
            angle=angle if angle is not None else angle_,
            center=rotation_center,
        )

        half_width = self.target_size.width // 2
        hslice = slice(rotation_center.x - half_width, rotation_center.x + half_width)

        if (self._crop_type is BirdViewCropType.FRONT_AREA_ONLY) or (
            self._crop_type is BirdViewCropType.DYNAMIC
        ):
            vslice = slice(
                rotation_center.y - self.target_size.height, rotation_center.y
            )
        elif self._crop_type is BirdViewCropType.FRONT_AND_REAR_AREA:
            half_height = self.target_size.height // 2
            vslice = slice(
                rotation_center.y - half_height, rotation_center.y + half_height
            )
        else:
            raise NotImplementedError
        assert (
            vslice.start > 0 and hslice.start > 0
        ), "Trying to access negative indexes is not allowed, check for calculation errors!"
        car_on_the_bottom = rotated[vslice, hslice]

        reference_change = car_on_the_bottom[..., BirdViewMasks.AGENT.value].sum() == 0

        return car_on_the_bottom, reference_change
