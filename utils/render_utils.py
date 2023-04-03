import cv2
import numpy as np
import torch

from carla_env.bev import BirdViewProducer


def postprocess_bev(bev, bev_selected_channels, bev_calculate_offroad):

    bev[bev > 0.5] = 1
    bev[bev <= 0.5] = 0
    bev = bev.clone().detach().cpu().numpy()
    bev = np.transpose(bev, (1, 2, 0))

    if bev_calculate_offroad:
        bev_selected_channels.append(11)

    bev = BirdViewProducer.as_rgb_with_indices(bev, bev_selected_channels)
    bev = cv2.cvtColor(bev, cv2.COLOR_BGR2RGB)

    return bev


def postprocess_mask(mask):

    mask = mask.clone().detach().cpu().numpy()
    mask = (((mask - mask.min()) / (mask.max() - mask.min())) * 255).astype(np.uint8)

    mask = cv2.applyColorMap(mask, cv2.COLORMAP_JET)

    return mask


def postprocess_action(action, val=50):

    action = action.clone().detach().cpu().numpy()
    action = action * val
    action = action.astype(np.int32)

    return action


def postprocess_location(location, ego_current_location=None):

    if isinstance(location, torch.Tensor):

        location_ = np.zeros((3,))

        location = location.clone().detach().cpu().numpy()

        location = np.reshape(location, (np.prod(location.shape),))

        location_[: location.shape[0]] = location

        if ego_current_location is not None:

            location_[-1] = ego_current_location.z

        location = location_

    else:

        location = np.array([location.x, location.y, location.z])

    return location


def world_2_pixel(world_point, world_2_camera, height, width, fov):
    """
    Convert world coordinates to pixel coordinates.
    """

    FOV = fov
    HEIGHTH = height
    WIDTH = width
    FOCAL = WIDTH / (2.0 * np.tan(FOV * np.pi / 360.0))

    K = np.identity(3)
    K[0, 0] = FOCAL
    K[1, 1] = FOCAL
    K[0, 2] = WIDTH / 2.0
    K[1, 2] = HEIGHTH / 2.0

    world_point_ = np.ones((4,))
    world_point_[: world_point.shape[0]] = world_point

    # Transform the points from world space to camera space.
    sensor_points = np.dot(world_2_camera, world_point_)

    # New we must change from UE4's coordinate system to an "standard"
    # camera coordinate system (the same used by OpenCV):

    # ^ z                       . z
    # |                        /
    # |              to:      +-------> x
    # | . x                   |
    # |/                      |
    # +-------> y             v y

    # This can be achieved by multiplying by the following matrix:
    # [[ 0,  1,  0 ],
    #  [ 0,  0, -1 ],
    #  [ 1,  0,  0 ]]

    # Or, in this case, is the same as swapping:
    # (x, y ,z) -> (y, -z, x)
    point_in_camera_coords = np.array(
        [sensor_points[1], sensor_points[2] * -1, sensor_points[0]]
    )
    # print(point_in_camera_coords[2])
    point_in_camera_coords[0] -= 0
    point_in_camera_coords[1] -= 1.7
    # point_in_camera_coords[1] -= 1
    # point_in_camera_coords[2] += 5
    # Finally we can use our K matrix to do the actual 3D -> 2D.
    points_2d = K @ point_in_camera_coords

    # Remember to normalize the x, y values by the 3rd value.
    points_2d = np.array(
        [points_2d[0] / points_2d[2], points_2d[1] / points_2d[2], points_2d[2]]
    )

    # At this point, points_2d[0, :] contains all the x and points_2d[1, :]
    # contains all the y values of our points. In order to properly
    # visualize everything on a screen, the points that are out of the screen
    # must be discarded, the same with points behind the camera projection
    # plane.
    points_2d = points_2d.T
    points_in_canvas_mask = (
        (points_2d[0] > 0.0)
        & (points_2d[0] < WIDTH)
        & (points_2d[1] > 0.0)
        & (points_2d[1] < HEIGHTH)
        & (points_2d[2] > 0.0)
    )
    pixel_points = points_2d[points_in_canvas_mask]

    if pixel_points.shape[0] == 0:
        return None
    # Convert to integer
    pixel_points = pixel_points.astype(np.int32)

    return pixel_points[0]


def world_2_bev(loc, ego_loc, ego_yaw, image_height, image_width, pixels_per_meter=20):
    """Convert world coordinates to BEV coordinates"""
    # Calculate the displacement vector between ego location and world location
    displacement_meters = ego_loc - loc
    displacement_meters = displacement_meters[:2]
    # Rotate the displacement vector by the ego yaw
    ego_yaw = np.deg2rad(-ego_yaw)
    rotated_displacement_meters = (
        np.array(
            [[np.cos(ego_yaw), -np.sin(ego_yaw)], [np.sin(ego_yaw), np.cos(ego_yaw)]]
        )
        @ displacement_meters
    )
    # Convert to pixels
    displacement_pixels = rotated_displacement_meters * pixels_per_meter
    # Change x and y axis
    displacement_pixels[0], displacement_pixels[1] = (
        displacement_pixels[1],
        displacement_pixels[0],
    )
    # Flip the y axis
    displacement_pixels[0] *= -1
    # Add to the top of the ego vehicle
    bev_loc = displacement_pixels + np.array([image_width // 2, image_height])

    # Check whether it is in the image
    if (
        bev_loc[0] < 0
        or bev_loc[0] >= image_width
        or bev_loc[1] < 0
        or bev_loc[1] >= image_height
    ):
        return None

    # Convert to integer
    bev_loc = bev_loc.astype(np.int32)

    return bev_loc
