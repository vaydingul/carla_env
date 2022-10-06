import numpy as np


def world_2_bev(
        loc,
        ego_loc,
        ego_yaw,
        image_height,
        image_width,
        pixels_per_meter=18):
    """Convert world coordinates to BEV coordinates"""
    # Calculate the displacement vector between ego location and world location
    displacement_meters = ego_loc - loc
    displacement_meters = displacement_meters[:-1]
    # Rotate the displacement vector by the ego yaw
    ego_yaw = np.deg2rad(-ego_yaw)
    rotated_displacement_meters = np.array(
        [[np.cos(ego_yaw), -np.sin(ego_yaw)], [np.sin(ego_yaw), np.cos(ego_yaw)]]) @ displacement_meters
    # Convert to pixels
    displacement_pixels = rotated_displacement_meters * pixels_per_meter
    # Change x and y axis
    displacement_pixels[0], displacement_pixels[1] = displacement_pixels[1], displacement_pixels[0]
    # Flip the y axis
    displacement_pixels[0] *= -1
    # Add the center of the image
    bev_loc = displacement_pixels + \
        np.array([image_width // 2, image_height // 2])

    return bev_loc
