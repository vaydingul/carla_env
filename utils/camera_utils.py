import numpy as np


def world_2_pixel(world_point, world_2_camera, height, width):
    """
    Convert world coordinates to pixel coordinates.
    """

    FOV = 100
    WIDTH = width
    HEIGHTH = height
    FOCAL = WIDTH / (2.0 * np.tan(FOV * np.pi / 360.0))

    K = np.identity(3)
    K[0, 0] = FOCAL
    K[1, 1] = FOCAL
    K[0, 2] = WIDTH / 2.0
    K[1, 2] = HEIGHTH / 2.
	

    world_point_ = np.ones((4, ))
    world_point_[:3] = world_point

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
    point_in_camera_coords = np.array([
        sensor_points[1],
        sensor_points[2] * -1,
        sensor_points[0]])
    # print(point_in_camera_coords[2])
    point_in_camera_coords[0] -= 0
    point_in_camera_coords[1] -= 1.7
    #point_in_camera_coords[1] -= 1
    #point_in_camera_coords[2] += 5
    # Finally we can use our K matrix to do the actual 3D -> 2D.
    points_2d = K @ point_in_camera_coords

    # Remember to normalize the x, y values by the 3rd value.
    points_2d = np.array([
        points_2d[0] / points_2d[2],
        points_2d[1] / points_2d[2],
        points_2d[2]])

    # At this point, points_2d[0, :] contains all the x and points_2d[1, :]
    # contains all the y values of our points. In order to properly
    # visualize everything on a screen, the points that are out of the screen
    # must be discarded, the same with points behind the camera projection plane.
    points_2d = points_2d.T
    points_in_canvas_mask = \
        (points_2d[0] > 0.0) & (points_2d[0] < WIDTH) & \
        (points_2d[1] > 0.0) & (points_2d[1] < HEIGHTH) & \
        (points_2d[2] > 0.0)
    pixel_points = points_2d[points_in_canvas_mask]

    return pixel_points
