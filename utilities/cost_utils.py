import torch
from utilities.config_utils import parse_yml


def sample_weight(mu, std):
    """Sample a weight from a normal distribution"""
    return torch.normal(mu, std, ())


def transfer_cost_weight(cost_weight):
    """
    If cost_weight is a path to another yml file, read the cost_weight in that yml file.
    Otherwise, return the cost_weight as is. However, check the structure of the cost weight
    """

    cost_weight_new = {}
    if isinstance(cost_weight, str):
        cost_weight_new = parse_yml(cost_weight)
        if "cost_weight" in cost_weight_new:
            cost_weight_new = cost_weight_new["cost_weight"]

    elif isinstance(cost_weight, dict):
        for k, v in cost_weight.items():
            cost_weight_new[k] = transfer_cost_weight(v)

    else:
        cost_weight_new = cost_weight

    return cost_weight_new


def create_coordinate_mask(nx, ny, pixels_per_meter, device):
    """Create a matrix such that the value at each index is the distance from the center of the image"""

    meters_per_pixel = 1 / pixels_per_meter

    # Create a matrix of the same size as the image
    X, Y = torch.meshgrid(
        torch.arange(nx, device=device), torch.arange(ny, device=device)
    )

    # Calculate the distance from the center of the image
    x = X - nx // 2
    y = Y  # - ny // 2
    # Convert to meters
    x = x * meters_per_pixel
    y = y * meters_per_pixel

    # Concatenate the meshes
    coordinate_mask = torch.stack([x, y], axis=2)

    return coordinate_mask, X, Y


def align_coordinate_mask_with_ego_vehicle(x, y, yaw, coordinate_mask):
    """Translate and rotate the coordinate mask"""

    # Calculate the rotation matrix
    theta = yaw
    # rotation_matrix = torch.stack([torch.tensor([[torch.cos(theta[t]), -torch.sin(theta[t])], [
    # torch.sin(theta[t]), torch.cos(theta[t])]])for t in
    # range(theta.shape[0])]).to(x.device)
    rotation_matrix = create_2x2_rotation_tensor_from_angle_tensor(theta)

    aligned_coordinate_mask = coordinate_mask.clone()
    # Translate the coordinate mask
    aligned_coordinate_mask[..., 0] -= y.unsqueeze(-1).repeat(
        1,
        1,
        aligned_coordinate_mask.shape[2],
        aligned_coordinate_mask.shape[3],
    )
    aligned_coordinate_mask[..., 1] -= x.unsqueeze(-1).repeat(
        1,
        1,
        aligned_coordinate_mask.shape[2],
        aligned_coordinate_mask.shape[3],
    )

    # Rotate the coordinate mask
    rotation_matrix.unsqueeze_(2).unsqueeze_(3)
    aligned_coordinate_mask.unsqueeze_(-1)
    aligned_coordinate_mask = rotation_matrix @ aligned_coordinate_mask
    aligned_coordinate_mask.squeeze_(-1)
    return aligned_coordinate_mask


def calculate_mask(
    aligned_coordinate_mask, dx, dy, width, length, alpha=0.1, side_mask=True
):
    """Calculate the mask for the ego vehicle"""

    term_1 = (dx - torch.abs(aligned_coordinate_mask[..., 0:1])) / (dx - (width / 2))
    term_2 = (dy - torch.abs(aligned_coordinate_mask[..., 1:2])) / (dy - (length / 2))

    mask_car = torch.maximum(term_1, torch.tensor(0)) * torch.minimum(
        torch.maximum(term_2, torch.tensor(0)), torch.tensor(1.0)
    )
    mask_car = torch.pow(mask_car, alpha)
    mask_car = mask_car.flip(-2).permute((0, 1, 3, 2, -1)).squeeze(-1)

    if not side_mask:
        return mask_car

    mask_side = torch.maximum(term_1, torch.tensor(0)) * torch.maximum(
        term_2, torch.tensor(0)
    )
    mask_side = torch.pow(mask_side, alpha)

    mask_side = mask_side.flip(-2).permute((0, 1, 3, 2, -1)).squeeze(-1)

    return (mask_car, mask_side)


def rotate(location, yaw, location_next, yaw_next):
    """Rotate the location in next time step to write in the frame of first time-step"""

    # Calculate the rotation matrix
    theta = -yaw
    rotation_matrix = torch.tensor(
        [[torch.cos(theta), -torch.sin(theta)], [torch.sin(theta), torch.cos(theta)]]
    )

    # Rotate the coordinate mask
    location_next_prime = rotation_matrix @ location_next.T
    location_prime = rotation_matrix @ location.T

    # Translate the coordinate mask
    delta_x = location_next_prime[0] - location_prime[0]
    delta_y = location_next_prime[1] - location_prime[1]
    delta_theta = yaw_next - yaw

    return (delta_x, delta_y, delta_theta)


def rotate_batched(location, yaw):
    """Rotate the location in next time step to write in the frame of first time-step"""

    # Calculate the rotation matrix
    # theta = -yaw[0]
    theta = -yaw[:, 0]

    # rotation_matrix = torch.tensor([[torch.cos(
    # theta), -torch.sin(theta)], [torch.sin(theta), torch.cos(theta)]],
    # device=location.device)

    rotation_matrix = create_2x2_rotation_tensor_from_angle_tensor(theta)

    # Rotate the coordinate mask
    rotated_coordinates = torch.matmul(rotation_matrix, location.mT).mT

    # Translate the coordinate mask
    # delta_x = rotated_coordinates[1:, 0:1] - rotated_coordinates[0, 0:1]
    # delta_y = rotated_coordinates[1:, 1:2] - rotated_coordinates[0, 1:2]
    delta_x = rotated_coordinates[:, 1:, 0:1] - rotated_coordinates[:, 0:1, 0:1]
    delta_y = rotated_coordinates[:, 1:, 1:2] - rotated_coordinates[:, 0:1, 1:2]
    delta_theta = yaw[:, 1:, 0:1] - yaw[:, 0:1, 0:1]
    return (delta_x, delta_y, delta_theta)


def create_2x2_rotation_tensor_from_angle_tensor(angle_tensor):
    """Create a 2x2 rotation tensor from an angle tensor"""

    rotation_tensor = torch.zeros(
        *angle_tensor.shape[:-1],
        2,
        2,
        device=angle_tensor.device,
        dtype=angle_tensor.dtype,
    )
    cos = torch.cos(angle_tensor).squeeze(-1)
    sin = torch.sin(angle_tensor).squeeze(-1)
    rotation_tensor[..., 0, 0] = cos
    rotation_tensor[..., 0, 1] = -sin
    rotation_tensor[..., 1, 0] = sin
    rotation_tensor[..., 1, 1] = cos
    return rotation_tensor


def _test_transfer_cost_weight_func():
    import tempfile
    import os
    import yaml

    # Test case 1: Vanilla cost weight
    cost_weight = {
        "weight_a": 1,
        "weight_b": 1,
        "weight_c": 1,
    }

    transferred_cost_weight = transfer_cost_weight(cost_weight)

    print(cost_weight)
    print(transferred_cost_weight)

    # Test case 2: Cost weight with a string to a YML file
    cost_weight_to_yml = {
        "cost_weight": {
            "weight_a_yml": 1,
            "weight_b_yml": 1,
            "weight_c_yml": 1,
        }
    }

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_file = os.path.join(temp_dir, "test.yml")
        with open(temp_file, "w") as f:
            yaml.dump(cost_weight_to_yml, f)

        cost_weight = {
            "weight_a": 1,
            "weight_b": 1,
            "weight_c": 1,
            "weight_d": f"{str(temp_file)}",
        }

        transferred_cost_weight = transfer_cost_weight(cost_weight=cost_weight)

    print(cost_weight_to_yml)
    print(cost_weight)
    print(transferred_cost_weight)


def _test_mask_generation():
    speed = 3
    vehicle_width = 2
    vehicle_length = 4
    dx = (
        1.5 * (torch.maximum(torch.tensor(0.1), torch.tensor(speed)) + vehicle_length)
        + 1
    )
    dy = (vehicle_width / 2) + 3

    # Visualization for testing purposes
    coordinate_mask, X, Y = create_coordinate_mask(500, 500, 10, "cpu")
    coordinate_mask_ = coordinate_mask.repeat(1, 1, 1, 1, 1)
    x = torch.tensor([-10.0]).repeat(1, 1, 1)
    y = torch.tensor([10.0]).repeat(1, 1, 1)
    yaw = torch.deg2rad(torch.tensor([-30.0]).repeat(1, 1, 1))
    aligned_coordinate_mask = align_coordinate_mask_with_ego_vehicle(
        x, y, yaw, coordinate_mask_
    )

    mask_car, mask_side = calculate_mask(
        aligned_coordinate_mask, dy, dx, vehicle_width, vehicle_length, 1
    )  # 8 2 2.1 4.9 1

    # coordinate_mask, X, Y = create_coordinate_mask(40, 20, 1)
    # aligned_coordinate_mask = align_coordinate_mask_with_ego_vehicle(0, 0, 0, coordinate_mask)
    # mask_car, mask_side = calculate_mask(aligned_coordinate_mask, 3, 2, 2, 3, 1)
    aligned_coordinate_mask = aligned_coordinate_mask[0, 0]
    mask_car = mask_car[0, 0]
    mask_side = mask_side[0, 0]
    import matplotlib.pyplot as plt

    plt.figure()
    plt.quiver(X, Y, coordinate_mask[..., 0], coordinate_mask[..., 1])

    plt.figure()
    plt.quiver(X, Y, aligned_coordinate_mask[..., 0], aligned_coordinate_mask[..., 1])

    plt.figure()
    plt.imshow(coordinate_mask[..., 0], cmap="hot", interpolation="nearest")
    plt.colorbar()

    plt.figure()
    plt.imshow(coordinate_mask[..., 1], cmap="hot", interpolation="nearest")
    plt.colorbar()

    plt.figure()
    plt.imshow(aligned_coordinate_mask[..., 0], cmap="hot", interpolation="nearest")
    plt.colorbar()

    plt.figure()
    plt.imshow(aligned_coordinate_mask[..., 1], cmap="hot", interpolation="nearest")
    plt.colorbar()

    plt.figure()
    plt.imshow(mask_car, cmap="hot", interpolation="nearest")
    plt.colorbar()

    plt.figure()
    plt.imshow(mask_side, cmap="hot", interpolation="nearest")
    plt.colorbar()

    plt.show()


if __name__ == "__main__":
    print("Testing:")
    print("Type C for cost weight transfer")
    print("Type M for mask generation")
    test_case = input("Enter test case: ")

    if test_case == "C":
        _test_transfer_cost_weight_func()

    elif test_case == "M":
        _test_mask_generation()
