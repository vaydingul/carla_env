import torch


def create_coordinate_mask(nx, ny, pixels_per_meter, device):
    """Create a matrix such that the value at each index is the distance from the center of the image"""

    meters_per_pixel = 1 / pixels_per_meter

    # Create a matrix of the same size as the image
    X, Y = torch.meshgrid(
        torch.arange(
            nx, device=device), torch.arange(
            ny, device=device))

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
    rotation_matrix = torch.stack([torch.tensor([[torch.cos(theta[t]), -torch.sin(theta[t])], [
                                  torch.sin(theta[t]), torch.cos(theta[t])]])for t in range(theta.shape[0])]).to(x.device)

    aligned_coordinate_mask = coordinate_mask.clone()
    # Translate the coordinate mask
    aligned_coordinate_mask[..., 0:1] -= y.reshape(-1, 1, 1, 1)
    aligned_coordinate_mask[..., 1:2] -= x.reshape(-1, 1, 1, 1)

    # Rotate the coordinate mask
    rotation_matrix.unsqueeze_(1).unsqueeze_(2)
    aligned_coordinate_mask.unsqueeze_(-1)
    aligned_coordinate_mask = rotation_matrix @ aligned_coordinate_mask
    aligned_coordinate_mask.squeeze_(-1)
    return aligned_coordinate_mask


def calculate_mask(aligned_coordinate_mask, dx, dy, width, length, alpha=0.1):
    """Calculate the mask for the ego vehicle"""

    term_1 = (
        dx - torch.abs(aligned_coordinate_mask[..., 0:1])) / (dx - (width / 2))
    term_2 = (
        dy - torch.abs(aligned_coordinate_mask[..., 1:2])) / (dy - (length / 2))

    mask_car = torch.maximum(term_1, torch.tensor(
        0)) * torch.minimum(torch.maximum(term_2, torch.tensor(0)), torch.tensor(1))
    mask_car = torch.pow(mask_car, alpha)

    mask_side = torch.maximum(term_1, torch.tensor(
        0)) * torch.maximum(term_2, torch.tensor(0))
    mask_side = torch.pow(mask_side, alpha)

    mask_car = mask_car.flip(-2).permute((0, 2, 1, 3)).squeeze(-1)
    mask_side = mask_side.flip(-2).permute((0, 2, 1, 3)).squeeze(-1)

    return (mask_car, mask_side)


def rotate(location, yaw, location_next, yaw_next):
    """Rotate the location in next time step to write in the frame of first time-step"""

    # Calculate the rotation matrix
    theta = -yaw
    rotation_matrix = torch.tensor(
        [[torch.cos(theta), -torch.sin(theta)], [torch.sin(theta), torch.cos(theta)]])

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
    theta = -yaw[0]
    rotation_matrix = torch.tensor([[torch.cos(
        theta), -torch.sin(theta)], [torch.sin(theta), torch.cos(theta)]], device=location.device)

    # Rotate the coordinate mask
    rotated_coordinates = torch.matmul(rotation_matrix, location.mT).mT

    # Translate the coordinate mask
    delta_x = rotated_coordinates[1:, 0:1] - rotated_coordinates[0, 0:1]
    delta_y = rotated_coordinates[1:, 1:2] - rotated_coordinates[0, 1:2]
    delta_theta = yaw[1:, 0:1] - yaw[0, 0:1]
    return (delta_x, delta_y, delta_theta)


if __name__ == "__main__":

    speed = 3
    vehicle_width = 2
    vehicle_length = 4
    dx = 1.5 * (torch.maximum(torch.tensor(0.1),
                              torch.tensor(speed)) + vehicle_length) + 1
    dy = (vehicle_width / 2) + 3

    # Visualization for testing purposes
    coordinate_mask, X, Y = create_coordinate_mask(800, 600, 18, 'cpu')
    aligned_coordinate_mask = align_coordinate_mask_with_ego_vehicle(
        10, 10, 30, coordinate_mask)

    mask_car, mask_side = calculate_mask(
        aligned_coordinate_mask, dy, dx, vehicle_width, vehicle_length, 1)  # 8 2 2.1 4.9 1

    # coordinate_mask, X, Y = create_coordinate_mask(40, 20, 1)
    # aligned_coordinate_mask = align_coordinate_mask_with_ego_vehicle(0, 0, 0, coordinate_mask)
    # mask_car, mask_side = calculate_mask(aligned_coordinate_mask, 3, 2, 2, 3, 1)

    import matplotlib.pyplot as plt

    plt.figure()
    plt.quiver(X, Y, coordinate_mask[..., 0], coordinate_mask[..., 1])

    plt.figure()
    plt.quiver(
        X, Y, aligned_coordinate_mask[..., 0], aligned_coordinate_mask[..., 1])

    plt.figure()
    plt.imshow(coordinate_mask[..., 0], cmap='hot', interpolation='nearest')
    plt.colorbar()

    plt.figure()
    plt.imshow(coordinate_mask[..., 1], cmap='hot', interpolation='nearest')
    plt.colorbar()

    plt.figure()
    plt.imshow(aligned_coordinate_mask[..., 0],
               cmap='hot', interpolation='nearest')
    plt.colorbar()

    plt.figure()
    plt.imshow(aligned_coordinate_mask[..., 1],
               cmap='hot', interpolation='nearest')
    plt.colorbar()

    plt.figure()
    plt.imshow(mask_car.flip(-1).T, cmap='hot', interpolation='nearest')
    plt.colorbar()

    plt.figure()
    plt.imshow(mask_side.flip(-1).T, cmap='hot', interpolation='nearest')
    plt.colorbar()

    plt.show()