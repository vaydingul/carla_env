import torch


def create_coordinate_mask(nx, ny, pixels_per_meter):
    """Create a matrix such that the value at each index is the distance from the center of the image"""

    meters_per_pixel = 1 / pixels_per_meter

    # Create a matrix of the same size as the image
    X, Y = torch.meshgrid(torch.arange(nx), torch.arange(ny))
    # Calculate the distance from the center of the image
    x = X - nx // 2
    y = Y - ny // 2
    # Convert to meters
    x = x * meters_per_pixel
    y = y * meters_per_pixel

    # Concatenate the meshes
    coordinate_mask = torch.stack([x, y], axis=2)

    return coordinate_mask, X, Y


def align_coordinate_mask_with_ego_vehicle(x, y, yaw, coordinate_mask):
    """Translate and rotate the coordinate mask"""

    # Calculate the rotation matrix
    theta = torch.deg2rad(torch.tensor(yaw))
    rotation_matrix = torch.tensor(
        [[torch.cos(theta), -torch.sin(theta)], [torch.sin(theta), torch.cos(theta)]])

    aligned_coordinate_mask = coordinate_mask.clone()
    # Translate the coordinate mask
    aligned_coordinate_mask[..., 0] -= x
    aligned_coordinate_mask[..., 1] -= y

    # Rotate the coordinate mask
    aligned_coordinate_mask = aligned_coordinate_mask @ rotation_matrix

    return aligned_coordinate_mask


def calculate_mask(aligned_coordinate_mask, dx, dy, width, length, alpha=0.1):

    term_1 = (
        dx - torch.abs(aligned_coordinate_mask[..., 0])) / (dx - (length / 2))
    term_2 = (
        dy - torch.abs(aligned_coordinate_mask[..., 1])) / (dy - (width / 2))

    mask_car = torch.maximum(term_1, torch.tensor(
        0)) * torch.minimum(torch.maximum(term_2, torch.tensor(0)), torch.tensor(1))
    mask_car = torch.pow(mask_car, alpha)

    mask_side = torch.maximum(term_1, torch.tensor(
        0)) * torch.maximum(term_2, torch.tensor(0))
    mask_side = torch.pow(mask_side, alpha)

    return mask_car, mask_side


if __name__ == "__main__":

    speed = 3
    vehicle_width = 2
    vehicle_length = 4
    dx = 1.5 * (torch.maximum(torch.tensor(0.1),
                torch.tensor(speed)) + vehicle_length) + 1
    dy = (vehicle_width / 2) + 3

    # Visualization for testing purposes
    coordinate_mask, X, Y = create_coordinate_mask(800, 600, 18)
    aligned_coordinate_mask = align_coordinate_mask_with_ego_vehicle(
        10, 10, 30, coordinate_mask)

    mask_car, mask_side = calculate_mask(
        aligned_coordinate_mask, dx, dy, vehicle_width, vehicle_length, 1)  # 8 2 2.1 4.9 1

    # coordinate_mask, X, Y = create_coordinate_mask(20, 20, 1)
    # aligned_coordinate_mask = align_coordinate_mask_with_ego_vehicle(5, 5, 30, coordinate_mask)
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
    plt.imshow(mask_car.flip(0), cmap='hot', interpolation='nearest')
    plt.colorbar()

    plt.figure()
    plt.imshow(mask_side.flip(0), cmap='hot', interpolation='nearest')
    plt.colorbar()

    plt.show()
