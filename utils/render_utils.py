import cv2
import numpy as np
from carla_env.bev import BirdViewProducer


def postprocess_bev(bev, bev_selected_channels):

    bev[bev > 0.5] = 1
    bev[bev <= 0.5] = 0
    bev = bev.clone().detach().cpu().numpy()
    bev = np.transpose(bev, (1, 2, 0))
    bev = BirdViewProducer.as_rgb_with_indices(
        bev, bev_selected_channels
    )
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
