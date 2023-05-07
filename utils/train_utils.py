import torch
import numpy as np


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def organize_device(device):
    if isinstance(device, int):
        return f"cuda:{device}"

    return device


def to(dict_, device=None, index_start=None, index_end=None):
    if index_start is not None and index_end is not None:
        dict_new = {}
        for k in dict_.keys():
            dict_new[k] = dict_[k][:, index_start:index_end].to(device)
        return dict_new
    elif (index_start is not None) and (index_end is None):
        dict_new = {}
        for k in dict_.keys():
            dict_new[k] = dict_[k][:, index_start:].to(device)
        return dict_new
    elif (index_start is None) and (index_end is not None):
        dict_new = {}
        for k in dict_.keys():
            dict_new[k] = dict_[k][:, :index_end].to(device)
        return dict_new
    else:
        for k in dict_.keys():
            dict_[k] = dict_[k].to(device)
        return dict_


def clone(dict_):
    dict_new = {}
    for k in dict_.keys():
        dict_new[k] = dict_[k].clone()
    return dict_new


def cat(list_of_dicts, dim):
    dict_new = {}
    for k in list_of_dicts[0].keys():
        dict_new[k] = torch.cat([d[k] for d in list_of_dicts], dim=dim)
    return dict_new


def stack(list_of_dicts, dim):
    dict_new = {}
    for k in list_of_dicts[0].keys():
        dict_new[k] = torch.stack([d[k] for d in list_of_dicts], dim=dim)
    return dict_new


def requires_grad(dict_, flag=True):
    for k in dict_.keys():
        dict_[k].requires_grad = flag
    return dict_
