import torch
import numpy as np
import numbers

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
    """
    Recursive function that moves all tensors in a dictionary to a device.
    """

    assert isinstance(dict_, dict), "Input must be a dictionary."

    dict_new = {}

    for k, v in dict_.items():
        if isinstance(v, dict):
            dict_new[k] = to(
                v, device=device, index_start=index_start, index_end=index_end
            )

        else:
            if index_start is not None and index_end is not None:
                dict_new[k] = dict_[k][:, index_start:index_end].to(device)

            elif (index_start is not None) and (index_end is None):
                dict_new[k] = dict_[k][:, index_start:].to(device)

            elif (index_start is None) and (index_end is not None):
                dict_new[k] = dict_[k][:, :index_end].to(device)

            else:
                dict_new[k] = dict_[k].to(device)

    return dict_new


def clone(dict_):
    """
    Recursive function that clones a dictionary of tensors.
    """

    assert isinstance(dict_, dict), "Input must be a dictionary."

    dict_new = {}
    for k, v in dict_.items():
        if isinstance(v, dict):
            dict_new[k] = clone(v)
        else:
            dict_new[k] = v.clone()

    return dict_new


def cat(list_of_dicts, dim):

    assert isinstance(list_of_dicts, list), "Input must be a list of dictionaries."
    
    elem = list_of_dicts[0]

    dict_new = {}

    for (k, v) in elem.items():

        if isinstance(v, dict):
            dict_new[k] = cat([d[k] for d in list_of_dicts], dim=dim)

        else:
            dict_new[k] = torch.cat([d[k] for d in list_of_dicts], dim=dim)
    
    return dict_new


def stack(list_of_dicts, dim):
    
    assert isinstance(list_of_dicts, list), "Input must be a list of dictionaries."
    
    elem = list_of_dicts[0]

    dict_new = {}

    for (k, v) in elem.items():

        if isinstance(v, dict):
            dict_new[k] = stack([d[k] for d in list_of_dicts], dim=dim)

        else:
            dict_new[k] = torch.stack([d[k] for d in list_of_dicts], dim=dim)
    
    return dict_new


def requires_grad(dict_, flag=True):
    """
    Recursive function that sets the requires_grad flag of all tensors in a dictionary.
    """

    assert isinstance(dict_, dict), "Input must be a dictionary."

    for k, v in dict_.items():
        if isinstance(v, dict):
            requires_grad(v, flag=flag)
        else:
            v.requires_grad = flag


def apply_torch_func(dict_, func, *args, **kwargs):
    assert isinstance(dict_, dict), "Input must be a dictionary."

    dict_new = {}
    
    for k, v in dict_.items():
        if isinstance(v, dict):
            dict_new[k] = apply_torch_func(v, func, *args, **kwargs)
        elif isinstance(v, numbers.Number) or isinstance(v, torch.Tensor):
            dict_new[k] = func(v, *args, **kwargs)

    return dict_new
