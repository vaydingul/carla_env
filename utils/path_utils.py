import pathlib
import os
from datetime import datetime


def check_latest_episode(path):
    """Check the latest episode in a directory"""
    path = pathlib.Path(path)
    if (not path.exists()) or (os.path.getsize(path) == 0):
        return 0
    else:
        # Get the latest episode
        latest_episode = max([int(x.name.split("_")[-1]) for x in path.iterdir()])
        return latest_episode + 1


def create_date_time_path(path):
    """Create a path with date and time"""
    path = pathlib.Path(path)
    date_ = pathlib.Path(datetime.today().strftime("%Y-%m-%d"))
    time_ = pathlib.Path(datetime.today().strftime("%H-%M-%S"))
    path = path / date_ / time_
    path.mkdir(parents=True, exist_ok=True)
    return path
