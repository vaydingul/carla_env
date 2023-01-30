import pathlib
import os


def check_latest_episode(path):
    """Check the latest episode in a directory"""
    path = pathlib.Path(path)
    if (not path.exists()) or (os.path.getsize(path) == 0):
        return 0
    else:
        # Get the latest episode
        latest_episode = max([int(x.name.split("_")[-1]) for x in path.iterdir()])
        return latest_episode + 1
