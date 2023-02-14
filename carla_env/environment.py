import abc


class Environment(abc.ABC):
    """Abstract base class for simulator environment"""

    @abc.abstractmethod
    def reset(self):
        """Reset the environment"""
        pass

    @abc.abstractmethod
    def step(self, action):
        """Perform an action in the environment"""
        pass

    @abc.abstractmethod
    def render(self):
        """Render the environment"""
        pass

    @abc.abstractmethod
    def close(self):
        """Close the environment"""
        pass

    @abc.abstractmethod
    def seed(self, seed):
        """Set the seed for the environment"""
        pass
