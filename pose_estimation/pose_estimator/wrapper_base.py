"""
Pose Estimation Wrapper Base
"""

from abc import ABC, abstractmethod


class PoseEstimationWrapperBase(ABC):
    """
    Pose Estimation Wrapper Base
    """

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def _initialize(self):
        """
        Initialize Pose Estimation Model
        """
        pass

    # @abstractmethod
    # def predict(self, rgb):
    #     """
    #     Estimate Pose from image
    #     """
    #     pass
