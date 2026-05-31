from abc import ABC, abstractmethod

from ..core.emg import Recording


class BaseImporter(ABC):
    """Base class for EMG data importers."""

    @abstractmethod
    def load(self, filepath: str) -> Recording:
        """
        Load EMG data from file.

        Args:
            filepath: Path to the input file

        Returns:
            Recording: Recording object containing the loaded data
        """
        pass
