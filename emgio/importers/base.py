from abc import ABC, abstractmethod

from ..core.emg import EMG


class BaseImporter(ABC):
    """Base class for EMG data importers."""

    @abstractmethod
    def load(self, filepath: str) -> EMG:
        """
        Load EMG data from file.

        Args:
            filepath: Path to the input file

        Returns:
            EMG: EMG object containing the loaded data
        """
        pass
