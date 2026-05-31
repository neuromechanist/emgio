"""emgio (evolving into biosigIO): import/export biosignal recordings across formats.

The core class is :class:`Recording` (modality-agnostic: EEG/EMG/iEEG/MEG/...).
``EMG`` is a deprecated alias of ``Recording`` that still works but emits a
``DeprecationWarning``; it will be removed in biosigio 1.0.0.
"""

from .core.emg import Recording
from .exporters.edf import EDFExporter
from .importers.trigno import TrignoImporter
from .version import __version__, __version_info__

__all__ = [
    "Recording",
    "TrignoImporter",
    "EDFExporter",
    "__version__",
    "__version_info__",
]


def __getattr__(name: str):
    """Resolve the deprecated ``EMG`` alias to ``Recording`` with a warning (PEP 562)."""
    if name == "EMG":
        import warnings

        warnings.warn(
            "emgio.EMG is a deprecated alias of emgio.Recording and will be removed "
            "in biosigio 1.0.0; use emgio.Recording instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return Recording
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
