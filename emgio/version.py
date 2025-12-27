"""Version information for EMGIO."""

__version__ = "0.2.0"
__version_info__ = (0, 2, 0)


def get_version() -> str:
    """Get the current version string."""
    return __version__


def get_version_info() -> tuple:
    """Get the version info tuple (major, minor, patch)."""
    return __version_info__
