import importlib.metadata

try:
    __version__ = importlib.metadata.version(__name__ or __package__)
except importlib.metadata.PackageNotFoundError:
    # Package not installed, use fallback version
    __version__ = "3.0.0"

version_split = __version__.split(".")
__spec_version__ = (1000 * int(version_split[0])) + (10 * int(version_split[1])) + (1 * int(version_split[2]))
