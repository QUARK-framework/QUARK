from pathlib import Path


def get_project_root() -> Path:
    """Define project root. Any module which calls get_project_root can be
    moved without changing program behavior."""
    return Path(__file__).parent.parent


def get_project_output() -> Path:
    """Return output folder. If not exist, create it before."""
    output_path = get_project_root() / "output"
    output_path.mkdir(exist_ok=True)
    return output_path


def get_src_root() -> Path:
    """Return src folder."""
    return get_project_root() / "src"


def get_logging_path() -> Path:
    """Return path to the logging config file."""
    return get_src_root() / "logging.conf"
