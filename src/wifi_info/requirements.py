from shutil import which
from warnings import warn


class Applications:
    """Holds the required applications."""

    REQUIRED = ["iw"]
    """A list of required applications."""
    RECOMMENDED = {"iwconfig": "It is used as a fallback for iw"}
    """A dictionary of recommended applications and their reasons."""


class ApplicationNotInstalledException(Exception):
    """Raised when an application is not installed."""


class ApplicationNotInstalledWarning(UserWarning):
    """Raised when an application is not installed."""


def check_application(name: str) -> bool:
    """Checks if the application is installed
    in the current environment.

    Args:
        name (str): name or path of the application.

    Returns:
        bool: True if the application is installed, False otherwise.
    """
    return which(name) is not None


def check_required_applications() -> bool:
    """Checks if all required applications are installed
    in the current environment.

    This function will raise an exception if any of the checks fail
    and report the missing application.

    Returns:
        bool: True if all required applications are installed, False otherwise.
    """
    for app in Applications.REQUIRED:
        if not check_application(app):
            raise ApplicationNotInstalledException(f"{app} is not installed.")
    return True


def check_recommended_applications() -> bool:
    """Checks if all recommended applications are installed
    in the current environment.

    This function will raise a warning if any of the checks fail
    and report the missing application.

    Returns:
        bool: True if all recommended applications are installed, False otherwise.
    """
    ret_val = True
    for app, reason in Applications.RECOMMENDED.items():
        if not check_application(app):
            warn(ApplicationNotInstalledWarning(f"{app} is not installed. {reason}"))
            ret_val = False
    return ret_val


def run_checks() -> None:
    """Runs all required checks.

    This function will raise an exception if any of the checks fail.
    It should be called at the start of the program.
    """
    check_required_applications()
    check_recommended_applications()
