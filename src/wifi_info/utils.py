import ipaddress
import logging
import socket
import subprocess
import typing
from re import match

import netifaces
from pydantic import validate_call


class ExternalError(Exception):
    """Raised when an external command fails."""


class InterfaceError(Exception):
    """Raised when an error with an interface is encountered."""


class InterfaceNotFoundError(InterfaceError):
    """Raised when an interface is not found."""


class InterfaceNotReadyError(InterfaceError):
    """Raised when an interface is not ready."""


class InterfaceMismatchError(InterfaceError):
    """Raised when an interface does not match the expected one.

    An example would be when a wireless interface is expected
    but a wired interface is found.
    """


class InvalidAddressError(ValueError):
    """Raised when an IP or MAC address is invalid."""


def get_command_output(
    cmd: str | typing.Sequence[str], shell: bool = False, timeout: float | None = None
) -> str:
    """Executes a command and returns its output.

    Args:
        cmd (str | list[str]): Command to execute.
        shell (bool, optional): Whether or not to execute on shell. Defaults to False.
        timeout (float, optional): The maximum timeout for the command in seconds. Defaults to None.

    Returns:
        str: The command's output if successful.
        One of ["invalid", "timeout", "unavailable"] otherwise.
    """
    logging.debug(f"Executing command '{cmd}'...")
    try:
        return subprocess.run(
            cmd,
            shell=shell,
            timeout=timeout,
            check=True,
            universal_newlines=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        ).stdout
    except subprocess.TimeoutExpired:
        return "timeout"
    except subprocess.CalledProcessError:
        return "invalid"
    # except FileNotFoundError:
    #     return "unavailable"


def verify_network_interface(interface_name: str) -> bool:
    """Verifies if a network interface is available.

    This function raises an exception if the interface is not available.

    Args:
        interface_name (str): The name of the network interface.

    Returns:
        bool: True if the interface is available.
    """

    cmd = f"cat /sys/class/net/{interface_name}/operstate"
    output = get_command_output(cmd, shell=True, timeout=10)
    logging.debug(f"Checking network interface '{interface_name}'...")
    logging.debug(output)

    if output == "invalid":
        logging.exception(f"Network interface '{interface_name}' does not exist.")
        raise InterfaceNotFoundError(
            f"Network interface '{interface_name}' does not exist."
        )
    if output == "timeout":
        logging.exception(
            f"Unable to get status of network interface '{interface_name}'. Timeout exceeded."
        )
        raise TimeoutError(
            f"Unable to get status of network interface '{interface_name}'."
        )

    if not output.startswith("up"):
        logging.exception(f"Network interface '{interface_name}' is not ready.")
        raise InterfaceNotReadyError(
            f"Network interface '{interface_name}' is not ready."
        )

    return True


def verify_connection(ip_address: str, port: str | int) -> bool:
    """Create a socket and verify if a connection
    can be established on the provided ip and port.

    Args:
        ip (str): The ip address of the connection.
        port (str | int): The port of the connection.

    Returns:
        bool: True if connection could be established.
        False if it failed to establish.
    """
    logging.debug(f"Verifying if connection is available on '{ip_address}:{port}'...")
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(10)
    result = sock.connect_ex((ip_address, int(port)))
    sock.close()

    logging.debug(f"Connection errno: {result}")
    return result == 0


def verify_mac_address(mac_address: str) -> bool:
    """Verifies if a MAC address is valid.

    Args:
        mac_address (str): The MAC address to verify.

    Returns:
        bool: True if the MAC address is valid, false otherwise.
    """
    logging.debug(f"Verifying MAC address '{mac_address}'...")
    return match(r"^([0-9A-Fa-f]{2}[:-]){5}([0-9A-Fa-f]{2})$", mac_address) is not None


@validate_call
def verify_ip_address(ip_address: str) -> bool:
    """Verifies if an IP address is valid.

    Args:
        ip_address (str): The IP address to verify.

    Returns:
        bool: True if the IP address is valid, false otherwise.
    """
    logging.debug(f"Verifying IP address '{ip_address}'...")

    try:
        ipaddress.ip_address(ip_address)
    except ValueError:
        return False
    return True


def get_interface_ipv4_address(interface_name: str) -> ipaddress.IPv4Address:
    """Looks up the IPv4 address of an interface.

    Args:
        interface_name (str): The name of the network interface.

    Returns:
        ipaddress.IPv4Address: The IPv4 address of the interface.
        Returns '0.0.0.0' if the interface does not have an IPv4 address.
    """
    logging.debug(f"Getting IPv4 address of interface '{interface_name}'...")

    addresses = netifaces.ifaddresses(interface_name)

    try:
        return ipaddress.IPv4Address(addresses[netifaces.AF_INET][0]["addr"])
    except (KeyError, IndexError):
        return ipaddress.IPv4Address("0.0.0.0")
