from ipaddress import IPv4Address

import netifaces
import pytest
from pydantic import ValidationError

from wifi_info import utils


@pytest.fixture()
def command_output_override(monkeypatch, request):
    marker = request.node.get_closest_marker("output")

    def ret_output(
        cmd: str | list[str], shell: bool = False, timeout: float | None = None
    ):
        del cmd, shell, timeout  # unused
        return marker.args[0]

    monkeypatch.setattr(utils, "get_command_output", ret_output)


@pytest.fixture()
def interface_address(monkeypatch, request):
    output = request.param

    def ret_address(interface: str):
        del interface  # unused
        return output

    monkeypatch.setattr(netifaces, "ifaddresses", ret_address)


@pytest.mark.parametrize(
    "cmd, expected",
    [
        ("echo hello", "hello\n"),
        ("sleep 5", "timeout"),
        ("cat ___igneaongeapn___", "invalid"),
        ("__ongeaomwapvmeb__", "invalid"),
    ],
)
def test_get_command_output(cmd, expected):
    assert utils.get_command_output(cmd, shell=True, timeout=2) == expected


@pytest.mark.parametrize(
    "ip_address, expected",
    [
        ("192.168.1.1", True),
        ("192.168.1", False),
        ("127.0.0.1", True),
        ("0.0.0.0", True),
        ("256.0.0.0", False),
        ("asdf", False),
        ("", False),
    ],
)
def test_many_verify_ip_address(ip_address, expected):
    assert utils.verify_ip_address(ip_address) is expected


@pytest.mark.output("timeout")
def test_verify_network_interface_timeout(command_output_override):
    with pytest.raises(TimeoutError):
        utils.verify_network_interface("eth0")


@pytest.mark.output("down")
def test_verify_network_interface_down(command_output_override):
    with pytest.raises(utils.InterfaceNotReadyError):
        utils.verify_network_interface("eth0")


@pytest.mark.parametrize(
    "ip_address, expected_exception",
    [
        (265, ValidationError),
        (0.0, ValidationError),
    ],
)
def test_exception_verify_ip_address(ip_address, expected_exception):
    with pytest.raises(expected_exception):
        utils.verify_ip_address(ip_address)


@pytest.mark.parametrize(
    "mac_address, expected",
    [
        ("00:00:00:00:00:00", True),
        ("ad:be:ff:52:78:e5", True),
        ("00:00:00:00:00", False),
        ("00:00:gg:00:00:00", False),
        ("asdf", False),
        ("", False),
    ],
)
def test_many_verify_mac_address(mac_address, expected):
    assert utils.verify_mac_address(mac_address) is expected


@pytest.mark.parametrize(
    "mac_address, expected_exception",
    [
        (55, TypeError),
        (0.0, TypeError),
    ],
)
def test_exception_verify_mac_address(mac_address, expected_exception):
    with pytest.raises(expected_exception):
        utils.verify_mac_address(mac_address)


def test_verify_network_interface():
    interfaces = netifaces.interfaces()
    interfaces.remove("lo")
    interfaces = [
        i for i in interfaces if netifaces.AF_INET in netifaces.ifaddresses(i)
    ]

    assert utils.verify_network_interface(interfaces[0])


@pytest.mark.parametrize(
    "interface_name, expected_exception",
    [
        ("asdf", utils.InterfaceNotFoundError),
        ("lo", utils.InterfaceNotReadyError),
    ],
)
def test_verify_network_interface_exception(interface_name, expected_exception):
    with pytest.raises(expected_exception):
        utils.verify_network_interface(interface_name)


def test_get_interface_ipv4_address():
    assert utils.get_interface_ipv4_address("lo") == IPv4Address("127.0.0.1")


@pytest.mark.parametrize(
    "interface_name",
    [
        ("",),
        ("asdf",),
    ],
)
def test_get_interface_ipv4_address_invalid_interface(interface_name):
    with pytest.raises(ValueError):
        utils.get_interface_ipv4_address("asdf")


@pytest.mark.parametrize(
    "interface_address",
    [
        {},
        {netifaces.AF_LINK: [{}]},
        {netifaces.AF_INET: [{"broadcast": "ff:ff:ff:ff:ff:ff"}]},
    ],
    indirect=True,
)
def test_get_interface_ipv4_address_no_inet(interface_address):
    assert utils.get_interface_ipv4_address(interface_address) == IPv4Address("0.0.0.0")
