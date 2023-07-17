import pytest
from pydantic import ValidationError

from wifi_info import utils, wireless


@pytest.fixture()
def fake_interface(monkeypatch):
    def verify_interface(interface_name):
        del interface_name  # unused
        return True

    def get_interface_ip(interface_name):
        del interface_name  # unused
        return "0.0.0.1"

    monkeypatch.setattr(utils, "verify_network_interface", verify_interface)
    monkeypatch.setattr(utils, "get_interface_ipv4_address", get_interface_ip)


@pytest.fixture()
def fake_metrics(monkeypatch, request, fake_interface):
    # iw info
    iw_info = """Interface mock_interface
        [...]
        addr <interface_mac>
        ssid mock_ssid
        [...]
        channel 42 (7777 MHz), width: 80 MHz, center1: 5210 MHz
        txpower 700.00 dBm
        [...]
    """
    # iw link
    iw_link = """Connected to <station_mac> (on mock_interface)
        SSID: mock_ssid
        freq: 2447
        RX: 2368642908 bytes (1622705 packets)
        TX: 23742420 bytes (181813 packets)
        signal: -333 dBm
        rx bitrate: 144.4 MBit/s MCS 15 short GI
        tx bitrate: 144.4 MBit/s MCS 15 short GI
    """
    # iwconfig
    iwconfig = """wlp0s20f3  IEEE 802.11  ESSID:"mock_ssid"  
            [...]
            Power Management:on
            Link Quality=70/70  Signal level=-999 dBm  
            [...]
    """

    ret_dict = {"info_used": False, "link_used": False, "iwconfig_used": False}

    marker = request.node.get_closest_marker("missing_metrics")
    if marker is None:
        missing = []
    else:
        missing = marker.args[0]

    marker = request.node.get_closest_marker("interface_mac")
    if marker is None:
        interface_mac = "aa:bb:cc:dd:ee:ff"
    else:
        interface_mac = marker.args[0]

    marker = request.node.get_closest_marker("station_mac")
    if marker is None:
        station_mac = "00:11:22:33:44:55"
    else:
        station_mac = marker.args[0]

    def get_cmd_output(
        cmd: str | list[str], shell: bool = False, timeout: float | None = None
    ):
        del shell, timeout  # unused
        if "info" in cmd:
            output = iw_info.replace("<interface_mac>", interface_mac)
            ret_dict["info_used"] = True
        elif "link" in cmd:
            output = iw_link.replace("<station_mac>", station_mac)
            ret_dict["link_used"] = True
        elif "iwconfig" in cmd:
            output = iwconfig
            ret_dict["iwconfig_used"] = True
        else:
            return "invalid"

        output = "\n".join(
            line
            for line in output.splitlines()
            if not any(keyword in line for keyword in missing)
        )
        return output

    monkeypatch.setattr(utils, "get_command_output", get_cmd_output)

    return ret_dict


def test_get_wireless_metrics(fake_metrics, fake_interface):
    metrics = wireless.get_wireless_metrics("mock_wifi")
    assert metrics.interface_name == "mock_interface"
    assert metrics.interface_mac_address == "aa:bb:cc:dd:ee:ff"
    assert metrics.ssid == "mock_ssid"
    assert metrics.ssid_mac_address == "00:11:22:33:44:55"
    assert metrics.channel == 42
    assert metrics.channel_frequency == 7777e6
    assert metrics.tx_power == 700.00
    assert metrics.signal_strength == -333


def test_get_wireless_metrics_exception():
    with pytest.raises(utils.InterfaceNotFoundError):
        wireless.get_wireless_metrics("asdf")


def test_get_wireless_metrics_wired(fake_interface):
    with pytest.raises(utils.InterfaceMismatchError):
        wireless.get_wireless_metrics("mock_eth0")


@pytest.mark.missing_metrics(["signal"])
def test_get_wireless_metrics_iwconfig_fallback(fake_metrics):
    metrics = wireless.get_wireless_metrics("mock_wifi")
    assert metrics.signal_strength == -999
    assert fake_metrics["iwconfig_used"] is True


@pytest.mark.interface_mac("invalid_mac")
def test_get_wireless_metrics_invalid_interface_mac(fake_metrics):
    with pytest.raises(ValidationError):
        wireless.get_wireless_metrics("mock_wifi")


@pytest.mark.station_mac("invalid_mac")
def test_get_wireless_metrics_invalid_station_mac(fake_metrics):
    with pytest.raises(ValidationError):
        wireless.get_wireless_metrics("mock_wifi")


@pytest.mark.missing_metrics(["ssid"])
def test_get_wireless_metrics_parse_error(fake_metrics):
    with pytest.raises(wireless.ParseError):
        wireless.get_wireless_metrics("mock_wifi")
