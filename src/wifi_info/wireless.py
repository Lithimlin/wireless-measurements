import logging
from datetime import datetime, timedelta
from ipaddress import IPv4Address
from re import findall, search
from typing import Any, Optional, Sequence

import iperf3  # type: ignore[import]
from pydantic import (
    AliasPath,
    BaseModel,
    ByteSize,
    Field,
    SerializeAsAny,
    TypeAdapter,
    ValidationError,
    field_serializer,
    field_validator,
    model_serializer,
    model_validator,
)

from wifi_info import settings, utils


class ParseError(Exception):
    """Raised when a command's output cannot be parsed."""


class WirelessMetrics(BaseModel):
    """A model for wireless metrics."""

    interface_name: str
    interface_mac_address: str
    interface_ip_address: IPv4Address
    channel: int
    channel_frequency: float
    tx_power: float
    ssid: str
    ssid_mac_address: str
    signal_strength: int

    @field_validator("interface_mac_address", "ssid_mac_address")
    def _validate_mac_address(cls, value: str) -> str:
        if not utils.verify_mac_address(value):
            raise utils.InvalidAddressError(f"Invalid MAC address '{value}'.")
        return value

    @field_serializer("interface_ip_address")
    def serialize_ip_address(self, value: IPv4Address) -> str:
        return str(value)


def get_wireless_metrics(
    interface: settings.InterfaceSettings | str,
) -> WirelessMetrics:
    """Collects the metrics from a wireless interface.

    Args:
        interface (settings.InterfaceSettings | str): An interface setting
            or name of the wireless interface as a string.

    Raises:
        utils.InterfaceMismatchError: Raised when the interface is not a wireless interface.
        ParseError: Raised when the output from an invoked command cannot be parsed.

    Returns:
        WirelessMetrics: The gathered metrics.
    """
    if isinstance(interface, str):
        interface = settings.InterfaceSettings(name=interface)
    interface_name = interface.name
    logging.info(f"Collecting wireless metrics from '{interface_name}'...")
    # utils.verify_network_interface(interface_name)

    cmd = f"iw {interface_name} info"
    output = utils.get_command_output(cmd, shell=True, timeout=10).replace("\t", " ")
    logging.debug(f"Output:\n{output}")
    if output == "invalid":
        logging.exception(
            f"Network interface '{interface_name}' is not a wireless interface."
        )
        raise utils.InterfaceMismatchError(
            f"Network interface '{interface_name}' is not a wireless interface."
        )

    def parse_metric(source: str, regex: str) -> str:
        try:
            match = findall(regex, source)[0]
        except IndexError as err:
            logging.exception(
                f"Unable to parse metric from wireless interface '{interface_name}'."
            )
            raise ParseError(
                f"Unable to parse metric from wireless interface '{interface_name}'."
            ) from err
        return match

    if search(r"type AP", output):
        raise utils.InterfaceError(
            f"Wireless interface '{interface_name}' is currently used as an access point. Cannot collect metrics."
        )

    out_interface_name = parse_metric(output, r"(?<=Interface )(.*)")
    interface_mac_address = parse_metric(output, r"(?<=addr )(.*)")

    channel_str = parse_metric(output, r"(?<=channel )(.*)(?=\,)").split(" ")
    channel = int(channel_str[0])
    channel_frequency = float(channel_str[1].replace("(", "")) * 1e6

    tx_power_str = float(parse_metric(output, r"(?<=txpower )(.*)(?= dBm)"))
    tx_power = float(tx_power_str)

    ssid = parse_metric(output, r"(?<=ssid )(.*)")

    cmd = f"iw {interface_name} link"
    output = utils.get_command_output(cmd, shell=True, timeout=10).replace("\t", " ")
    logging.debug(f"Output:\n{output}")

    ssid_mac_address = parse_metric(output, r"(?<=Connected to )(.*)(?= \()")

    try:
        signal_level = parse_metric(output, r"(?<=signal: )(.*)(?= dBm)").strip()
    except ParseError:
        logging.info(
            "Could not get signal strength from iw. Falling back to iwconfig..."
        )
        cmd = f"iwconfig {interface_name}"
        iwconfig = utils.get_command_output(cmd, shell=True, timeout=10).replace(
            "\t", " "
        )
        logging.debug(f"Output:\n{iwconfig}")

        signal_level = parse_metric(iwconfig, r"(?<=Signal level=)(.*)(?= dBm)")
    signal_strength = int(signal_level)

    metrics = WirelessMetrics(
        interface_name=out_interface_name,
        interface_mac_address=interface_mac_address,
        interface_ip_address=interface.ip_address,
        channel=channel,
        channel_frequency=channel_frequency,
        tx_power=tx_power,
        ssid=ssid,
        ssid_mac_address=ssid_mac_address,
        signal_strength=signal_strength,
    )

    logging.debug(f"Metrics: {metrics.model_dump()}")
    return metrics


#######################################################################################


class SocketConnection(BaseModel):
    """A model for a socket connection."""

    socket: int = Field(..., serialization_alias="socket_id")
    local_host: str
    local_port: int
    remote_host: str
    remote_port: int


class Iperf3Stream(BaseModel):
    """A model for an iperf3 stream."""

    socket: Optional[int] = Field(default=None, serialization_alias="socket_id")

    start: timedelta
    end: timedelta
    seconds: timedelta
    bytes: ByteSize
    bits_per_second: float
    sender: bool

    retransmits: Optional[int] = None

    rtt: Optional[int] = None
    rttvar: Optional[int] = None

    max_rtt: Optional[int] = None
    min_rtt: Optional[int] = None
    mean_rtt: Optional[int] = None

    jitter_ms: Optional[float] = None
    lost_packets: Optional[int] = None
    packets: Optional[int] = None
    lost_percent: Optional[float] = None

    out_of_order: Optional[int] = None

    def serialize_with_timestamp(self, basetime: datetime) -> dict[str, Any]:
        serialized = self.model_dump()
        serialized["start"] = basetime + serialized["start"]
        serialized["end"] = basetime + serialized["end"]
        return serialized


class Iperf3Interval(BaseModel):
    """A model for an iperf3 interval."""

    streams: list[Iperf3Stream]
    sum: Iperf3Stream

    @field_validator("streams", mode="before")
    def _parse_streams(cls, value: Sequence[dict[str, Any]]):
        if isinstance(value, Sequence) and all(
            isinstance(stream, dict) for stream in value
        ):
            ret_val: Sequence[Iperf3Stream] = TypeAdapter(
                list[Iperf3Stream]
            ).validate_python(value)
            return ret_val
        return value

    @field_validator("sum", mode="before")
    def _parse_sum(cls, value: dict[str, Any]):
        if isinstance(value, dict):
            ret_val: Iperf3Stream = TypeAdapter(Iperf3Stream).validate_python(value)
            return ret_val
        return value

    def serialize_with_timestamp(self, basetime: datetime) -> dict[str, Any]:
        serialized = {
            "streams": [
                stream.serialize_with_timestamp(basetime) for stream in self.streams
            ],
            "sum": self.sum.serialize_with_timestamp(basetime)
            if self.sum is not None
            else None,
        }
        return serialized


class Iperf3EndStreams(BaseModel):
    """A model for an iperf3 end streams."""

    sender: Optional[Iperf3Stream] = None
    receiver: Optional[Iperf3Stream] = None

    udp: Optional[Iperf3Stream] = None

    @model_validator(mode="after")  # type: ignore[arg-type]
    def _validate_model(self) -> "Iperf3EndStreams":
        if self.udp is None and self.sender is None and self.receiver is None:
            raise ValueError("Either sender and receiver or udp must be set.")
        if (self.sender is not None and self.receiver is None) or (
            self.sender is None and self.receiver is not None
        ):
            raise ValueError("Either sender and receiver must both be set.")
        if self.udp is not None and (
            self.sender is not None or self.receiver is not None
        ):
            raise ValueError("Sender or receiver cannot be set if udp is set.")
        return self

    def serialize_with_timestamp(self, basetime: datetime) -> dict[str, dict[str, Any]]:
        if self.udp is not None:
            return self.udp.serialize_with_timestamp(basetime)
        if self.sender is not None and self.receiver is not None:
            return {
                "sender": self.sender.serialize_with_timestamp(basetime),
                "receiver": self.receiver.serialize_with_timestamp(basetime),
            }
        return {}


class Iperf3End(BaseModel):
    """A model for iperf3 end metrics."""

    streams: Sequence[Iperf3EndStreams]
    sum: Optional[Iperf3Stream] = None
    sum_sent: Iperf3Stream
    sum_received: Iperf3Stream

    def serialize_with_timestamp(self, basetime: datetime) -> dict[str, Any]:
        serialized = {
            "streams": [
                stream.serialize_with_timestamp(basetime) for stream in self.streams
            ],
            "sum_sent": self.sum_sent.serialize_with_timestamp(basetime),
            "sum_received": self.sum_received.serialize_with_timestamp(basetime),
        }
        if self.sum is not None:
            serialized["sum"] = self.sum.serialize_with_timestamp(basetime)
        return serialized


class Iperf3Metrics(BaseModel):
    """A model for iperf3 metrics."""

    connection_information: Sequence[SocketConnection] = Field(
        ..., validation_alias=AliasPath("start", "connected")
    )
    timestamp: datetime = Field(
        ..., validation_alias=AliasPath("start", "timestamp", "timesecs")
    )
    settings: settings.Iperf3Settings
    intervals: Sequence[Iperf3Interval]
    end: Iperf3End

    @model_validator(mode="before")
    def _model_parser(cls, iperf3_dict: dict[str, Any]) -> dict[str, Any]:
        iperf3_dict["settings"] = settings.Iperf3Settings(
            host=iperf3_dict["start"]["connecting_to"]["host"],
            port=iperf3_dict["start"]["connecting_to"]["port"],
            protocol=iperf3_dict["start"]["test_start"]["protocol"].lower(),
            direction=iperf3_dict["start"]["test_start"]["reverse"],
            duration=iperf3_dict["start"]["test_start"]["duration"],
            streams=iperf3_dict["start"]["test_start"]["num_streams"],
            bind_address=iperf3_dict["start"]["test_start"]["bind_address"],
            retries=iperf3_dict["start"]["test_start"]["tries"],
        )
        return iperf3_dict

    @model_serializer
    def _serialize_model(self) -> dict[str, Any]:
        serialized = {
            "connection_information": [
                connection.model_dump() for connection in self.connection_information
            ],
            "timestamp": self.timestamp,
            "settings": self.settings.model_dump(),
            "intervals": [
                interval.serialize_with_timestamp(self.timestamp)
                for interval in self.intervals
            ],
            "end": self.end.serialize_with_timestamp(self.timestamp),
        }
        return serialized


def get_iperf3_metrics(config: settings.Iperf3Settings) -> Iperf3Metrics:
    """Run iperf3 and collect metrics.

    Args:
        config (settings.Iperf3Settings): The settings to use.

    Raises:
        ParseError: If no metrics could be retrieved from iperf3.
        utils.ExternalError: If iperf3 itself ran into an issue.

    Returns:
        Iperf3Metrics: The collected metrics.
    """
    if not isinstance(config, settings.Iperf3Settings):
        config = settings.Iperf3Settings(**config)

    client = iperf3.Client()
    client.server_hostname = config.host
    client.port = config.port
    client.protocol = config.protocol
    if config.bind_address:
        client.bind_address = config.bind_address
    client.reverse = config.reverse
    client.duration = config.duration
    client.num_streams = config.streams
    client.verbose = True
    logging.info(f"Running iperf3 with target '{config.url}'...")
    logging.debug(f"Config: {config.model_dump()}")

    iperf_results = None

    for t in range(config.retries):
        logging.debug(f"Try {t + 1} of {config.retries + 1}...")
        iperf_results = client.run()

        logging.debug(f"Output:\n{iperf_results}")

        if iperf_results and "error" in iperf_results.json:
            break

        try:
            json_data = iperf_results.json  # type: ignore[attr-defined]
            json_data["start"]["test_start"]["bind_address"] = config.bind_address
            json_data["start"]["test_start"]["tries"] = t
            return TypeAdapter(Iperf3Metrics).validate_python(json_data)
        except (AttributeError, KeyError, ValidationError):
            logging.exception(f"Unable to get metrics from iperf3.")
            iperf_results = None
            continue

    if not iperf_results:
        logging.exception(f"Unable to get metrics from iperf3.")
        raise ParseError(f"Unable to get metrics from iperf3.")

    logging.exception(
        f"External error while running iperf3: {iperf_results.json['error']}"
    )
    raise utils.ExternalError(
        f"External error while running iperf3: {iperf_results.json['error']}"
    )
