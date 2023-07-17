import logging
import time
from datetime import timedelta
from multiprocessing import Process

import iperf3  # type: ignore[import]
import numpy as np
import pytest

from wifi_info import utils, wireless
from wifi_info.settings import Iperf3Settings


def check_timestamps_close(ts_a, ts_b, /, tol=timedelta(milliseconds=1)):
    return abs(ts_a - ts_b) < tol


@pytest.fixture()
def iperf_server():
    server = iperf3.Server()
    server.bind_address = "127.0.0.1"
    server.port = 5201
    server.verbose = False

    proc = Process(target=server.run, name="iperf3-server")
    proc.start()
    while not proc.is_alive():
        time.sleep(0.1)

    yield proc
    proc.join(2)
    if proc.is_alive():
        proc.terminate()


@pytest.fixture()
def mock_client(monkeypatch, request):
    def run(self):
        del self  # unused
        return request.param

    monkeypatch.setattr(iperf3.Client, "run", run)


@pytest.mark.parametrize(
    "ip_address, port, expected",
    [
        ("127.0.0.1", 5201, True),
        ("127.0.0.2", 5201, False),
        ("127.0.0.1", 5202, False),
    ],
)
def test_verify_connection(iperf_server, ip_address, port, expected):
    connection = utils.verify_connection(ip_address, port)
    assert connection is expected


def test_get_iperf3_metrics_retries():
    with pytest.raises(ValueError):
        wireless.get_iperf3_metrics(Iperf3Settings(host="127.0.0.2", retries=-1))


@pytest.mark.parametrize(
    "ip_address, port",
    [
        ("127.0.0.1", 5202),
        ("127.0.0.2", 5201),
    ],
)
def test_get_iperf3_metrics_error(ip_address, port):
    with pytest.raises(utils.ExternalError):
        wireless.get_iperf3_metrics(
            Iperf3Settings(host=ip_address, port=port, duration=1)
        )


@pytest.mark.parametrize(
    "protocol",
    [
        ("tcp"),
        ("udp"),
    ],
)
def test_get_iperf3_metrics_settings(iperf_server, protocol):
    in_settings = Iperf3Settings(host="127.0.0.1", duration=1, protocol=protocol)
    metrics = wireless.get_iperf3_metrics(in_settings)

    assert metrics.settings.host == in_settings.host
    assert metrics.settings.port == in_settings.port
    assert metrics.settings.protocol == protocol
    assert metrics.settings.duration == in_settings.duration
    assert metrics.settings.direction == in_settings.direction
    assert metrics.settings.streams == in_settings.streams


@pytest.mark.parametrize(
    "download, num_streams",
    [
        ("download", 1),
        ("upload", 1),
        ("download", 3),
        ("upload", 3),
    ],
)
def test_get_iperf3_metrics_output_tcp_download(iperf_server, download, num_streams):
    metrics = wireless.get_iperf3_metrics(
        Iperf3Settings(
            host="127.0.0.1", duration=2, direction=download, streams=num_streams
        )
    )

    is_sender = download == "upload"

    assert all(
        stream.sender == is_sender
        for interval in metrics.intervals
        for stream in interval.streams
    )
    assert all(interval.sum.sender == is_sender for interval in metrics.intervals)
    assert all(
        (stream.sender is not None and stream.sender.sender == is_sender)
        and (stream.receiver is not None and stream.receiver.sender == is_sender)
        for stream in metrics.end.streams
    )
    assert metrics.end.sum_sent.sender == is_sender
    assert metrics.end.sum_received.sender == is_sender


@pytest.mark.parametrize(
    "download, num_streams",
    [
        ("download", 1),
        ("upload", 1),
        ("download", 3),
        ("upload", 3),
    ],
)
def test_get_iperf3_metrics_output_udp_download(iperf_server, download, num_streams):
    metrics = wireless.get_iperf3_metrics(
        Iperf3Settings(
            host="127.0.0.1",
            duration=2,
            direction=download,
            streams=num_streams,
            protocol="udp",
        )
    )

    is_sender = download == "upload"

    assert all(
        stream.sender == is_sender
        for interval in metrics.intervals
        for stream in interval.streams
    )
    assert all(interval.sum.sender == is_sender for interval in metrics.intervals)
    assert all(
        stream.udp is not None and stream.udp.sender == is_sender
        for stream in metrics.end.streams
    )
    assert metrics.end.sum is not None and metrics.end.sum.sender == is_sender
    assert metrics.end.sum_sent.sender == True
    assert metrics.end.sum_received.sender == False


@pytest.mark.parametrize(
    "num_streams",
    [
        (1),
        (5),
        (10),
    ],
)
def test_get_iperf3_metrics_output_num_streams(iperf_server, num_streams):
    metrics = wireless.get_iperf3_metrics(
        Iperf3Settings(host="127.0.0.1", duration=2, streams=num_streams)
    )

    assert metrics.settings.streams == num_streams
    assert len(metrics.connection_information) == num_streams
    assert len(metrics.end.streams) == num_streams
    assert all(len(interval.streams) == num_streams for interval in metrics.intervals)


@pytest.mark.parametrize(
    "duration, protocol",
    [
        (1, "tcp"),
        (5, "tcp"),
        (5, "udp"),
        (10, "udp"),
    ],
)
def test_get_iperf3_metrics_output_duration(iperf_server, duration, protocol):
    metrics = wireless.get_iperf3_metrics(
        Iperf3Settings(host="127.0.0.1", duration=duration, protocol=protocol)
    )

    assert metrics.settings.duration == duration
    assert metrics.end.sum is None or check_timestamps_close(
        metrics.end.sum.end, timedelta(seconds=duration)
    )
    assert check_timestamps_close(metrics.end.sum_sent.end, timedelta(seconds=duration))
    assert check_timestamps_close(
        metrics.end.sum_received.end, timedelta(seconds=duration)
    )


class MockResult:
    def __init__(self, json: dict):
        self.json = json


@pytest.mark.parametrize(
    "mock_client",
    [
        None,
        MockResult({"start": {}}),
        MockResult({"end": {}}),
    ],
    indirect=True,
)
def test_get_iperf3_metrics_parse_error(mock_client):
    with pytest.raises(wireless.ParseError):
        wireless.get_iperf3_metrics(Iperf3Settings(host="127.0.0.1", duration=1))
