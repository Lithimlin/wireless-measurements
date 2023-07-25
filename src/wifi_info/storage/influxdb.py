import logging
from datetime import datetime, timedelta
from socket import gethostname
from typing import Any, Sequence

from influxdb_client import InfluxDBClient, Point  # type: ignore[import]
from influxdb_client.client.write_api import SYNCHRONOUS  # type: ignore

from wifi_info.settings import InfluxDBSettings
from wifi_info.wireless import Iperf3Metrics, WirelessMetrics


class InfluxDBStorage:
    def __init__(self, settings: InfluxDBSettings):
        self.settings = settings
        self.client = InfluxDBClient(
            url=settings.url,
            token=settings.token.get_secret_value(),
            org=settings.org,
        )
        self.write_api = self.client.write_api(write_options=SYNCHRONOUS)
        logging.info(f"{gethostname()} - InfluxDB opened")

    def __del__(self):
        self.write_api.close()
        self.client.close()
        logging.info(f"{gethostname()} - InfluxDB closed")

    def write_data(self, data: Iperf3Metrics | WirelessMetrics):
        if isinstance(data, Iperf3Metrics):
            self.write_iperf_metrics(data)
        elif isinstance(data, WirelessMetrics):
            self.write_wireless_metrics(data)

    def write_iperf_metrics(self, data: Iperf3Metrics):
        points = _convert_iperf_metrics_to_points(data)
        (p.tag("host", gethostname()) for p in points)
        self.write_api.write(bucket=self.settings.bucket, record=points)

    def write_wireless_metrics(self, data: WirelessMetrics):
        point = _convert_wireless_metrics_to_points(data)
        point.tag("host", gethostname())
        self.write_api.write(bucket=self.settings.bucket, record=point)


def _convert_wireless_metrics_to_points(data: WirelessMetrics) -> Point:
    point = Point.from_dict(
        {
            "measurement": "wireless",
            "fields": _check_dict(data.model_dump()),
        }
    )
    return point


def _convert_iperf_metrics_to_points(data: Iperf3Metrics) -> list[Point]:
    points: list[Point] = []

    basetime = data.timestamp
    settings_dump = _check_dict(data.settings.model_dump())

    for connection in data.connection_information:
        connection_dump = settings_dump | _check_dict(connection.model_dump())
        for interval in data.intervals:
            for i_stream in filter(
                lambda s: s.socket == connection.socket, interval.streams
            ):
                point = Point.from_dict(
                    {
                        "measurement": "iperf3",
                        "tags": {"type": "interval stream"},
                        "time": basetime + i_stream.start,
                        "fields": connection_dump
                        | _check_dict(i_stream.serialize_with_timestamp(basetime)),
                    }
                )
                points.append(point)
            point = Point.from_dict(
                {
                    "measurement": "iperf3",
                    "tags": {"type": "interval sum"},
                    "time": basetime + interval.sum.start,
                    "fields": connection_dump
                    | _check_dict(interval.sum.serialize_with_timestamp(basetime)),
                },
            )
            points.append(point)
        for e_stream in data.end.streams:
            if e_stream.udp is not None and e_stream.udp.socket == connection.socket:
                point = Point.from_dict(
                    {
                        "measurement": "iperf3",
                        "tags": {"type": "end stream"},
                        "time": basetime + e_stream.udp.start,
                        "fields": connection_dump
                        | _check_dict(e_stream.udp.serialize_with_timestamp(basetime)),
                    }
                )
                points.append(point)
            if (
                e_stream.receiver is not None
                and e_stream.receiver.socket == connection.socket
            ):
                point = Point.from_dict(
                    {
                        "measurement": "iperf3",
                        "tags": {"type": "end stream", "side": "receiver"},
                        "time": basetime + e_stream.receiver.start,
                        "fields": connection_dump
                        | _check_dict(
                            e_stream.receiver.serialize_with_timestamp(basetime)
                        ),
                    }
                )
                points.append(point)
            if (
                e_stream.sender is not None
                and e_stream.sender.socket == connection.socket
            ):
                point = Point.from_dict(
                    {
                        "measurement": "iperf3",
                        "tags": {"type": "end stream", "side": "sender"},
                        "time": basetime + e_stream.sender.start,
                        "fields": connection_dump
                        | _check_dict(
                            e_stream.sender.serialize_with_timestamp(basetime)
                        ),
                    }
                )
                points.append(point)

    connection_dump = settings_dump | _check_dict(connection.model_dump())
    connection_dump.pop("socket")

    if data.end.sum is not None:
        point = Point.from_dict(
            {
                "measurement": "iperf3",
                "tags": {"type": "end sum"},
                "time": basetime + data.end.sum.start,
                "fields": connection_dump
                | _check_dict(data.end.sum.serialize_with_timestamp(basetime)),
            }
        )
        points.append(point)
    point = Point.from_dict(
        {
            "measurement": "iperf3",
            "tags": {"type": "end received"},
            "time": basetime + data.end.sum_received.start,
            "fields": connection_dump
            | _check_dict(data.end.sum_received.serialize_with_timestamp(basetime)),
        }
    )
    points.append(point)
    point = Point.from_dict(
        {
            "measurement": "iperf3",
            "tags": {"type": "end sent"},
            "time": basetime + data.end.sum_sent.start,
            "fields": connection_dump
            | _check_dict(data.end.sum_sent.serialize_with_timestamp(basetime)),
        }
    )
    points.append(point)

    return points


def _check_dict(data: dict[str, Any]) -> dict[str, Any]:
    data_cpy = data.copy()
    for k, v in data.items():
        if v is None:
            data_cpy.pop(k, None)
        if isinstance(v, datetime):
            data_cpy[k] = int(v.strftime("%s"))
        if isinstance(v, timedelta):
            data_cpy[k] = v.seconds * 1e6 + v.microseconds
    return data_cpy
