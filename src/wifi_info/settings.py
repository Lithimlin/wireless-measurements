import logging
from datetime import datetime
from functools import cached_property
from ipaddress import IPv4Address
from typing import Any, Iterable, Literal, Optional

from influxdb_client import InfluxDBClient  # type: ignore[import]
from influxdb_client.client.write_api import SYNCHRONOUS
from pandas import DataFrame
from pydantic import (
    AliasChoices,
    Field,
    NonNegativeInt,
    SecretStr,
    computed_field,
    field_validator,
    model_serializer,
)
from pydantic_settings import BaseSettings, SettingsConfigDict

from wifi_info import utils


class ServerSettings(BaseSettings):
    host: str = Field(..., exclude=True)
    port: int = Field(..., exclude=True)

    @computed_field  # type: ignore[misc]
    @property
    def url(self) -> str:
        return f"{self.host}:{self.port}"


class _InfluxDBConstants:
    QUERY_START = """from(bucket: "{bucket}")
    |> range(start: {start}, stop: {end})"""
    QUERY_END = (
        '|> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")'
    )


class InfluxDBSettings(ServerSettings):
    port: int = Field(default=8086, exclude=True)
    org: str
    bucket: str
    token: SecretStr = Field(
        ...,
        validation_alias=AliasChoices(
            "token", "push_token", "query_token", "bucket_token"
        ),
    )

    @cached_property
    def write_api(self) -> InfluxDBClient.write_api:
        return InfluxDBClient(
            url=self.url,
            token=self.token.get_secret_value(),
            org=self.org,
        ).write_api(write_options=SYNCHRONOUS)

    @cached_property
    def query_api(self) -> InfluxDBClient.query_api:
        return InfluxDBClient(
            url=self.url,
            token=self.token.get_secret_value(),
            org=self.org,
        ).query_api()

    def build_query(self, start: datetime, end: datetime, filters: str) -> str:
        def format_datetime(dt: datetime) -> str:
            string = dt.strftime("%Y-%m-%dT%H:%M:%S%z")
            if string[-3] != ":":
                string = f"{string[:-2]}:{string[-2:]}"
            return string

        return (
            _InfluxDBConstants.QUERY_START.format(
                bucket=self.bucket,
                start=format_datetime(start),
                end=format_datetime(end),
            )
            + filters
            + _InfluxDBConstants.QUERY_END
        )

    @staticmethod
    def build_filters(tag: str, values: Iterable[str]) -> str:
        searches = " or ".join([f'r["{tag}"] == "{value}"' for value in values])
        return f"|> filter(fn: (r) => {searches})" if searches != "" else ""

    @staticmethod
    def build_false_filters(tag: str, values: Iterable[str]) -> str:
        searches = " and ".join([f'r["{tag}"] != "{value}"' for value in values])
        return f"|> filter(fn: (r) => {searches})" if searches != "" else ""

    @staticmethod
    def build_drop_columns(columns: Iterable[str]) -> str:
        columns = map(lambda s: '"' + s + '"', columns)
        return f"|> drop(columns: [{', '.join(columns)}])"

    def query_data_frame(self, query: str) -> DataFrame | list[DataFrame]:
        return self.query_api.query_data_frame(query, self.org)

    def write(self, record: Any, **kwargs) -> None:
        self.write_api.write(org=self.org, bucket=self.bucket, record=record, **kwargs)
        self.write_api.flush()


class Iperf3Settings(ServerSettings):
    port: int = Field(default=5201, exclude=True)
    protocol: Literal["tcp", "udp"] = "tcp"
    direction: Literal["download", "upload"] = "download"
    duration: int = 30
    streams: int = 5
    bind_address: Optional[str] = None
    retries: NonNegativeInt = 2

    @property
    def reverse(self) -> bool:
        return self.direction == "download"

    @field_validator("bind_address")
    def validate_bind_address(cls, value: Optional[str]) -> Optional[str]:
        if not value:
            return value
        if value == "*" or utils.verify_ip_address(value):
            return value
        raise ValueError(f"Invalid IP address: {value}")

    @field_validator("direction", mode="before")
    def validate_direction(cls, value: str | bool | int) -> str:
        logging.debug(f"Validation direction: {value}...")
        if isinstance(value, bool) or isinstance(value, int):
            return "download" if value else "upload"
        if isinstance(value, str):
            if value in ["download", "upload"]:
                return value
            raise ValueError(f"Invalid direction: {value}")
        raise TypeError(f"Invalid type: {type(value)}")


class InterfaceSettings(BaseSettings):
    name: str

    @field_validator("name")
    def validate_name(cls, value: str) -> str:
        utils.verify_network_interface(value)
        return value

    @computed_field  # type: ignore[misc]
    @property
    def ip_address(self) -> IPv4Address:
        return utils.get_interface_ipv4_address(self.name)
