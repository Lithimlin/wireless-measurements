import logging
from ipaddress import IPv4Address
from typing import Literal, Optional

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
