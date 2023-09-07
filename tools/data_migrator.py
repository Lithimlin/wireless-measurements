from datetime import datetime
from typing import Iterable

import module_logging
import pandas as pd
from pydantic import Field
from pydantic_json_source import JsonConfigSettingsSource
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

from wifi_info.settings import InfluxDBSettings

MODULE_LOGGER = module_logging.get_logger(module_logging.logging.INFO)


class MigrationSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=("../.env", "../.env.eval", ".env", ".env.eval"),
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        env_prefix="EVAL_",
        json_file="migrator.json",
        json_file_encoding="utf-8",
        extra="ignore",
    )  # type: ignore[typeddict-unknown-key]

    start: datetime
    end: datetime

    source: InfluxDBSettings
    target: InfluxDBSettings

    measurements: list[str] = ["drone_metrics", "iperf3", "wireless"]

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            JsonConfigSettingsSource(settings_cls),
            env_settings,
            file_secret_settings,
        )

    def get_data(self, measurement: str) -> pd.DataFrame:
        MODULE_LOGGER.info("Getting data")
        query = self.source.build_query(
            self.start,
            self.end,
            InfluxDBSettings.build_filters("_measurement", [measurement]),
        )
        MODULE_LOGGER.debug("Query: %s", query)
        return self.source.query_data_frame(query)

    def write_data(self, data: pd.DataFrame, **kwargs) -> None:
        MODULE_LOGGER.info("Writing data")
        self.target.write(data, **kwargs)

    def migrate_measurement(self, measurement: str) -> None:
        data = self.get_data(measurement)
        kwargs = dict(
            data_frame_measurement_name=measurement,
            data_frame_tag_columns=["side", "type", "hostname", "host"],
            data_frame_timestamp_column="_time",
        )

        def get_description(data: pd.DataFrame) -> pd.DataFrame:
            types = data.dtypes
            counts = data.count()
            return pd.concat(
                (types, counts),
                axis=1,
                join="inner",
            ).rename(columns={0: "type", 1: "count"})

        MODULE_LOGGER.debug("Read data:")
        if isinstance(data, list):
            for datum in data:
                if datum.empty:
                    MODULE_LOGGER.debug("\n%s", datum)
                    MODULE_LOGGER.warning("Datum is empty, but there may be more.")
                    continue
                else:
                    MODULE_LOGGER.debug("\n%s", get_description(datum))
                    MODULE_LOGGER.debug("\n%s", datum.head(2))
                    self.write_data(datum, **kwargs)
        else:
            if data.empty:
                MODULE_LOGGER.debug("\n%s", data)
                MODULE_LOGGER.warning("Data is empty.")
                return
            else:
                MODULE_LOGGER.debug("\n%s", get_description(data))
                MODULE_LOGGER.debug("\n%s", data.head(2))
                self.write_data(data, **kwargs)

    def migrate_data(self) -> None:
        for measurement in self.measurements:
            MODULE_LOGGER.info("Migrating '%s' data", measurement)
            self.migrate_measurement(measurement)


def main():
    MODULE_LOGGER.info("Starting migration")
    migrator = MigrationSettings()
    MODULE_LOGGER.debug("Settings: %s", migrator)
    migrator.migrate_data()


if __name__ == "__main__":
    main()
