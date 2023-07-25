#!/usr/bin/env python3
from datetime import datetime
from typing import Optional

import geopandas as gpd  # type: ignore[import]
import matplotlib.pyplot as plt  # type: ignore[import]
import numpy as np
import pandas as pd  # type: ignore[import]
import seaborn as sn  # type: ignore[import]
from influxdb_client import InfluxDBClient  # type: ignore[import]
from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from wifi_info.settings import InfluxDBSettings

GPS_FIELDS = {
    "lat": "rtk_lat",
    "lon": "rtk_lon",
    "alt": "rtk_height_from_sea",
    # "alt": "height_fusion",
}
EXTRA_UAV_FIELDS = {
    "vel_x": "rtk_velocity_x",
    "vel_y": "rtk_velocity_y",
    "vel_z": "rtk_velocity_z",
    "status": "status_flight",
}
IPERF_FIELDS = {
    "bps": "bits_per_second",
}
PLOT_TARGET = "bps"
STREAM_TYPES = ["interval stream"]


class EvaluationSettings(BaseSettings):
    """
    Settings for the evaluation module.
    """

    model_config = SettingsConfigDict(
        env_file=(".env", ".env.eval"),
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        env_prefix="EVAL_",
        extra="ignore",
    )
    start: datetime
    end: datetime
    influxDB: InfluxDBSettings
    gps_fields: dict[str, str] = GPS_FIELDS
    extra_uav_fields: dict[str, str] = EXTRA_UAV_FIELDS
    iperf_fields: dict[str, str] = IPERF_FIELDS
    stream_types: list[str] = STREAM_TYPES
    plot_target: str = PLOT_TARGET
    crs_target: Optional[str] = Field(default=None, required=False)

    @field_validator("gps_fields")
    @classmethod
    def validate_gps_fields(cls, v: dict[str, str]) -> dict[str, str]:
        if not all(k in GPS_FIELDS for k in ["lat", "lon", "alt"]):
            raise ValueError("gps_fields must contain lat, lon and alt fields")
        return v

    @field_validator("iperf_fields")
    @classmethod
    def validate_iperf_fields(cls, v: dict[str, str]) -> dict[str, str]:
        if len(v) == 0:
            raise ValueError("iperf_fields must contain at least one field")
        return v

    @model_validator(mode="after")
    def validate_settings(self) -> "EvaluationSettings":
        if (
            not self.plot_target in self.iperf_fields.keys()
            and not self.plot_target in self.extra_uav_fields.keys()
        ):
            raise ValueError(
                "plot_target must be a key of iperf_fields or extra_uav_fields"
            )
        return self


QUERY_START = """from(bucket: "{bucket}")
|> range(start: {start}, stop: {end})"""
QUERY_END = '|> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")'
FIELD_TEMPLATE = 'r["_field"] == "{0}"'
TYPE_TEMPLATE = 'r["type"] == "{0}"'


def floor_time(dt: datetime) -> datetime:
    return dt.replace(microsecond=0)


def invert_dict(d: dict) -> dict:
    return {v: k for k, v in d.items()}


def merge_dicts(d1: dict, d2: dict) -> dict:
    return {**d1, **d2}


def build_iperf_query(settings: EvaluationSettings) -> str:
    fields = " or ".join(
        [FIELD_TEMPLATE.format(field) for field in settings.iperf_fields.values()]
    )
    types = " or ".join([TYPE_TEMPLATE.format(tpe) for tpe in settings.stream_types])
    return (
        QUERY_START.format(
            bucket=settings.influxDB.bucket,
            start=settings.start.isoformat(),
            end=settings.end.isoformat(),
        )
        + f"""|> filter(fn: (r) => r["_measurement"] == "iperf3")
        |> filter(fn: (r) => {fields})
        |> filter(fn: (r) => {types})"""
        + QUERY_END
    )


def build_uav_query(settings: EvaluationSettings) -> str:
    fields = " or ".join(
        [
            FIELD_TEMPLATE.format(field)
            for field in merge_dicts(
                settings.gps_fields, settings.extra_uav_fields
            ).values()
        ]
    )
    return (
        QUERY_START.format(
            bucket=settings.influxDB.bucket,
            start=settings.start.isoformat(),
            end=settings.end.isoformat(),
        )
        + f"""|> filter(fn: (r) => r["_measurement"] == "drone_metrics")
        |> filter(fn: (r) => {fields})"""
        + QUERY_END
    )


def process_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        raise ValueError(
            "DataFrame is empty. Perhaps incorrect filters were specified?"
        )
    df["_time"] = df["_time"].apply(floor_time)
    df = df.drop_duplicates(subset="_time")
    df = df.set_index("_time").drop(
        ["result", "table", "_start", "_stop", "_measurement"], axis=1
    )
    return df


def main():
    # Read settings
    eval_settings = EvaluationSettings()

    # Get InfluxDB client and query API
    client = InfluxDBClient(
        url=eval_settings.influxDB.url,
        token=eval_settings.influxDB.token.get_secret_value(),
        org=eval_settings.influxDB.org,
    )
    query_api = client.query_api()

    # get iperf data
    iperf_query = build_iperf_query(eval_settings)
    df_iperf = query_api.query_data_frame(iperf_query, org=eval_settings.influxDB.org)
    df_iperf = process_dataframe(df_iperf)
    df_iperf = df_iperf.rename(columns=invert_dict(eval_settings.iperf_fields))

    # get uav data
    uav_query = build_uav_query(eval_settings)
    df_uav = query_api.query_data_frame(uav_query, org=eval_settings.influxDB.org)
    df_uav = process_dataframe(df_uav)
    # rename fields
    df_uav = df_uav.rename(
        columns=invert_dict(
            merge_dicts(eval_settings.gps_fields, eval_settings.extra_uav_fields)
        )
    )

    # merge DataFrames
    df = df_iperf.join(df_uav)
    # filter data
    # df = df.loc[(df["status"] == 2) & (df["type"] == "interval sum")]
    # fix zero-altitudes
    df["alt"] = df["alt"].replace(0, np.nan).interpolate(method="linear")
    # normalize to zero
    if "sea" in eval_settings.gps_fields["alt"]:
        df["alt"] = df["alt"] - df["alt"].min()

    # create GeoDataFrame
    gdf = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df["lon"], df["lat"], df["alt"], crs="WGS84")
    )
    # convert if requested
    if eval_settings.crs_target is not None:
        gdf = gdf.to_crs(eval_settings.crs_target)

    print(gdf.head())

    # plot data
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(10, 10))
    ax.scatter3D(
        gdf["geometry"].x,
        gdf["geometry"].y,
        gdf["geometry"].z,
        c=gdf[eval_settings.plot_target],
        cmap="viridis",
    )
    plt.show()


if __name__ == "__main__":
    main()
