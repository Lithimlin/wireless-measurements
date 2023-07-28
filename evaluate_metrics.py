#!/usr/bin/env python3
import logging
from datetime import datetime
from typing import Optional, Sequence

import geopandas as gpd  # type: ignore[import]
import matplotlib.pyplot as plt  # type: ignore[import]
import numpy as np
import pandas as pd  # type: ignore[import]
import seaborn as sns  # type: ignore[import]
from influxdb_client import InfluxDBClient  # type: ignore[import]
from mpl_toolkits.mplot3d.axes3d import Axes3D  # type: ignore[import]
from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from scipy.interpolate import griddata  # type: ignore[import]

from wifi_info.settings import InfluxDBSettings


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
    gps_fields: dict[str, str] = {
        "lat": "rtk_lat",
        "lon": "rtk_lon",
        "alt": "rtk_height_from_sea",
        # "alt": "height_fusion",
    }
    extra_uav_fields: dict[str, str] = {
        "vel_x": "rtk_velocity_x",
        "vel_y": "rtk_velocity_y",
        "vel_z": "rtk_velocity_z",
        "status": "status_flight",
    }
    iperf_fields: dict[str, str] = {
        "bps": "bits_per_second",
    }
    stream_types: list[str] = ["interval stream"]
    plot_target: str = "bps"
    crs_target: Optional[str] = Field(default=None, required=False)

    @field_validator("gps_fields")
    @classmethod
    def validate_gps_fields(cls, v: dict[str, str]) -> dict[str, str]:
        if not all(k in v.keys() for k in ["lat", "lon", "alt"]):
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


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


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


def get_iperf_frame(
    settings: EvaluationSettings, query_api: InfluxDBClient.query_api
) -> pd.DataFrame:
    iperf_query = build_iperf_query(settings)

    df_iperf = query_api.query_data_frame(iperf_query, org=settings.influxDB.org)
    df_iperf = process_dataframe(df_iperf)
    df_iperf = df_iperf.rename(columns=invert_dict(settings.iperf_fields))

    return df_iperf


def get_uav_frame(
    settings: EvaluationSettings, query_api: InfluxDBClient.query_api
) -> pd.DataFrame:
    uav_query = build_uav_query(settings)

    df_uav = query_api.query_data_frame(uav_query, org=settings.influxDB.org)
    df_uav = process_dataframe(df_uav)
    df_uav = df_uav.rename(
        columns=invert_dict(merge_dicts(settings.gps_fields, settings.extra_uav_fields))
    )

    return df_uav


def get_geodata(settings: EvaluationSettings) -> gpd.GeoDataFrame:
    # Get InfluxDB client and query API
    client = InfluxDBClient(
        url=settings.influxDB.url,
        token=settings.influxDB.token.get_secret_value(),
        org=settings.influxDB.org,
    )
    query_api = client.query_api()

    # get DataFrames
    df_iperf = get_iperf_frame(settings, query_api)
    df_uav = get_uav_frame(settings, query_api)

    # merge DataFrames
    df = df_iperf.join(df_uav)

    # fix zero-altitudes
    df["alt"] = df["alt"].replace(0, np.nan).interpolate(method="linear")
    # normalize to zero
    if "sea" in settings.gps_fields["alt"]:
        df["alt"] = df["alt"] - df["alt"].min()

    # create GeoDataFrame
    gdf = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df["lon"], df["lat"], df["alt"], crs="WGS84")
    )
    # convert if requested
    if settings.crs_target is not None:
        gdf = gdf.to_crs(settings.crs_target)

    return gdf


def get_interpolated_geoframe(
    geometry: gpd.GeoSeries,
    values: gpd.GeoSeries,
    nx: int = 15,
    ny: int = 15,
    nz: int = 10,
    method: str = "linear",
) -> gpd.GeoDataFrame:
    points = (
        geometry.x,
        geometry.y,
        geometry.z,
    )
    X, Y, Z = np.meshgrid(
        np.linspace(geometry.x.min(), geometry.x.max(), nx),
        np.linspace(geometry.y.min(), geometry.y.max(), ny),
        np.linspace(geometry.z.min(), geometry.z.max(), nz),
    )
    V = griddata(
        points,
        values,
        (X.flatten(), Y.flatten(), Z.flatten()),
        method=method,
    )
    gdf = gpd.GeoDataFrame(
        {"V": V},
        geometry=gpd.points_from_xy(
            X.flatten(), Y.flatten(), Z.flatten(), crs=geometry.crs
        ),
    )

    return gdf


def plot_path(
    ax: Axes3D, geometry: gpd.GeoSeries, color: str = "grey", linestyle: str = ":"
):
    ax.plot3D(geometry.x, geometry.y, geometry.z, c=color, ls=linestyle)


def plot_path_stem(
    ax: Axes3D, geometry: gpd.GeoSeries, markerfmt: str = ".", linefmt: str = "C7:"
):
    ax.stem(
        geometry.x,
        geometry.y,
        geometry.z,
        markerfmt=markerfmt,
        linefmt=linefmt,
    )


def plot_4d_data(
    ax: Axes3D, geometry: gpd.GeoSeries, values: gpd.GeoSeries, cmap: str = "viridis"
):
    ax.scatter(
        geometry.x,
        geometry.y,
        geometry.z,
        c=values,
        cmap=cmap,
    )


def plot_interpolated_data(
    ax: Axes3D,
    geometry: gpd.GeoSeries,
    values: gpd.GeoSeries,
    nx: int = 15,
    ny: int = 15,
    nz: int = 10,
    method: str = "linear",
    cmap: str = "viridis",
):
    gdf = get_interpolated_geoframe(
        geometry, values, nx=nx, ny=ny, nz=nz, method=method
    )
    X, Y, Z = gdf.geometry.x, gdf.geometry.y, gdf.geometry.z
    ax.scatter(X, Y, Z, c=gdf.V, cmap=cmap)


def plot_4d_contours(
    ax: Axes3D,
    geometry: gpd.GeoSeries,
    values: gpd.GeoSeries,
    n_contours: int = 5,
    nx: int = 15,
    ny: int = 15,
    nz: int = 10,
    cmap: str = "viridis",
    method: str = "nearest",
    limit_contours: Optional[Sequence[int]] = None,
):
    if limit_contours is None:
        limit_contours = range(n_contours)
    gdf = get_interpolated_geoframe(
        geometry, values, nx=nx, ny=ny, nz=nz, method=method
    )
    colors = sns.color_palette(cmap, n_colors=n_contours)
    gdf["cat"] = pd.cut(gdf.V, bins=n_contours)
    for i, (color, cat) in enumerate(zip(colors, gdf.cat.cat.categories)):
        if i not in limit_contours:
            continue
        sub_df = gdf.loc[gdf.cat == cat, :]
        X, Y = sub_df.geometry.x.values, sub_df.geometry.y.values
        Z = sub_df.geometry.z.values
        ax.plot_trisurf(X, Y, Z, color=color, alpha=0.2)


#########################################################################################################################


def main():
    setup_logging()
    # Read settings
    eval_settings = EvaluationSettings()

    # get GeoDataFrame
    gdf = get_geodata(eval_settings)
    logging.info("Initial GeoDataFrame:")
    logging.info(gdf.head())

    # plot data
    sns.set_style("darkgrid")
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(10, 10))

    # Plot UAV path as stem
    # plot_path_stem(ax, gdf.geometry)

    # Plot UAV path
    # plot_path(ax, gdf.geometry)

    # Plot target data
    # plot_4d_data(ax, gdf.geometry, gdf[eval_settings.plot_target])

    # Interpolate data
    plot_interpolated_data(
        ax,
        gdf.geometry,
        gdf[eval_settings.plot_target],
        nx=50,
        ny=50,
        nz=20,
        method="linear",
    )

    # Plot contours
    # plot_4d_contours(
    #     ax,
    #     gdf.geometry,
    #     gdf[eval_settings.plot_target],
    #     n_contours=2,
    #     nx=50,
    #     ny=50,
    #     nz=20,
    #     limit_contours=None,
    #     method="linear",
    # )

    plt.show()


if __name__ == "__main__":
    main()
