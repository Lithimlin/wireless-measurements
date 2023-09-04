#!/usr/bin/env python3
import json
import logging
import sys
from datetime import datetime, timedelta
from enum import StrEnum, auto
from functools import cached_property
from pathlib import Path
from typing import Annotated, Any, ClassVar, Optional, Sequence

import geopandas as gpd  # type: ignore[import]
import matplotlib.pyplot as plt  # type: ignore[import]
import numpy as np
import pandas as pd  # type: ignore[import]
import seaborn as sns  # type: ignore[import]
from influxdb_client import InfluxDBClient  # type: ignore[import]
from mpl_toolkits.mplot3d.axes3d import Axes3D  # type: ignore[import]
from pydantic import (
    Field,
    PositiveInt,
    TypeAdapter,
    computed_field,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict
from scipy.interpolate import griddata  # type: ignore[import]

from wifi_info.settings import InfluxDBSettings


def get_logger(level=logging.DEBUG) -> logging.Logger:
    """get a logger for the module with the given level

    Args:
        level (optional): Set the logging level of the logger. Defaults to logging.DEBUG.

    Returns:
        logging.Logger: The logger for the module.
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(level)

    channel = logging.StreamHandler()
    channel.setLevel(level)

    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    channel.setFormatter(formatter)
    logger.addHandler(channel)
    return logger


MODULE_LOGGER = get_logger(logging.DEBUG)


def floor_time(time: datetime) -> datetime:
    return time.replace(microsecond=0)


def invert_dict(dict_val: dict) -> dict:
    return {v: k for k, v in dict_val.items()}


def merge_dicts(dict_1: dict, dict_2: dict) -> dict:
    return {**dict_1, **dict_2}


class InfluxDBConstants:
    QUERY_START = """from(bucket: "{bucket}")
    |> range(start: {start}, stop: {end})"""
    QUERY_END = (
        '|> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")'
    )


def build_influx_query(
    bucket: str, start: datetime, end: datetime, filters: str
) -> str:
    return (
        InfluxDBConstants.QUERY_START.format(
            bucket=bucket,
            start=start.isoformat(),
            end=end.isoformat(),
        )
        + filters
        + InfluxDBConstants.QUERY_END
    )


def build_influx_filters(tag: str, values: Sequence[str]) -> str:
    searches = " or ".join([f'r["{tag}"] == "{value}"' for value in values])
    return f"|> filter(fn: (r) => {searches})"


class PlotSettings(BaseSettings):
    DEFAULT_DICT: ClassVar[dict[str, Any]] = dict()
    kwargs: dict[str, Any] = Field(default={}, required=False)

    @field_validator("kwargs")
    @classmethod
    def _validate_kwargs(cls, kwargs: dict[str, Any]) -> dict[str, Any]:
        return merge_dicts(cls.DEFAULT_DICT, kwargs)

    def plot(self, *args, **kwargs) -> None:
        raise NotImplementedError(
            "The plotter of the base class was called but is not intended to be used"
        )


class AxesSettings(BaseSettings):
    title: Optional[str] = Field(default=None, required=False)
    xlabel: Optional[str] = Field(default=None, required=False)
    ylabel: Optional[str] = Field(default=None, required=False)

    kwargs: dict[str, Any] = Field(
        default={}, required=False, description="additional kwargs for subplots"
    )

    def validate_targets(self, *args, **kwargs) -> bool:
        raise NotImplementedError(
            "The validator of the base class was called but is not intended to be used"
        )

    def plot(self, *args, **kwargs) -> None:
        raise NotImplementedError(
            "The plotter of the base class was called but is not intended to be used"
        )


class Plot2DType(StrEnum):
    LINE = auto()


class Plot2DSettings(PlotSettings):
    def plot(self, *args, **kwargs) -> None:
        raise NotImplementedError(
            "The plotter of the 2D base class was called but is not intended to be used"
        )


class Line2DSettings(Plot2DSettings):
    def plot(self, ax: plt.Axes, data: gpd.GeoSeries) -> None:
        ax.plot(data, **self.kwargs)


Plots2DUnion = Annotated[Line2DSettings, "Union of available 2D plots"]


class Axes2DDefaults(AxesSettings):
    targets: list[str] = ["bps"]

    plots: dict[Plot2DType, Plots2DUnion] = {Plot2DType.LINE: Line2DSettings()}

    @field_validator("plots")
    @classmethod
    def _validate_plots(
        cls, data: dict[str, dict] | dict[Plot2DType, Plots2DUnion]
    ) -> dict[Plot2DType, Plots2DUnion]:
        ret_data: dict[Plot2DType, Plots2DUnion] = {}

        for plot_type, plot_settings in data.items():
            if plot_type not in Plot2DType:
                raise ValueError(
                    f"Plot type '{plot_type}' is not a valid plot type. "
                    f"Must be one of {list(Plot2DType)}"
                )
            plot_type = Plot2DType(plot_type)

            match plot_type:
                case Plot2DType.LINE:
                    ret_settings = TypeAdapter(Line2DSettings).validate_python(
                        plot_settings
                    )

            ret_data[plot_type] = ret_settings

        return ret_data

    def validate_targets(self, values: list[str]) -> bool:
        for target in self.targets:
            if target not in values:
                raise ValueError(f"Plot target '{target}' must be a key of {values}")
        return True


class Axes2DSettings(Axes2DDefaults):
    def plot(
        self,
        fig: plt.Figure,
        subplots: tuple[PositiveInt, PositiveInt],
        ax_index: PositiveInt,
        data: gpd.GeoDataFrame,
    ) -> plt.Axes:
        ax = fig.add_subplot(*subplots, ax_index, **self.kwargs)

        for plot_2d in self.plots.values():
            plot_2d.plot(ax, data[self.targets])

        ax.set_title(self.title)
        ax.set_xlabel(self.xlabel)
        ax.set_ylabel(self.ylabel)

        return ax


class TwinAxes2DDefaults(Axes2DDefaults):
    twin_targets: list[str]

    ytwinlabel: Optional[str] = Field(default=None, required=False)

    def validate_targets(self, values: list[str]) -> bool:
        for target in self.twin_targets:
            if target not in values:
                raise ValueError(
                    f"Twin plot target '{target}' must be a key of {values}"
                )
        return super(TwinAxes2DDefaults, self).validate_targets(values)


class TwinAxes2DSettings(TwinAxes2DDefaults):
    def plot(
        self,
        fig: plt.Figure,
        subplots: tuple[PositiveInt, PositiveInt],
        ax_index: PositiveInt,
        data: gpd.GeoDataFrame,
    ) -> plt.Axes:
        ax_left = fig.add_subplot(*subplots, ax_index, **self.kwargs)
        ax_right = ax_left.twinx()

        for plot_2d in self.plots.values():
            plot_2d.plot(ax_left, data[self.targets])
        for plot_2d in self.plots.values():
            plot_2d.plot(ax_right, data[self.twin_targets])

        ax_left.set_title(self.title)
        ax_left.set_xlabel(self.xlabel)
        ax_left.set_ylabel(self.ylabel)

        return ax_left


class Plot3DType(StrEnum):
    PATH = auto()
    STEM = auto()
    SCATTER = auto()
    INTERPOLATED = auto()
    CONTOUR = auto()


class Plot3DSettings(PlotSettings):
    pass


class Path3DSettings(Plot3DSettings):
    DEFAULT_DICT = dict(color="grey", linestyle=":")

    def plot(
        self,
        ax: Axes3D,
        geometry: gpd.GeoSeries,
        *args,
        **kwargs,
    ) -> None:
        ax.plot3D(geometry.x, geometry.y, geometry.z, **self.kwargs)


class Stem3DSettings(Plot3DSettings):
    DEFAULT_DICT = dict(markerfmt=".", linefmt="C7:")

    def plot(
        self,
        ax: Axes3D,
        geometry: gpd.GeoSeries,
        *args,
        **kwargs,
    ) -> None:
        ax.stem(
            geometry.x,
            geometry.y,
            geometry.z,
            **self.kwargs,
        )


class Scatter3DSettings(Plot3DSettings):
    DEFAULT_DICT = dict(cmap="viridis")

    def plot(
        self,
        ax: Axes3D,
        geometry: gpd.GeoSeries,
        values: gpd.GeoSeries,
    ) -> None:
        ax.scatter(
            geometry.x,
            geometry.y,
            geometry.z,
            c=values,
            **self.kwargs,
        )


class InterplolationSettings3D(BaseSettings):
    num_points: tuple[PositiveInt, PositiveInt, PositiveInt] = Field(
        default=(15, 15, 10),
        description="Number of interpolation points to plot in the lat, lon, and alt axes",
    )
    interpolation_method: str = Field(default="linear", required=False)

    def interpolate_data(
        self, geometry: gpd.GeoSeries, target_values: gpd.GeoSeries
    ) -> gpd.GeoDataFrame:
        num_points_x, num_points_y, num_points_z = self.num_points

        points = (geometry.x, geometry.y, geometry.z)
        coords_x, coords_y, coords_z = np.meshgrid(
            np.linspace(geometry.x.min(), geometry.x.max(), num_points_x),
            np.linspace(geometry.y.min(), geometry.y.max(), num_points_y),
            np.linspace(geometry.z.min(), geometry.z.max(), num_points_z),
        )
        values = griddata(
            points,
            target_values,
            (coords_x.flatten(), coords_y.flatten(), coords_z.flatten()),
            method=self.interpolation_method,
        )

        return gpd.GeoDataFrame(
            {"target": values},
            geometry=gpd.points_from_xy(
                coords_x.flatten(),
                coords_y.flatten(),
                coords_z.flatten(),
                crs=geometry.crs,
            ),
        )


class Interpolated3DSettings(Scatter3DSettings, InterplolationSettings3D):
    def plot(
        self,
        ax: Axes3D,
        geometry: gpd.GeoSeries,
        values: gpd.GeoSeries,
    ) -> None:
        data = self.interpolate_data(geometry, values)
        super(Interpolated3DSettings, self).plot(ax, data.geometry, data.target)


class Contour3DSettings(Plot3DSettings, InterplolationSettings3D):
    DEFAULT_DICT = dict(cmap="viridis")

    number_contours: int = Field(default=5, required=False)
    limit_contours: list[int] | range = Field(default=[], required=False)

    def plot(
        self,
        ax: Axes3D,
        geometry: gpd.GeoSeries,
        values: gpd.GeoSeries,
    ) -> None:
        data = self.interpolate_data(geometry, values)
        values = data.target
        if len(self.limit_contours) == 0:
            self.limit_contours = range(self.number_contours)

        kwargs = self.kwargs.copy()

        colors = sns.color_palette(kwargs.pop("cmap"), n_colors=self.number_contours)
        data["category"] = pd.cut(values, bins=self.limit_contours)

        for i, (color, cat) in enumerate(zip(colors, data.category.cat.categories)):
            if i not in self.limit_contours:
                continue
            sub_data = data.loc[data.category == cat, :]
            sub_geometry = sub_data.geometry
            X, Y, Z = (
                sub_geometry.x.values,
                sub_geometry.y.values,
                sub_geometry.z.values,
            )
            ax.plot_surface(X, Y, Z, color=color, **kwargs)


Plots3DUnion = Annotated[
    Path3DSettings
    | Stem3DSettings
    | Scatter3DSettings
    | Interpolated3DSettings
    | Contour3DSettings,
    "Union of possible 3D plots",
]


class Axes3DDefaults(AxesSettings):
    target: str = "bps"
    crs_target: Optional[str] = Field(default=None, required=False)

    zlabel: Optional[str] = Field(default=None, required=False)

    plots: dict[Plot3DType, Plots3DUnion] = {
        Plot3DType.INTERPOLATED: Interpolated3DSettings()
    }

    @field_validator("plots")
    @classmethod
    def _validate_plots(
        cls, data: dict[str, dict] | dict[Plot3DType, Plots3DUnion]
    ) -> dict[Plot3DType, Plots3DUnion]:
        ret_data: dict[Plot3DType, Plots3DUnion] = {}
        for plot_type, plot_settings in data.items():
            if plot_type not in Plot3DType:
                raise ValueError(
                    f"Plot type '{plot_type}' is not a valid plot type. "
                    f"Must be one of {list(Plot3DType)}"
                )
            plot_type = Plot3DType(plot_type)
            ret_settings: Plots3DUnion
            match plot_type:
                case Plot3DType.PATH:
                    ret_settings = TypeAdapter(Path3DSettings).validate_python(
                        plot_settings
                    )
                case Plot3DType.STEM:
                    ret_settings = TypeAdapter(Stem3DSettings).validate_python(
                        plot_settings
                    )
                case Plot3DType.SCATTER:
                    ret_settings = TypeAdapter(Scatter3DSettings).validate_python(
                        plot_settings
                    )
                case Plot3DType.INTERPOLATED:
                    ret_settings = TypeAdapter(Interpolated3DSettings).validate_python(
                        plot_settings
                    )
                case Plot3DType.CONTOUR:
                    ret_settings = TypeAdapter(Contour3DSettings).validate_python(
                        plot_settings
                    )
                case _:
                    raise ValueError(f"Plot type {plot_type} not supported")
            ret_data[plot_type] = ret_settings

        return ret_data

    def validate_targets(self, values: list[str]) -> bool:
        if self.target not in values:
            raise ValueError(f"Plot target '{self.target}' must be a key of {values}")
        return True


class Axes3DSettings(Axes3DDefaults):
    def plot(
        self,
        fig: plt.Figure,
        subplots: tuple[PositiveInt, PositiveInt],
        ax_index: PositiveInt,
        data: gpd.GeoDataFrame,
    ) -> Axes3D:
        if self.crs_target:
            data = data.to_crs(self.crs_target)

        ax = fig.add_subplot(*subplots, ax_index, projection="3d", **self.kwargs)
        for plot_3d in self.plots.values():
            plot_3d.plot(ax, data.geometry, data[self.target])

        ax.set_xlabel(self.xlabel)
        ax.set_ylabel(self.ylabel)
        ax.set_zlabel(self.zlabel)
        ax.set_title(self.title)

        return ax


AxesUnion = Annotated[
    Axes2DSettings | TwinAxes2DSettings | Axes3DSettings, "Union of possible axes"
]


class FigureDefaults(BaseSettings):
    subfigures: list[TwinAxes2DSettings | Axes2DSettings | Axes3DSettings] = [
        Axes2DSettings()
    ]
    num_cols: PositiveInt = 1
    num_rows: PositiveInt = 1

    title: str = Field(default="", required=False)
    xlabel: str = Field(default="", required=False)
    ylabel: str = Field(default="", required=False)

    tight_layout: bool = Field(default=True, required=False)

    kwargs: dict[str, Any] = Field(
        default={}, required=False, description="additional kwargs for figure"
    )

    @model_validator(mode="before")
    @classmethod
    def _validate_figure_settings(cls, data: dict[str, Any]) -> dict[str, Any]:
        match data:
            case {"subfigures": subfigures, "num_cols": num_cols, "num_rows": num_rows}:
                if num_cols * num_rows < len(subfigures):
                    raise ValueError(
                        "Number of rows and columns cannot fit the specified number of subfigures"
                    )
            case {"subfigures": subfigures, "num_cols": num_cols}:
                data["num_rows"] = int(np.ceil(len(subfigures) / num_cols))
            case {"subfigures": subfigures, "num_rows": num_rows}:
                data["num_cols"] = int(np.ceil(len(subfigures) / num_rows))
            case {"subfigures": subfigures}:
                data["num_cols"] = int(np.ceil(np.sqrt(len(subfigures))))
                data["num_rows"] = int(np.ceil(len(subfigures) / data["num_cols"]))
        return data

    def validate_targets(self, values: list[str]) -> bool:
        return all(subfigure.validate_targets(values) for subfigure in self.subfigures)


class FigureSettings(FigureDefaults):
    DEFAULTS: ClassVar[FigureDefaults] = FigureDefaults()

    @model_validator(mode="before")
    @classmethod
    def _fill_model_defaults(cls, data: dict[str, Any]) -> dict[str, Any]:
        if "subfigures" not in data:
            data["subfigures"] = cls.DEFAULTS.subfigures
        if "num_cols" not in data:
            data["num_cols"] = cls.DEFAULTS.num_cols
        if "num_rows" not in data:
            data["num_rows"] = cls.DEFAULTS.num_rows

        if "title" not in data:
            data["title"] = cls.DEFAULTS.title
        if "xlabel" not in data:
            data["xlabel"] = cls.DEFAULTS.xlabel
        if "ylabel" not in data:
            data["ylabel"] = cls.DEFAULTS.ylabel

        if "tight_layout" not in data:
            data["tight_layout"] = cls.DEFAULTS.tight_layout

        if "kwargs" not in data:
            data["kwargs"] = cls.DEFAULTS.kwargs
        return data

    def plot(self, data: gpd.GeoDataFrame) -> plt.Figure:
        fig = plt.figure(**self.kwargs)
        subplots = (self.num_rows, self.num_cols)

        for i, subfigure in enumerate(self.subfigures):
            subfigure.plot(fig, subplots, i + 1, data)

        if self.title:
            fig.suptitle(self.title)
        if self.xlabel:
            fig.supxlabel(self.xlabel)
        if self.ylabel:
            fig.supylabel(self.ylabel)
        if self.tight_layout:
            fig.tight_layout()

        return fig


class CorrelationCalcDefaults(BaseSettings):
    targets: list[str] = Field(default=["bps", "rssi"], required=True, min_length=2)

    def validate_targets(self, values: list[str]) -> bool:
        for target in self.targets:
            if target not in values:
                raise ValueError(
                    f"Calculation target '{target}' must be a key of {values}"
                )
        return True


class CorrelationCalcSettings(CorrelationCalcDefaults):
    """
    Settings for a correlation calculation.
    """

    def calculate(self, data: gpd.GeoDataFrame) -> pd.DataFrame:
        return data[self.targets].corr()


class FrameType(StrEnum):
    """
    The type of data frame/source.
    """

    UAV = auto()
    IPERF = auto()
    WIRELESS = auto()


class ExperimentDefaults(BaseSettings):
    retrieval_offsets: dict[FrameType, timedelta] = Field(default={}, required=False)
    offset_range: range | dict[str, int] = Field(default=range(0, 0), required=False)

    influxDB: InfluxDBSettings = Field(
        default=InfluxDBSettings(
            host="127.0.0.1", org="my-org", bucket="my-bucket", token="my-token"  # type: ignore[arg-type]
        ),
        required=False,
    )

    gps_fields: dict[str, str] = Field(
        default={
            "lat": "rtk_lat",
            "lon": "rtk_lon",
            "alt": "rtk_height_from_sea",
        },
        required=False,
    )
    extra_uav_fields: dict[str, str] = Field(default={}, required=False)
    iperf_fields: dict[str, str] = Field(
        default={
            "bps": "bits_per_second",
        },
        required=False,
    )
    wireless_fields: dict[str, str] = Field(
        default={
            "rssi": "signal_strength",
        },
        required=False,
    )
    stream_types: list[str] = Field(default=["interval sum"], required=False)

    figures: list[FigureSettings] = Field(
        default=[
            FigureSettings(),
            FigureSettings(subfigures=[Axes3DSettings()]),
        ]
    )

    correlations: list[CorrelationCalcSettings] = Field(
        default=[],
        required=False,
    )

    crs_target: Optional[str] = Field(default=None, required=False)

    @field_validator("offset_range")
    @classmethod
    def _validate_offset_range(cls, data: dict[str, int] | range) -> range:
        if isinstance(data, range):
            return data

        if "start" not in data:
            data["start"] = 0
        if "step" not in data:
            data["step"] = 1
        return range(data["start"], data["end"], data["step"])

    @property
    def eval_field_names(self) -> list[str]:
        keys = list(self.iperf_fields.keys())
        keys.extend(self.wireless_fields.keys())
        if self.extra_uav_fields:
            keys.extend(self.extra_uav_fields.keys())
        return keys

    @property
    def uav_fields(self) -> dict[str, str]:
        if self.extra_uav_fields:
            return merge_dicts(self.gps_fields, self.extra_uav_fields)
        return self.gps_fields

    @field_validator("gps_fields")
    @classmethod
    def _validate_gps_fields(cls, data: dict[str, str]) -> dict[str, str]:
        if not all(k in data.keys() for k in ["lat", "lon", "alt"]):
            raise ValueError("gps_fields must contain 'lat', 'lon' and 'alt' fields")
        return data

    @model_validator(mode="after")  # type: ignore[arg-type]
    @classmethod
    def _validate_settings(cls, data: "ExperimentDefaults") -> "ExperimentDefaults":
        MODULE_LOGGER.debug("Validating experiment settings")
        for figure in data.figures:
            figure.validate_targets(data.eval_field_names)
        for corr in data.correlations:
            corr.validate_targets(data.eval_field_names)

        return data


class ExperimentSettings(ExperimentDefaults):
    """
    Settings for an experiment evaluation.
    """

    DEFAULTS: ClassVar[ExperimentDefaults] = ExperimentDefaults()

    start: datetime
    end: datetime

    @model_validator(mode="before")
    @classmethod
    def _fill_model_defaults(cls, data: dict) -> dict:
        MODULE_LOGGER.debug("Filling model defaults")

        if "influxDB" not in data:
            data["influxDB"] = cls.DEFAULTS.influxDB
        if "gps_fields" not in data:
            data["gps_fields"] = cls.DEFAULTS.gps_fields
        if "extra_uav_fields" not in data:
            data["extra_uav_fields"] = cls.DEFAULTS.extra_uav_fields
        if "iperf_fields" not in data:
            data["iperf_fields"] = cls.DEFAULTS.iperf_fields
        if "wireless_fields" not in data:
            data["wireless_fields"] = cls.DEFAULTS.wireless_fields
        if "stream_types" not in data:
            data["stream_types"] = cls.DEFAULTS.stream_types
        if "figures" not in data:
            data["figures"] = cls.DEFAULTS.figures
        if "correlations" not in data:
            data["correlations"] = cls.DEFAULTS.correlations

        return data

    def _get_offset_time(self, frame_type: FrameType) -> tuple[datetime, datetime]:
        start = self.start
        end = self.end

        if frame_type in self.retrieval_offsets:
            start += self.retrieval_offsets[frame_type]
            end += self.retrieval_offsets[frame_type]

        return start, end

    @cached_property
    def query_api(self) -> InfluxDBClient.query_api:
        return InfluxDBClient(
            url=self.influxDB.url,
            token=self.influxDB.token.get_secret_value(),
            org=self.influxDB.org,
        ).query_api()

    @property
    def uav_query(self) -> str:
        start, end = self._get_offset_time(FrameType.UAV)

        return build_influx_query(
            self.influxDB.bucket,
            start,
            end,
            f"""
            {build_influx_filters("_measurement", ["drone_metrics"])}
            {build_influx_filters("_field", list(self.uav_fields.values()))}""",
        )

    @property
    def iperf_query(self) -> str:
        start, end = self._get_offset_time(FrameType.IPERF)

        return build_influx_query(
            self.influxDB.bucket,
            start,
            end,
            f"""
            {build_influx_filters("_measurement", ["iperf3"])}
            {build_influx_filters("_field", list(self.iperf_fields.values()))}
            {build_influx_filters("type", self.stream_types)}""",
        )

    @property
    def wireless_query(self) -> str:
        start, end = self._get_offset_time(FrameType.WIRELESS)

        return build_influx_query(
            self.influxDB.bucket,
            start,
            end,
            f"""
            {build_influx_filters("_measurement", ["wireless"])}
            {build_influx_filters("_field", list(self.wireless_fields.values()))}""",
        )

    @staticmethod
    def _process_dataframe(data: pd.DataFrame) -> pd.DataFrame:
        MODULE_LOGGER.debug("Processing dataframe\n%s", data.describe())
        if data.empty:
            raise ValueError(
                "DataFrame is empty. Perhaps incorrect filters were specified?"
            )
        data["_time"] = data["_time"].apply(floor_time)
        data = data.drop_duplicates(subset="_time")
        data = data.set_index("_time").drop(
            ["result", "table", "_start", "_stop", "_measurement"], axis=1
        )
        return data

    def _get_uav_data(self) -> pd.DataFrame:
        data = self.query_api.query_data_frame(self.uav_query, org=self.influxDB.org)
        if FrameType.UAV in self.retrieval_offsets:
            data["_time"] = data["_time"].apply(
                lambda t: t - self.retrieval_offsets[FrameType.UAV]
            )
        data = ExperimentSettings._process_dataframe(data)
        data = data.rename(columns=invert_dict(self.uav_fields))
        return data

    def _get_iperf_data(self) -> pd.DataFrame:
        data = self.query_api.query_data_frame(self.iperf_query, org=self.influxDB.org)
        if FrameType.IPERF in self.retrieval_offsets:
            data["_time"] = data["_time"].apply(
                lambda t: t - self.retrieval_offsets[FrameType.IPERF]
            )
        data = ExperimentSettings._process_dataframe(data)
        data = data.rename(columns=invert_dict(self.iperf_fields))
        return data

    def _get_wireless_data(self) -> pd.DataFrame:
        data = self.query_api.query_data_frame(
            self.wireless_query, org=self.influxDB.org
        )
        if FrameType.WIRELESS in self.retrieval_offsets:
            data["_time"] = data["_time"].apply(
                lambda t: t - self.retrieval_offsets[FrameType.WIRELESS]
            )
        data = ExperimentSettings._process_dataframe(data)
        data = data.rename(columns=invert_dict(self.wireless_fields))
        return data

    def _collect_frames(self) -> pd.DataFrame:
        uav_data = self._get_uav_data()
        iperf_data = self._get_iperf_data()
        wireless_data = self._get_wireless_data()

        return uav_data.join(iperf_data).join(wireless_data)

    @cached_property
    def geo_data(self) -> gpd.GeoDataFrame:
        data = self._collect_frames()

        MODULE_LOGGER.debug("Collected data:\n%s", data.describe())

        # fix zero-altitudes
        data["alt"] = data["alt"].replace(0, np.nan).interpolate(method="linear")
        # normalize to zero
        if "sea" in self.gps_fields["alt"]:
            data["alt"] = data["alt"] - data["alt"].min()

        rows_before = len(data.index)
        data = data.dropna()
        rows_after = len(data.index)
        MODULE_LOGGER.debug(
            "Dropped %d rows due to missing values. %d rows remaining.",
            rows_before - rows_after,
            rows_after,
        )

        # create GeoDataFrame
        geo_data = gpd.GeoDataFrame(
            data,
            geometry=gpd.points_from_xy(
                data["lon"], data["lat"], data["alt"], crs="WGS84"
            ),
        )
        # convert if requested
        if self.DEFAULTS.crs_target is not None:
            geo_data = geo_data.to_crs(ExperimentSettings.DEFAULTS.crs_target)

        return geo_data

    def plot(self):
        for figure in self.figures:
            figure.plot(self.geo_data)

    def calculate(self):
        for corr in self.correlations:
            corr.calculate(self.geo_data)


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

    influxDB: InfluxDBSettings
    experiments_file: Path = Field(..., required=True)

    @computed_field  # type: ignore[misc]
    @cached_property
    def experiments(self) -> list[ExperimentSettings]:
        with open(self.experiments_file, "r", encoding="utf-8") as file:
            experiments_json = json.load(file)

        if "defaults" in experiments_json:
            ExperimentSettings.DEFAULTS = TypeAdapter(
                ExperimentDefaults
            ).validate_python(experiments_json["defaults"])
        else:
            ExperimentSettings.DEFAULTS = ExperimentDefaults()

        if not "experiments" in experiments_json:
            raise ValueError("No experiments found in experiments file")
        experiments_dict = experiments_json["experiments"]

        MODULE_LOGGER.debug(
            "Loaded %d experiments from %s",
            len(experiments_dict),
            self.experiments_file,
        )
        experiments: list[ExperimentSettings] = TypeAdapter(
            list[ExperimentSettings]
        ).validate_python(experiments_dict)
        MODULE_LOGGER.debug(experiments)
        return experiments


####################################################################################################


# def plot_with_offsets(
#     settings: EvaluationSettings, start: int = -64600, end: int = -64500, step=10
# ):
#     offset_range = range(start, end + 1, step)
#     num_plots = len(offset_range)

#     aspect_shape = (4, 3)
#     aspect_factor = num_plots / np.prod(aspect_shape)
#     ncols = int(np.ceil(aspect_shape[0] * aspect_factor))
#     nrows = int(np.ceil(aspect_shape[1] * aspect_factor))

#     # prepare data plot
#     fig, axes = plt.subplots(
#         ncols=ncols, nrows=nrows, subplot_kw={"projection": "3d"}, figsize=(10, 10)
#     )
#     flat_axes = []
#     if isinstance(axes, Axes3D):
#         flat_axes.append(axes)
#     else:
#         flat_axes = axes.flatten()

#     # get GeoDataFrame
#     for ax, offset in zip(flat_axes, offset_range):
#         t_offset = timedelta(seconds=offset)
#         MODULE_LOGGER.info("Iteration start; Offset: %s seconds (%s)", offset, t_offset)
#         gdf = get_geodata(settings, t_offset)

#         # select only datapoints where the UAV is waiting
#         gdf = gdf[gdf["mission_status"] == 2]

#         if offset == offset_range[0]:
#             MODULE_LOGGER.debug("Initial GeoDataFrame:")
#             MODULE_LOGGER.debug(gdf.head())
#             MODULE_LOGGER.info(
#                 "Correlation of BPS and Signal Strength: %s",
#                 gdf[["bps", "signal_strength"]].corr().loc["signal_strength", "bps"],
#             )

#         # Plot UAV path as stem
#         # plot_path_stem(ax, gdf.geometry)

#         # Plot UAV path
#         plot_path(ax, gdf.geometry)

#         # Plot target data
#         plot_4d_data(ax, gdf.geometry, gdf[settings.plot_target])

#         # Interpolate data
#         # plot_interpolated_data(
#         #     ax,
#         #     gdf.geometry,
#         #     gdf[settings.plot_target],
#         #     nx=20,
#         #     ny=20,
#         #     nz=12,
#         #     method="linear",
#         #     alpha=0.6,
#         # )

#         # Plot contours
#         # plot_4d_contours(
#         #     ax,
#         #     gdf.geometry,
#         #     gdf[settings.plot_target],
#         #     n_contours=2,
#         #     nx=50,
#         #     ny=50,
#         #     nz=20,
#         #     limit_contours=None,
#         #     method="linear",
#         # )

#         ax.set_title(f"Offset: {t_offset}")

#     fig.tight_layout()
#     return fig


# def plot_bps_ss_correlation(settings: EvaluationSettings):
#     iperf_wireless_cols = ["bps", "signal_strength"]
#     gdf = get_geodata(settings, timedelta(seconds=-64600)).loc[:, iperf_wireless_cols]
#     MODULE_LOGGER.info("\n%s", gdf.head())

#     fig, ax_left = plt.subplots()
#     ax_right = ax_left.twinx()

#     ax_left.plot(gdf["bps"], color="C0")
#     ax_right.plot(gdf["signal_strength"], color="C21")
#     ax_left.set_ylabel("Throughput (bps)")
#     ax_right.set_ylabel("Signal Strength (dBm)")
#     correlation = gdf.corr().loc["signal_strength", "bps"]
#     ax_left.set_title(f"Correlation of BPS and Signal Strength: {correlation:.3f}")
#     fig.tight_layout()


####################################################################################################


def main():
    sns.set_style("darkgrid")
    # Read settings
    eval_settings = EvaluationSettings()
    for experiment in eval_settings.experiments:
        experiment.plot()
        print(experiment.calculate())

    plt.show()


if __name__ == "__main__":
    main()
