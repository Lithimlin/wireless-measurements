#!/usr/bin/env python3

import json
import logging
import sys
from abc import ABC, abstractmethod
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


def addLoggingLevel(levelName, levelNum, methodName=None):
    """
    Comprehensively adds a new logging level to the `logging` module and the
    currently configured logging class.

    `levelName` becomes an attribute of the `logging` module with the value
    `levelNum`. `methodName` becomes a convenience method for both `logging`
    itself and the class returned by `logging.getLoggerClass()` (usually just
    `logging.Logger`). If `methodName` is not specified, `levelName.lower()` is
    used.

    To avoid accidental clobberings of existing attributes, this method will
    raise an `AttributeError` if the level name is already an attribute of the
    `logging` module or if the method name is already present

    Example
    -------
    >>> addLoggingLevel('TRACE', logging.DEBUG - 5)
    >>> logging.getLogger(__name__).setLevel("TRACE")
    >>> logging.getLogger(__name__).trace('that worked')
    >>> logging.trace('so did this')
    >>> logging.TRACE
    5

    """
    if not methodName:
        methodName = levelName.lower()

    if hasattr(logging, levelName):
        raise AttributeError("{} already defined in logging module".format(levelName))
    if hasattr(logging, methodName):
        raise AttributeError("{} already defined in logging module".format(methodName))
    if hasattr(logging.getLoggerClass(), methodName):
        raise AttributeError("{} already defined in logger class".format(methodName))

    # This method was inspired by the answers to Stack Overflow post
    # http://stackoverflow.com/q/2183233/2988730, especially
    # http://stackoverflow.com/a/13638084/2988730
    def logForLevel(self, message, *args, **kwargs):
        if self.isEnabledFor(levelNum):
            self._log(levelNum, message, args, **kwargs)

    def logToRoot(message, *args, **kwargs):
        logging.log(levelNum, message, *args, **kwargs)

    logging.addLevelName(levelNum, levelName)
    setattr(logging, levelName, levelNum)
    setattr(logging.getLoggerClass(), methodName, logForLevel)
    setattr(logging, methodName, logToRoot)


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
        "%(asctime)s %(levelname)s: \t%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    channel.setFormatter(formatter)
    logger.addHandler(channel)
    return logger


addLoggingLevel("VERBOSE", logging.DEBUG - 5)
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


class PlotSettings(BaseSettings, ABC):
    DEFAULT_DICT: ClassVar[dict[str, Any]] = dict()
    kwargs: dict[str, Any] = Field(default={}, required=False)

    @field_validator("kwargs")
    @classmethod
    def _validate_kwargs(cls, kwargs: dict[str, Any]) -> dict[str, Any]:
        return merge_dicts(cls.DEFAULT_DICT, kwargs)

    @model_validator(mode="before")
    @classmethod
    def _debug_print(cls, data: dict[str, Any]) -> dict[str, Any]:
        MODULE_LOGGER.verbose(f"Creating {cls.__name__}: {data}")
        return data

    def plot(self, *args, **kwargs) -> None:
        raise NotImplementedError(
            f"Tried calling {__name__} method on {type(self).__name__} object"
        )


class AxesSettings(BaseSettings, ABC):
    title: Optional[str] = Field(default=None, required=False)
    xlabel: Optional[str] = Field(default=None, required=False)
    ylabel: Optional[str] = Field(default=None, required=False)

    kwargs: dict[str, Any] = Field(
        default={}, required=False, description="additional kwargs for subplots"
    )

    @model_validator(mode="before")
    @classmethod
    def _debug_print(cls, data: dict[str, Any]) -> dict[str, Any]:
        MODULE_LOGGER.verbose(f"Creating {cls.__name__}: {data}")
        return data

    @abstractmethod
    def validate_targets(self, *args, **kwargs) -> bool:
        pass

    @abstractmethod
    def plot(self, *args, **kwargs) -> None:
        pass


class Plot2DType(StrEnum):
    LINE = auto()


class Plot2DSettings(PlotSettings):
    pass


class Line2DSettings(Plot2DSettings):
    def plot(self, ax: plt.Axes, data: gpd.GeoSeries) -> None:
        ax.plot(data, **self.kwargs)


Plots2DUnion = Annotated[Line2DSettings, "Union of available 2D plots"]


class Axes2DDefaults(AxesSettings):
    targets: list[str] = ["bps"]

    plots: dict[Plot2DType, Plots2DUnion] = {Plot2DType.LINE: Line2DSettings()}

    @field_validator("plots", mode="before")
    @classmethod
    def _validate_plots(
        cls, data: dict[str, dict] | dict[Plot2DType, Plots2DUnion]
    ) -> dict[Plot2DType, Plots2DUnion]:
        ret_data: dict[Plot2DType, Plots2DUnion] = {}

        for plot_type, plot_settings in data.items():
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

    def plot(self, *args, **kwargs) -> None:
        raise NotImplementedError(
            f"Tried calling {__name__} method on {type(self).__name__} object"
        )


class Axes2DSettings(Axes2DDefaults):
    DEFAULTS: ClassVar[Axes2DDefaults] = Axes2DDefaults()

    @model_validator(mode="before")
    @classmethod
    def _fill_model_defaults(cls, data: dict[str, Any]) -> dict[str, Any]:
        MODULE_LOGGER.debug(f"Filling defaults in {cls.__name__}")
        for field in type(cls.DEFAULTS).model_fields:
            if field not in data:
                data[field] = getattr(cls.DEFAULTS, field)
        return data

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
    DEFAULTS: ClassVar[Axes2DDefaults] = Axes2DDefaults()

    twin_targets: list[str] = ["rssi"]

    ytwinlabel: Optional[str] = Field(default=None, required=False)

    @model_validator(mode="before")
    @classmethod
    def _fill_model_defaults(cls, data: dict[str, Any]) -> dict[str, Any]:
        MODULE_LOGGER.debug(f"Filling defaults in {cls.__name__}")
        for field in type(cls.DEFAULTS).model_fields:
            if field not in data:
                data[field] = getattr(cls.DEFAULTS, field)
        return data

    def validate_targets(self, values: list[str]) -> bool:
        for target in self.twin_targets:
            if target not in values:
                raise ValueError(
                    f"Twin plot target '{target}' must be a key of {values}"
                )
        return super(TwinAxes2DDefaults, self).validate_targets(values)

    def plot(self, *args, **kwargs) -> None:
        raise NotImplementedError(
            f"Tried calling {__name__} method on {type(self).__name__} object"
        )


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
        ax_right._get_lines.prop_cycler = ax_left._get_lines.prop_cycler
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


class InterpolationDefaults3D(BaseSettings):
    num_points: tuple[PositiveInt, PositiveInt, PositiveInt] = Field(
        default=(15, 15, 10),
        description="Number of interpolation points to plot in the lat, lon, and alt axes",
    )
    interpolation_method: str = Field(default="linear", required=False)


class InterplolationSettings3D(InterpolationDefaults3D):
    DEFAULTS: ClassVar[InterpolationDefaults3D] = InterpolationDefaults3D()

    @model_validator(mode="before")
    @classmethod
    def _fill_model_defaults(cls, data: dict[str, Any]) -> dict[str, Any]:
        MODULE_LOGGER.debug(f"Filling defaults in {cls.__name__}")
        for field in type(cls.DEFAULTS).model_fields:
            if field not in data:
                data[field] = getattr(cls.DEFAULTS, field)
        return data

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


class Contour3DDefaults(Plot3DSettings, InterplolationSettings3D):
    number_contours: int = Field(default=5, required=False)
    limit_contours: list[int] | range = Field(default=[], required=False)

    def plot(self, *args, **kwargs) -> None:
        raise NotImplementedError(
            f"Tried calling {__name__} method on {type(self).__name__} object"
        )


class Contour3DSettings(Contour3DDefaults):
    DEFAULT_DICT = dict(cmap="viridis")
    DEFAULTS: ClassVar[Contour3DDefaults] = Contour3DDefaults()

    @model_validator(mode="before")
    @classmethod
    def _fill_model_defaults(cls, data: dict[str, Any]) -> dict[str, Any]:
        MODULE_LOGGER.debug(f"Filling defaults in {cls.__name__}")
        for field in type(cls.DEFAULTS).model_fields:
            if field not in data:
                data[field] = getattr(cls.DEFAULTS, field)
        return data

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

    @field_validator("plots", mode="before")
    @classmethod
    def _parse_plots(
        cls, data: dict[str, dict] | dict[Plot3DType, Plots3DUnion]
    ) -> dict[Plot3DType, Plots3DUnion]:
        ret_data: dict[Plot3DType, Plots3DUnion] = {}
        for plot_type, plot_settings in data.items():
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

    def plot(self, *args, **kwargs) -> None:
        raise NotImplementedError(
            f"Tried calling {__name__} method on {type(self).__name__} object"
        )


class Axes3DSettings(Axes3DDefaults):
    DEFAULTS: ClassVar[Axes3DDefaults] = Axes3DDefaults()

    @model_validator(mode="before")
    @classmethod
    def _fill_model_defaults(cls, data: dict[str, Any]) -> dict[str, Any]:
        MODULE_LOGGER.debug(f"Filling defaults in {cls.__name__}")
        for field in type(cls.DEFAULTS).model_fields:
            if field not in data:
                data[field] = getattr(cls.DEFAULTS, field)
        return data

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


class AxesType(StrEnum):
    Axes2D = "Axes2D"
    TwinAxes2D = "TwinAxes2D"
    Axes3D = "Axes3D"


class FigureDefaults(BaseSettings):
    subfigures: list[AxesUnion] = [Axes2DSettings()]
    num_cols: PositiveInt = 1
    num_rows: PositiveInt = 1

    title: str = Field(default="", required=False)
    xlabel: str = Field(default="", required=False)
    ylabel: str = Field(default="", required=False)

    tight_layout: bool = Field(default=True, required=False)

    kwargs: dict[str, Any] = Field(
        default={}, required=False, description="additional kwargs for figure"
    )

    @field_validator("subfigures", mode="before")
    @classmethod
    def _parse_axes(cls, data: list[dict | AxesUnion]) -> list[AxesUnion]:
        MODULE_LOGGER.debug(f"Parsing subfigures: {data}")
        ret_data: list[AxesUnion] = []
        for axes_settings in data:
            if not isinstance(axes_settings, dict):
                MODULE_LOGGER.verbose(
                    f"Directly adding {type(axes_settings).__name__}({axes_settings.model_dump()}) to subfigures"
                )
                ret_data.append(axes_settings)
                continue

            axes_type = AxesType(axes_settings.pop("type"))
            ret_settings: AxesUnion
            match axes_type:
                case AxesType.Axes2D:
                    ret_settings = TypeAdapter(Axes2DSettings).validate_python(
                        axes_settings
                    )
                case AxesType.TwinAxes2D:
                    ret_settings = TypeAdapter(TwinAxes2DSettings).validate_python(
                        axes_settings
                    )
                case AxesType.Axes3D:
                    ret_settings = TypeAdapter(Axes3DSettings).validate_python(
                        axes_settings
                    )
                case _:
                    raise ValueError(f"Axes type {axes_type} not supported")
            ret_data.append(ret_settings)
        return ret_data

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
                data["num_cols"] = max(int(np.ceil(np.sqrt(len(subfigures)))), 1)
                data["num_rows"] = max(
                    int(np.ceil(len(subfigures) / data["num_cols"])), 1
                )
        return data

    @model_validator(mode="before")
    @classmethod
    def _debug_print(cls, data: dict[str, Any]) -> dict[str, Any]:
        MODULE_LOGGER.verbose(f"Creating {cls.__name__}: {data}")
        return data

    def validate_targets(self, values: list[str]) -> bool:
        return all(subfigure.validate_targets(values) for subfigure in self.subfigures)


class FigureSettings(FigureDefaults):
    DEFAULTS: ClassVar[FigureDefaults] = FigureDefaults()

    @model_validator(mode="before")
    @classmethod
    def _fill_model_defaults(cls, data: dict[str, Any]) -> dict[str, Any]:
        MODULE_LOGGER.debug(f"Filling defaults in {cls.__name__}")
        for field in type(cls.DEFAULTS).model_fields:
            if field not in data:
                data[field] = getattr(cls.DEFAULTS, field)
        return data

    def plot(self, data: gpd.GeoDataFrame) -> plt.Figure:
        fig = plt.figure(**self.kwargs)
        subplots = (self.num_rows, self.num_cols)

        axes: list[plt.Axes | Axes3D] = []
        axes3d: list[Axes3D] = []
        for i, subfigure in enumerate(self.subfigures):
            ax = subfigure.plot(fig, subplots, i + 1, data)
            axes.append(ax)
            if isinstance(subfigure, Axes3DSettings):
                axes3d.append(ax)

        # if len(axes3d) > 1:
        #     for ax3d in axes3d[1:]:
        #         axes3d[0].shareview(ax3d)

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

    DEFAULTS: ClassVar[CorrelationCalcDefaults] = CorrelationCalcDefaults()

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
        MODULE_LOGGER.debug("Validating experiment targets...")
        for figure in data.figures:
            figure.validate_targets(data.eval_field_names)
        for corr in data.correlations:
            corr.validate_targets(data.eval_field_names)

        return data

    @model_validator(mode="before")
    @classmethod
    def _debug_print(cls, data: dict[str, Any]) -> dict[str, Any]:
        MODULE_LOGGER.verbose(f"Creating {cls.__name__}: {data}")
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
    def _fill_model_defaults(cls, data: dict[str, Any]) -> dict[str, Any]:
        MODULE_LOGGER.debug(f"Filling defaults in {cls.__name__}")
        for field in type(cls.DEFAULTS).model_fields:
            if field not in data:
                data[field] = getattr(cls.DEFAULTS, field)
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
        MODULE_LOGGER.verbose("Processing dataframe\n%s", data.describe())
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
        figs = []
        for figure in self.figures:
            figs.append(figure.plot(self.geo_data))
        return figs

    def calculate(self):
        corrs = []
        for corr in self.correlations:
            corrs.append(corr.calculate(self.geo_data))
        return corrs


class EvaluationEnvelope(BaseSettings):
    line2d_defaults: Line2DSettings = Field(default=Line2DSettings())
    path3d_defaults: Path3DSettings = Field(default=Path3DSettings())
    stem3d_defaults: Stem3DSettings = Field(default=Stem3DSettings())
    scatter3d_defaults: Scatter3DSettings = Field(default=Scatter3DSettings())
    interpolation3d_defaults: InterpolationDefaults3D = Field(
        default=InterpolationDefaults3D()
    )
    interpolated3d_defaults: Interpolated3DSettings = Field(
        default=Interpolated3DSettings()
    )
    contour3d_defaults: Contour3DDefaults = Field(default=Contour3DDefaults())

    correlation_defaults: CorrelationCalcDefaults = Field(
        default=CorrelationCalcDefaults()
    )

    axes3d_defaults: Axes3DDefaults = Field(default=Axes3DDefaults())
    axes2d_defaults: Axes2DDefaults = Field(default=Axes2DDefaults())
    twin_axes2d_defaults: TwinAxes2DDefaults = Field(default=TwinAxes2DDefaults())

    figure_defaults: FigureDefaults = Field(default=FigureDefaults())

    experiment_defaults: ExperimentDefaults = Field(
        default=ExperimentDefaults(), alias="defaults"
    )

    experiments: list[ExperimentSettings]

    @field_validator("experiment_defaults")
    @classmethod
    def _set_experiment_defaults(cls, value: ExperimentDefaults) -> ExperimentDefaults:
        MODULE_LOGGER.debug("Setting experiment defaults")
        ExperimentSettings.DEFAULTS = value
        return value

    @field_validator("correlation_defaults")
    @classmethod
    def _set_correlation_defaults(
        cls, value: CorrelationCalcDefaults
    ) -> CorrelationCalcDefaults:
        MODULE_LOGGER.debug("Setting correlation calculation defaults")
        CorrelationCalcSettings.DEFAULTS = value
        return value

    @field_validator("figure_defaults")
    @classmethod
    def _set_figure_defaults(cls, value: FigureDefaults) -> FigureDefaults:
        MODULE_LOGGER.debug("Setting figure defaults")
        FigureSettings.DEFAULTS = value
        return value

    @field_validator("axes3d_defaults")
    @classmethod
    def _set_axes3d_defaults(cls, value: Axes3DDefaults) -> Axes3DDefaults:
        MODULE_LOGGER.debug("Setting axes3D defaults")
        Axes3DSettings.DEFAULTS = value
        return value

    @field_validator("axes2d_defaults")
    @classmethod
    def _set_axes2d_defaults(cls, value: Axes2DDefaults) -> Axes2DDefaults:
        MODULE_LOGGER.debug("Setting axes2D defaults")
        Axes2DSettings.DEFAULTS = value
        return value

    @field_validator("twin_axes2d_defaults")
    @classmethod
    def _set_twin_axes2d_defaults(cls, value: TwinAxes2DDefaults) -> TwinAxes2DDefaults:
        MODULE_LOGGER.debug("Setting twin axes2D defaults")
        TwinAxes2DSettings.DEFAULTS = value
        return value

    @field_validator("line2d_defaults")
    @classmethod
    def _set_line2d_defaults(cls, value: Line2DSettings) -> Line2DSettings:
        MODULE_LOGGER.debug("Setting line2D defaults")
        Line2DSettings.DEFAULT_DICT = value.kwargs
        return value

    @field_validator("path3d_defaults")
    @classmethod
    def _set_path3d_defaults(cls, value: Path3DSettings) -> Path3DSettings:
        MODULE_LOGGER.debug("Setting path3D defaults")
        Path3DSettings.DEFAULT_DICT = value.kwargs
        return value

    @field_validator("stem3d_defaults")
    @classmethod
    def _set_stem3d_defaults(cls, value: Stem3DSettings) -> Stem3DSettings:
        MODULE_LOGGER.debug("Setting stem3D defaults")
        Stem3DSettings.DEFAULT_DICT = value.kwargs
        return value

    @field_validator("scatter3d_defaults")
    @classmethod
    def _set_scatter3d_defaults(cls, value: Scatter3DSettings) -> Scatter3DSettings:
        MODULE_LOGGER.debug("Setting scatter3D defaults")
        Scatter3DSettings.DEFAULT_DICT = value.kwargs
        return value

    @field_validator("interpolated3d_defaults")
    @classmethod
    def _set_interpolated3d_defaults(
        cls, value: Interpolated3DSettings
    ) -> Interpolated3DSettings:
        MODULE_LOGGER.debug("Setting interpolated3D defaults")
        Interpolated3DSettings.DEFAULTS = InterpolationDefaults3D(
            num_points=value.num_points, interpolation_method=value.interpolation_method
        )
        Interpolated3DSettings.DEFAULT_DICT = value.kwargs
        return value

    @field_validator("interpolation3d_defaults")
    @classmethod
    def _set_interpolation3d_defaults(
        cls, value: InterpolationDefaults3D
    ) -> InterpolationDefaults3D:
        MODULE_LOGGER.debug("Setting interpolation3D defaults")
        InterplolationSettings3D.DEFAULTS = value
        return value

    @field_validator("contour3d_defaults")
    @classmethod
    def _set_contour3d_defaults(cls, value: Contour3DDefaults) -> Contour3DDefaults:
        MODULE_LOGGER.debug("Setting contour3D defaults")
        Contour3DSettings.DEFAULT_DICT = value.kwargs
        Contour3DSettings.DEFAULTS = value
        return value

    @model_validator(mode="before")
    @classmethod
    def _debug_print(cls, data: dict[str, Any]) -> dict[str, Any]:
        MODULE_LOGGER.verbose(f"Creating {cls.__name__}: {data}")
        return data


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

        envelope = TypeAdapter(EvaluationEnvelope).validate_python(experiments_json)
        experiments = envelope.experiments
        MODULE_LOGGER.debug(experiments)
        return experiments


####################################################################################################


def main():
    print("=================")
    sns.set_style("darkgrid")
    sns.set_palette("colorblind")
    # Read settings
    eval_settings = EvaluationSettings()
    for experiment in eval_settings.experiments:
        experiment.plot()
        for result in experiment.calculate():
            print(result)

    plt.show()


if __name__ == "__main__":
    main()
