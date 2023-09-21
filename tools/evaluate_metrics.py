#!/usr/bin/env python3

import copy
import json
import sys
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from enum import StrEnum, auto
from functools import cached_property
from pathlib import Path
from re import findall
from typing import Annotated, Any, Callable, ClassVar, Iterable, Literal, Optional
from zoneinfo import ZoneInfo

import geopandas as gpd  # type: ignore[import]
import matplotlib.pyplot as plt  # type: ignore[import]
import module_logging
import numpy as np
import pandas as pd  # type: ignore[import]
import seaborn as sns  # type: ignore[import]
from influxdb_client import InfluxDBClient  # type: ignore[import]
from matplotlib.cm import ScalarMappable  # type: ignore[import]
from matplotlib.collections import PathCollection  # type: ignore[import]
from matplotlib.colors import LogNorm, Normalize  # type: ignore[import]
from matplotlib.container import StemContainer  # type: ignore[import]
from mpl_toolkits.mplot3d.art3d import Line3D, Poly3DCollection  # type: ignore[import]
from mpl_toolkits.mplot3d.axes3d import Axes3D  # type: ignore[import]
from pydantic import (
    Field,
    PositiveInt,
    TypeAdapter,
    computed_field,
    field_serializer,
    field_validator,
    model_validator,
)
from pydantic_json_source import JsonConfigSettingsSource
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)
from scipy.interpolate import griddata  # type: ignore[import]

from wifi_info.settings import InfluxDBSettings

module_logging.addLoggingLevel("VERBOSE", module_logging.logging.DEBUG - 5)
MODULE_LOGGER = module_logging.get_logger(module_logging.logging.INFO)


def floor_time(time: datetime) -> datetime:
    return time.replace(microsecond=0)


def invert_dict(dict_val: dict) -> dict:
    return {v: k for k, v in dict_val.items()}


def merge_dicts(dict_1: dict, dict_2: dict) -> dict:
    return {**dict_1, **dict_2}


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
        MODULE_LOGGER.verbose(f"Creating {cls.__name__}: {data}")  # type: ignore[attr-defined]
        return data

    def plot(self, *args, **kwargs) -> Any:
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
    def _ignore_type_key(cls, data: dict[str, Any]) -> dict[str, Any]:
        if "type" in data:
            del data["type"]
        return data

    @model_validator(mode="before")
    @classmethod
    def _debug_print(cls, data: dict[str, Any]) -> dict[str, Any]:
        MODULE_LOGGER.verbose(f"Creating {cls.__name__}: {data}")  # type: ignore[attr-defined]
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
    def plot(self, ax: plt.Axes, data: gpd.GeoSeries) -> list[plt.Line2D]:
        # return data.plot(ax=ax, **self.kwargs)
        return ax.plot(data, **self.kwargs)


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
        MODULE_LOGGER.verbose(f"Filling defaults in {cls.__name__}")  # type: ignore[attr-defined]
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

        patches = []
        for plot_2d in self.plots.values():
            pp = plot_2d.plot(ax, data[self.targets])
            patches.extend(pp)

        ax.set_title(self.title)
        ax.set_xlabel(self.xlabel)
        ax.set_ylabel(self.ylabel)

        ax.legend(patches, self.targets)

        return ax


class TwinAxes2DDefaults(Axes2DDefaults):
    DEFAULTS: ClassVar[Axes2DDefaults] = Axes2DDefaults()

    twin_targets: list[str] = ["rssi"]

    ytwinlabel: Optional[str] = Field(default=None, required=False)

    @model_validator(mode="before")
    @classmethod
    def _fill_model_defaults(cls, data: dict[str, Any]) -> dict[str, Any]:
        MODULE_LOGGER.verbose(f"Filling defaults in {cls.__name__}")  # type: ignore[attr-defined]
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

        patches = []
        for plot_2d in self.plots.values():
            pp = plot_2d.plot(ax_left, data[self.targets])
            patches.extend(pp)
        ax_right._get_lines.prop_cycler = ax_left._get_lines.prop_cycler
        for plot_2d in self.plots.values():
            pp = plot_2d.plot(ax_right, data[self.twin_targets])
            patches.extend(pp)

        ax_left.set_title(self.title)
        ax_left.set_xlabel(self.xlabel)
        ax_left.set_ylabel(self.ylabel)
        ax_right.set_ylabel(self.ytwinlabel)

        # ax_left.legend()
        # ax_right.legend()
        ax_left.legend(patches, self.targets + self.twin_targets)

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
    ) -> list[Line3D]:
        return ax.plot3D(geometry.x, geometry.y, geometry.z, **self.kwargs)


class Stem3DSettings(Plot3DSettings):
    DEFAULT_DICT = dict(markerfmt=".", linefmt="C7:")

    def plot(
        self,
        ax: Axes3D,
        geometry: gpd.GeoSeries,
        *args,
        **kwargs,
    ) -> StemContainer:
        return ax.stem(
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
        **kwargs,
    ) -> PathCollection:
        return ax.scatter(
            geometry.x,
            geometry.y,
            geometry.z,
            c=values,
            **self.kwargs,
            **kwargs,
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
        MODULE_LOGGER.verbose(f"Filling defaults in {cls.__name__}")  # type: ignore[attr-defined]
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
        **kwargs,
    ) -> PathCollection:
        data = self.interpolate_data(geometry, values)
        return super(Interpolated3DSettings, self).plot(
            ax, data.geometry, data.target, **kwargs
        )


class Contour3DDefaults(Plot3DSettings, InterplolationSettings3D):
    number_contours: int = Field(default=5, required=False)
    limit_contours: list[int] | range = Field(default=[], required=False)

    @field_serializer("limit_contours")
    @classmethod
    def _validate_limit_contours(cls, limit_contours: list[int] | range) -> list[int]:
        return list(limit_contours)

    def plot(self, *args, **kwargs) -> list[Poly3DCollection]:
        raise NotImplementedError(
            f"Tried calling {__name__} method on {type(self).__name__} object"
        )


class Contour3DSettings(Contour3DDefaults):
    DEFAULT_DICT = dict(cmap="viridis")
    DEFAULTS: ClassVar[Contour3DDefaults] = Contour3DDefaults()

    @model_validator(mode="before")
    @classmethod
    def _fill_model_defaults(cls, data: dict[str, Any]) -> dict[str, Any]:
        MODULE_LOGGER.verbose(f"Filling defaults in {cls.__name__}")  # type: ignore[attr-defined]
        for field in type(cls.DEFAULTS).model_fields:
            if field not in data:
                data[field] = getattr(cls.DEFAULTS, field)
        return data

    def plot(
        self,
        ax: Axes3D,
        geometry: gpd.GeoSeries,
        values: gpd.GeoSeries,
        **kwargs,
    ) -> list[Poly3DCollection]:
        data = self.interpolate_data(geometry, values)
        values = data.target
        if len(self.limit_contours) == 0:
            self.limit_contours = range(self.number_contours)

        kwargs = merge_dicts(kwargs, self.kwargs.copy())

        colors = sns.color_palette(kwargs.pop("cmap"), n_colors=self.number_contours)
        data["category"] = pd.cut(values, bins=self.number_contours)

        poly_paths: list[Poly3DCollection]

        for i, (color, category) in enumerate(
            zip(colors, data.category.cat.categories)
        ):
            if i not in self.limit_contours:
                continue
            sub_data = data.loc[data.category == category, :]
            # sub_geometry = sub_data.geometry
            X, Y, Z = (
                sub_data.geometry.x.values,
                sub_data.geometry.y.values,
                sub_data.geometry.z.values,
            )
            poly_paths.append(ax.plot_trisurf(X, Y, Z, color=color, **kwargs))

        return poly_paths


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
        MODULE_LOGGER.verbose(f"Filling defaults in {cls.__name__}")  # type: ignore[attr-defined]
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
        **kwargs,
    ) -> Axes3D:
        if self.crs_target:
            data = data.to_crs(self.crs_target)

        ax = fig.add_subplot(*subplots, ax_index, projection="3d", **self.kwargs)
        for plot_3d in self.plots.values():
            ret = plot_3d.plot(
                ax, data.geometry, data[self.target], **kwargs, label=self.target
            )

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

    filters: list[str] = Field(default=[], required=False)

    title: str = Field(
        default="{experiment_name}",
        required=False,
        description="Can be a format string",
    )
    xlabel: str = Field(
        default="",
        required=False,
        description="Can be a format string",
    )
    ylabel: str = Field(
        default="",
        required=False,
        description="Can be a format string",
    )

    norm: Literal["linear", "log"] = Field(
        default="linear",
        required=False,
        description="Only affects the colorbar of 3D plots",
    )

    tight_layout: bool = Field(default=True, required=False)

    kwargs: dict[str, Any] = Field(
        default={}, required=False, description="additional kwargs for figure"
    )

    figure_name: str = Field(
        default="",
        required=False,
        description="Can be used to specify the output filename for the figure",
    )

    @field_validator("subfigures", mode="before")
    @classmethod
    def _parse_axes(cls, data: list[dict | AxesUnion]) -> list[AxesUnion]:
        MODULE_LOGGER.debug(f"Parsing subfigures: {data}")
        ret_data: list[AxesUnion] = []
        for axes_settings in data:
            if not isinstance(axes_settings, dict):
                MODULE_LOGGER.verbose(  # type: ignore[attr-defined]
                    f"Directly adding {type(axes_settings).__name__}({axes_settings.model_dump()}) to subfigures"
                )
                ret_data.append(axes_settings)
                continue

            axes_type = AxesType(axes_settings["type"])
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

    @field_validator("filters")
    @classmethod
    def _validate_filters(cls, filters: list[str]) -> list[str]:
        if any("\n" in fil for fil in filters):
            raise ValueError("Filters cannot contain newlines.")
        return filters

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
        MODULE_LOGGER.verbose(f"Creating {cls.__name__}: {data}")  # type: ignore[attr-defined]
        return data

    def validate_targets(self, values: list[str]) -> bool:
        return all(subfigure.validate_targets(values) for subfigure in self.subfigures)

    def filter_data(
        self, data: pd.DataFrame | gpd.GeoDataFrame
    ) -> pd.DataFrame | gpd.GeoDataFrame:
        rows_before = len(data.index)
        for fil in self.filters:
            data = data[eval(fil)]
        rows_after = len(data.index)
        MODULE_LOGGER.verbose(  # type: ignore[attr-defined]
            "Dropped %d rows due to set filters. %d rows remaining.",
            rows_before - rows_after,
            rows_after,
        )
        return data


class FigureSettings(FigureDefaults):
    DEFAULTS: ClassVar[FigureDefaults] = FigureDefaults()

    @model_validator(mode="before")
    @classmethod
    def _fill_model_defaults(cls, data: dict[str, Any]) -> dict[str, Any]:
        cls._validate_figure_settings(data)
        MODULE_LOGGER.verbose(f"Filling defaults in {cls.__name__}")  # type: ignore[attr-defined]
        for field in type(cls.DEFAULTS).model_fields:
            if field not in data:
                data[field] = copy.deepcopy(getattr(cls.DEFAULTS, field))
            if field in ["num_cols", "num_rows"] and data[field] < getattr(
                cls.DEFAULTS, field
            ):
                data[field] = getattr(cls.DEFAULTS, field)
        return data

    def format_string_fields(self, **kwargs):
        MODULE_LOGGER.debug(f"Formatting string fields with arguments: {kwargs}")
        MODULE_LOGGER.debug(
            f"Before:\ntitle: {self.title}\nxlabel: {self.xlabel}\nylabel: {self.ylabel}"
        )
        self.title = self.title.format(**kwargs)
        self.xlabel = self.xlabel.format(**kwargs)
        self.ylabel = self.ylabel.format(**kwargs)
        MODULE_LOGGER.debug(
            f"After:\ntitle: {self.title}\nxlabel: {self.xlabel}\nylabel: {self.ylabel}"
        )

    def annotate_figure(self, fig: plt.Figure):
        if self.title:
            fig.suptitle(self.title)
        if self.xlabel:
            fig.supxlabel(self.xlabel)
        if self.ylabel:
            fig.supylabel(self.ylabel)
        if self.tight_layout:
            fig.tight_layout()

    def plot(self, data: gpd.GeoDataFrame) -> plt.Figure:
        data = self.filter_data(data)
        fig = plt.figure(**self.kwargs)
        subplots = (self.num_rows, self.num_cols)

        fields = [
            e
            for e in data.columns.tolist()
            if e not in ["geometry", "host", "hostname", "type"]
        ]
        minmax = data.loc[:, fields].agg(["min", "max"])

        axes: list[plt.Axes | Axes3D] = []
        axes3d: list[Axes3D] = []
        cmap = "viridis"
        label = ""
        for subfig_index, subfigure in enumerate(self.subfigures):
            if isinstance(subfigure, Axes3DSettings):
                vmin, vmax = minmax.loc[["min", "max"], subfigure.target]
                match self.norm:
                    case "linear":
                        plot_norm = Normalize(vmin=vmin, vmax=vmax)
                    case "log":
                        if vmin == 0:
                            vmin = 1
                        plot_norm = LogNorm(vmin=vmin, vmax=vmax)
                ax = subfigure.plot(
                    fig,
                    subplots,
                    subfig_index + 1,
                    data,
                    norm=plot_norm,
                )
                last_3d_axes = ax
                axes3d.append(ax)
                cmap = subfigure.kwargs.get("cmap", cmap)
                label = subfigure.target
            else:
                ax = subfigure.plot(fig, subplots, subfig_index + 1, data)

        # if len(axes3d) > 1:
        #     for ax3d in axes3d[1:]:
        #         axes3d[0].shareview(ax3d)

        self.annotate_figure(fig)

        if len(axes3d) > 0:
            COLORBAR_WIDTH_IN = 1.0  # absolute width in inches

            width, height = fig.get_size_inches()
            right_adjust_frac = 1 - (COLORBAR_WIDTH_IN / width)
            fig.subplots_adjust(right=right_adjust_frac)
            cbar_ax = fig.add_axes(rect=(right_adjust_frac + 0.01, 0.02, 0.02, 0.925))

            vmin, vmax = minmax.loc[["min", "max"], label]
            match self.norm:
                case "linear":
                    bar_norm = Normalize(vmin=vmin, vmax=vmax)
                case "log":
                    if vmin == 0:
                        vmin = 1
                    bar_norm = LogNorm(vmin=vmin, vmax=vmax)

            fig.colorbar(
                plt.cm.ScalarMappable(cmap=cmap, norm=bar_norm),
                cax=cbar_ax,
                label=label,
            )

        return fig


class OffsetFigureSettings(FigureSettings):
    offset_range: range

    @field_validator("offset_range", mode="before")
    @classmethod
    def _validate_offset_range(cls, data: dict[str, int] | range) -> range:
        if isinstance(data, range):
            return data

        if "start" not in data:
            data["start"] = 0
        if "step" not in data:
            data["step"] = 1
        return range(data["start"], data["end"], data["step"])

    @field_serializer("offset_range")
    def _serialize_offset_range(self, value: range) -> dict[str, int]:
        return {
            "start": value.start,
            "end": value.stop,
            "step": value.step,
        }

    @model_validator(mode="wrap")
    def _set_figure_dimensions(self, handler: Callable) -> "OffsetFigureSettings":
        assert isinstance(self, dict)
        MODULE_LOGGER.verbose("Defining figure dimensions from data: %s", self)
        configured_dimensions = self.get("num_cols"), self.get("num_rows")

        model: "OffsetFigureSettings" = handler(self)

        match configured_dimensions:
            case (None, None):
                if len(model.subfigures) > 1:
                    model.num_cols = len(model.subfigures)
                    model.num_rows = len(model.offset_range)
                else:
                    model.num_cols = max(
                        int(np.ceil(np.sqrt(len(model.offset_range)))), 1
                    )
                    model.num_rows = max(
                        int(np.ceil(len(model.offset_range) / model.num_cols)), 1
                    )

            case (num_cols, None):
                model.num_rows = int(np.ceil(len(model.offset_range) / num_cols))
                model.num_cols = num_cols * len(model.subfigures)

            case (None, num_rows):
                model.num_cols = int(np.ceil(len(model.offset_range) / num_rows)) * len(
                    model.subfigures
                )
                model.num_rows = num_rows

            case (num_cols, num_rows):
                if num_cols * num_rows < len(model.offset_range):
                    raise ValueError(
                        "Number of rows and columns cannot fit the specified number of offsets"
                    )
                model.num_cols = num_cols * len(model.subfigures)

        MODULE_LOGGER.verbose(
            "Figure dimensions: %d cols, %d rows for %d subfigures with %d offsets",
            model.num_cols,
            model.num_rows,
            len(model.subfigures),
            len(model.offset_range),
        )
        return model

    @property
    def num_cols_per_subfigure(self) -> int:
        return self.num_cols // len(self.subfigures)

    def plot(self, data: Iterable[gpd.GeoDataFrame]) -> plt.Figure:
        fig = plt.figure(**self.kwargs)
        subplots = (self.num_rows, self.num_cols)

        axes: list[plt.Axes | Axes3D] = []
        axes3d: list[Axes3D] = []
        for subfig_index, subfigure in enumerate(self.subfigures):
            base_title = subfigure.title
            if base_title == "" or base_title is None:
                base_title = ""
            else:
                base_title = f"\n{base_title}"
            for offset_index, (datum, offset) in enumerate(
                zip(data, self.offset_range)
            ):
                col_index = offset_index // self.num_cols_per_subfigure
                subfigure.title = f"Offset: {offset} s{base_title}"
                MODULE_LOGGER.verbose(  # type: ignore[attr-defined]
                    "Plotting subfigure %d, offset %d (%s) at index %d",
                    subfig_index,
                    offset_index,
                    subfigure.title,
                    subfig_index * self.num_cols_per_subfigure
                    + self.num_cols * col_index
                    + offset_index % self.num_cols_per_subfigure
                    + 1,
                )
                ax = subfigure.plot(
                    fig,
                    subplots,
                    subfig_index * self.num_cols_per_subfigure
                    + self.num_cols * col_index
                    + offset_index % self.num_cols_per_subfigure
                    + 1,
                    datum,
                )
                axes.append(ax)
                if isinstance(subfigure, Axes3DSettings):
                    axes3d.append(ax)

        # if len(axes3d) > 1:
        #     for ax3d in axes3d[1:]:
        #         axes3d[0].shareview(ax3d)

        self.annotate_figure(fig)

        return fig


FiguresUnion = Annotated[
    FigureSettings | OffsetFigureSettings, "Union of possible figures"
]


class CorrelationCalcDefaults(BaseSettings):
    targets: list[str] = Field(default=["bps", "rssi"], required=True, min_length=2)

    filters: list[str] = Field(default=[], required=False)

    @field_validator("filters")
    @classmethod
    def _validate_filters(cls, filters: list[str]) -> list[str]:
        if any("\n" in fil for fil in filters):
            raise ValueError("Filters cannot contain newlines.")
        return filters

    def validate_targets(self, values: list[str]) -> bool:
        for target in self.targets:
            if target not in values:
                raise ValueError(
                    f"Calculation target '{target}' must be a key of {values}"
                )
        return True

    def filter_data(
        self, data: pd.DataFrame | gpd.GeoDataFrame
    ) -> pd.DataFrame | gpd.GeoDataFrame:
        rows_before = len(data.index)
        for fil in self.filters:
            data = data[eval(fil)]
        rows_after = len(data.index)
        MODULE_LOGGER.verbose(  # type: ignore[attr-defined]
            "Dropped %d rows due to set filters. %d rows remaining.",
            rows_before - rows_after,
            rows_after,
        )
        return data


class CorrelationCalcSettings(CorrelationCalcDefaults):
    """
    Settings for a correlation calculation.
    """

    DEFAULTS: ClassVar[CorrelationCalcDefaults] = CorrelationCalcDefaults()

    @model_validator(mode="before")
    @classmethod
    def _fill_model_defaults(cls, data: dict[str, Any]) -> dict[str, Any]:
        MODULE_LOGGER.verbose(f"Filling defaults in {cls.__name__}")  # type: ignore[attr-defined]
        for field in type(cls.DEFAULTS).model_fields:
            if field not in data:
                data[field] = copy.deepcopy(getattr(cls.DEFAULTS, field))
        return data

    def calculate(self, data: gpd.GeoDataFrame) -> pd.DataFrame:
        data = self.filter_data(data)
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

    figures: list[FiguresUnion] = Field(
        default=[
            FigureSettings(),
            FigureSettings(subfigures=[Axes3DSettings()]),
        ]
    )

    correlations: list[CorrelationCalcSettings] = Field(
        default=[],
        required=False,
    )

    rolling_window_period: int = Field(default=1, required=False)

    crs_target: Optional[str] = Field(default=None, required=False)

    output_file_template: Optional[str] = Field(default=None, required=False)

    output_kwargs: Optional[dict[str, Any]] = Field(
        default=dict(format="png", transparent=True, dpi=300),
        required=False,
    )

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

    @field_validator("output_file_template")
    @classmethod
    def _validate_output_file_template(cls, fmt: Optional[str]) -> Optional[str]:
        if fmt is None:
            return None

        for field in findall("{(.*?)}", fmt):
            if field not in FigureDefaults.model_fields.keys():
                raise ValueError(
                    f"Field '{field}' in output_file_template is not a valid field of FigureDefaults. Valid entires are {list(FigureDefaults.model_fields.keys())}",
                )
        return fmt

    @model_validator(mode="after")  # type: ignore[arg-type]
    def _validate_settings(self) -> "ExperimentDefaults":
        MODULE_LOGGER.debug("Validating experiment targets...")
        for figure in self.figures:
            figure.validate_targets(self.eval_field_names)
        for corr in self.correlations:
            corr.validate_targets(self.eval_field_names)

        return self

    @model_validator(mode="before")
    @classmethod
    def _debug_print(cls, data: dict[str, Any]) -> dict[str, Any]:
        MODULE_LOGGER.verbose(f"Creating {cls.__name__}: {data}")  # type: ignore[attr-defined]
        return data


class ExperimentSettings(ExperimentDefaults):
    """
    Settings for an experiment evaluation.
    """

    DEFAULTS: ClassVar[ExperimentDefaults] = ExperimentDefaults()

    experiment_name: str = Field(
        default="",
        required=False,
        description="Can be used in output filenames and figure titles via templates",
    )

    start: datetime
    end: datetime

    @model_validator(mode="before")
    @classmethod
    def _fill_model_defaults(cls, data: dict[str, Any]) -> dict[str, Any]:
        MODULE_LOGGER.verbose(f"Filling defaults in {cls.__name__}")  # type: ignore[attr-defined]
        for field in type(cls.DEFAULTS).model_fields:
            if field not in data:
                data[field] = copy.deepcopy(getattr(cls.DEFAULTS, field))
        return data

    @model_validator(mode="after")
    def _format_figure_strings(self) -> "ExperimentSettings":
        MODULE_LOGGER.debug("Formatting figure strings...")
        for figure in self.figures:
            figure.format_string_fields(
                **dict(
                    experiment_name=self.experiment_name,
                    rolling_window_period=self.rolling_window_period,
                )
            )

        return self

    @field_serializer("start", "end")
    def _serialize_start_end(self, value: datetime) -> str:
        return value.strftime("%Y-%m-%dT%H:%M:%S%z")

    def _get_offset_time(
        self, frame_type: FrameType, extra_offset: timedelta = timedelta(seconds=0)
    ) -> tuple[datetime, datetime]:
        start = self.start
        end = self.end

        if frame_type in self.retrieval_offsets:
            start += self.retrieval_offsets[frame_type] + extra_offset
            end += self.retrieval_offsets[frame_type] + extra_offset

        return start, end

    def uav_query(self, extra_offset: timedelta = timedelta(seconds=0)) -> str:
        start, end = self._get_offset_time(FrameType.UAV, extra_offset)

        return self.influxDB.build_query(
            start,
            end,
            f"""
            {InfluxDBSettings.build_filters("_measurement", ["drone_metrics"])}
            {InfluxDBSettings.build_filters("_field", self.uav_fields.values())}""",
        )

    def iperf_query(self, extra_offset: timedelta = timedelta(seconds=0)) -> str:
        start, end = self._get_offset_time(FrameType.IPERF, extra_offset)

        return self.influxDB.build_query(
            start,
            end,
            f"""
            {InfluxDBSettings.build_filters("_measurement", ["iperf3"])}
            {InfluxDBSettings.build_filters("_field", self.iperf_fields.values())}
            {InfluxDBSettings.build_filters("type", self.stream_types)}""",
        )

    def wireless_query(self, extra_offset: timedelta = timedelta(seconds=0)) -> str:
        start, end = self._get_offset_time(FrameType.WIRELESS, extra_offset)

        return self.influxDB.build_query(
            start,
            end,
            f"""
            {InfluxDBSettings.build_filters("_measurement", ["wireless"])}
            {InfluxDBSettings.build_filters("_field", self.wireless_fields.values())}""",
        )

    @staticmethod
    def _process_dataframe(data: pd.DataFrame) -> pd.DataFrame:
        MODULE_LOGGER.verbose(  # type: ignore[attr-defined]
            "Processing dataframe\n%s", data.describe() if not data.empty else data
        )
        if data.empty:
            raise ValueError(
                "DataFrame is empty. Perhaps incorrect filters were specified?"
            )
        data["_time"] = data["_time"].apply(floor_time)
        data = data.drop_duplicates(subset="_time")
        data = data.set_index("_time").drop(
            ["result", "table", "_start", "_stop", "_measurement"],
            axis=1,
            errors="ignore",
        )
        return data

    def _get_uav_data(
        self, extra_offset: timedelta = timedelta(seconds=0)
    ) -> pd.DataFrame:
        query = self.uav_query(extra_offset)
        MODULE_LOGGER.debug("UAV query: %s", query)
        data = self.influxDB.query_data_frame(query)

        if isinstance(data, list):
            raise ValueError("Got list of dataframes from uav query")

        if FrameType.UAV in self.retrieval_offsets:
            data["_time"] = data["_time"].apply(
                lambda t: t - self.retrieval_offsets[FrameType.UAV] - extra_offset
            )
        data = ExperimentSettings._process_dataframe(data)
        data = data.rename(columns=invert_dict(self.uav_fields))
        MODULE_LOGGER.debug("UAV data:\n%s", data.head(4))
        return data

    def _get_iperf_data(
        self, extra_offset: timedelta = timedelta(seconds=0)
    ) -> pd.DataFrame:
        query = self.iperf_query(extra_offset)
        MODULE_LOGGER.debug("Iperf query: %s", query)
        data = self.influxDB.query_data_frame(query)

        if isinstance(data, list):
            data = pd.concat(data)
            # raise ValueError("Got list of dataframes from iperf query")

        if FrameType.IPERF in self.retrieval_offsets:
            data["_time"] = data["_time"].apply(
                lambda t: t - self.retrieval_offsets[FrameType.IPERF] - extra_offset
            )
        data = ExperimentSettings._process_dataframe(data)
        data = data.rename(columns=invert_dict(self.iperf_fields))
        MODULE_LOGGER.debug("Iperf data:\n%s", data.head(4))
        return data

    def _get_wireless_data(
        self, extra_offset: timedelta = timedelta(seconds=0)
    ) -> pd.DataFrame:
        query = self.wireless_query(extra_offset)
        MODULE_LOGGER.debug("Wireless query: %s", query)
        data = self.influxDB.query_data_frame(query)

        if isinstance(data, list):
            raise ValueError("Got list of dataframes from wireless query")

        if FrameType.WIRELESS in self.retrieval_offsets:
            data["_time"] = data["_time"].apply(
                lambda t: t - self.retrieval_offsets[FrameType.WIRELESS] - extra_offset
            )
        data = ExperimentSettings._process_dataframe(data)
        data = data.rename(columns=invert_dict(self.wireless_fields))
        MODULE_LOGGER.debug("Wireless data:\n%s", data.head(4))
        return data

    def _collect_frames(
        self, extra_offset: timedelta = timedelta(seconds=0)
    ) -> pd.DataFrame:
        uav_data = self._get_uav_data(extra_offset)
        iperf_data = self._get_iperf_data(extra_offset)
        wireless_data = self._get_wireless_data(extra_offset)

        return uav_data.join(iperf_data).join(wireless_data)

    # @cached_property
    def get_geo_data(
        self, extra_offset: timedelta = timedelta(seconds=0)
    ) -> gpd.GeoDataFrame:
        data = self._collect_frames(extra_offset)

        MODULE_LOGGER.debug(
            "Collected data:\n%s", data.describe() if not data.empty else data
        )

        # fix zero-altitudes
        data["alt"] = data["alt"].replace(0, np.nan).interpolate(method="linear")
        # normalize to zero
        if "sea" in self.gps_fields["alt"]:
            data["alt"] = data["alt"] - data["alt"].min()

        rows_before = len(data.index)
        data = data.dropna(subset=self.gps_fields.keys())
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

        # print(geo_data.head(2))
        # geo_data = geo_data.loc[geo_data["mission_status"] == 2]

        return geo_data

    def get_offset_geo_data_list(self, offsets: range) -> list[gpd.GeoDataFrame]:
        return [self.get_geo_data(timedelta(seconds=offset)) for offset in offsets]

    @cached_property
    def cached_geo_data(self) -> gpd.GeoDataFrame:
        geo_data = self.get_geo_data()
        for field in self.eval_field_names:
            if field in ["mission_status"]:
                continue
            geo_data[field] = (
                geo_data[field]
                .rolling(window=f"{self.rolling_window_period}s", min_periods=1)
                .mean()
            )
        geo_data.index = geo_data.index.map(
            lambda x: (x - geo_data.index[0]).as_unit("s")
        )
        return geo_data

    def plot(self) -> list[plt.Figure]:
        figs = []
        for figure in self.figures:
            if isinstance(figure, OffsetFigureSettings):
                figs.append(
                    figure.plot(self.get_offset_geo_data_list(figure.offset_range))
                )
            else:
                figs.append(figure.plot(self.cached_geo_data))
            if self.output_file_template is not None:
                filename = (
                    self.output_file_template.format(**figure.model_dump())
                    .replace(" ", "-")
                    .replace("---", "-")
                    .lower()
                )
                path = Path(__file__).parent / filename
                path.parent.mkdir(parents=True, exist_ok=True)
                figs[-1].savefig(path, **self.output_kwargs)
        return figs

    def calculate(self) -> list[pd.DataFrame]:
        corrs = []
        for corr in self.correlations:
            corrs.append(corr.calculate(self.cached_geo_data))
        return corrs


class EvaluationEnvelope(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=("../.env", "../.env.eval", ".env", ".env.eval"),
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        env_prefix="EVAL_",
        json_file=("../eval.json", "../eval.jsonc", "eval.json", "eval.jsonc"),
        json_file_encoding="utf-8",
        extra="ignore",
    )  # type: ignore[typeddict-unknown-key]

    line2d_defaults: Line2DSettings = Field(default=Line2DSettings())
    path3d_defaults: Path3DSettings = Field(default=Path3DSettings())
    stem3d_defaults: Stem3DSettings = Field(default=Stem3DSettings())
    scatter3d_defaults: Scatter3DSettings = Field(default=Scatter3DSettings())
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

    experiment_defaults: ExperimentDefaults = Field(default=ExperimentDefaults())

    experiments: list[ExperimentSettings]

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

    @field_validator("experiment_defaults")
    @classmethod
    def _set_experiment_defaults(cls, value: ExperimentDefaults) -> ExperimentDefaults:
        MODULE_LOGGER.verbose("Setting experiment defaults")  # type: ignore[attr-defined]
        ExperimentSettings.DEFAULTS = value
        return value

    @field_validator("correlation_defaults")
    @classmethod
    def _set_correlation_defaults(
        cls, value: CorrelationCalcDefaults
    ) -> CorrelationCalcDefaults:
        MODULE_LOGGER.verbose("Setting correlation calculation defaults")  # type: ignore[attr-defined]
        CorrelationCalcSettings.DEFAULTS = value
        return value

    @field_validator("figure_defaults")
    @classmethod
    def _set_figure_defaults(cls, value: FigureDefaults) -> FigureDefaults:
        MODULE_LOGGER.verbose("Setting figure defaults")  # type: ignore[attr-defined]
        FigureSettings.DEFAULTS = value
        return value

    @field_validator("axes3d_defaults")
    @classmethod
    def _set_axes3d_defaults(cls, value: Axes3DDefaults) -> Axes3DDefaults:
        MODULE_LOGGER.verbose("Setting axes3D defaults")  # type: ignore[attr-defined]
        Axes3DSettings.DEFAULTS = value
        return value

    @field_validator("axes2d_defaults")
    @classmethod
    def _set_axes2d_defaults(cls, value: Axes2DDefaults) -> Axes2DDefaults:
        MODULE_LOGGER.verbose("Setting axes2D defaults")  # type: ignore[attr-defined]
        Axes2DSettings.DEFAULTS = value
        return value

    @field_validator("twin_axes2d_defaults")
    @classmethod
    def _set_twin_axes2d_defaults(cls, value: TwinAxes2DDefaults) -> TwinAxes2DDefaults:
        MODULE_LOGGER.verbose("Setting twin axes2D defaults")  # type: ignore[attr-defined]
        TwinAxes2DSettings.DEFAULTS = value
        return value

    @field_validator("line2d_defaults")
    @classmethod
    def _set_line2d_defaults(cls, value: Line2DSettings) -> Line2DSettings:
        MODULE_LOGGER.verbose("Setting line2D defaults")  # type: ignore[attr-defined]
        Line2DSettings.DEFAULT_DICT = value.kwargs
        return value

    @field_validator("path3d_defaults")
    @classmethod
    def _set_path3d_defaults(cls, value: Path3DSettings) -> Path3DSettings:
        MODULE_LOGGER.verbose("Setting path3D defaults")  # type: ignore[attr-defined]
        Path3DSettings.DEFAULT_DICT = value.kwargs
        return value

    @field_validator("stem3d_defaults")
    @classmethod
    def _set_stem3d_defaults(cls, value: Stem3DSettings) -> Stem3DSettings:
        MODULE_LOGGER.verbose("Setting stem3D defaults")  # type: ignore[attr-defined]
        Stem3DSettings.DEFAULT_DICT = value.kwargs
        return value

    @field_validator("scatter3d_defaults")
    @classmethod
    def _set_scatter3d_defaults(cls, value: Scatter3DSettings) -> Scatter3DSettings:
        MODULE_LOGGER.verbose("Setting scatter3D defaults")  # type: ignore[attr-defined]
        Scatter3DSettings.DEFAULT_DICT = value.kwargs
        return value

    @field_validator("interpolated3d_defaults")
    @classmethod
    def _set_interpolated3d_defaults(
        cls, value: Interpolated3DSettings
    ) -> Interpolated3DSettings:
        MODULE_LOGGER.verbose("Setting interpolated3D defaults")  # type: ignore[attr-defined]
        Interpolated3DSettings.DEFAULTS = InterpolationDefaults3D(
            num_points=value.num_points, interpolation_method=value.interpolation_method
        )
        Interpolated3DSettings.DEFAULT_DICT = value.kwargs
        return value

    @field_validator("contour3d_defaults")
    @classmethod
    def _set_contour3d_defaults(cls, value: Contour3DDefaults) -> Contour3DDefaults:
        MODULE_LOGGER.verbose("Setting contour3D defaults")  # type: ignore[attr-defined]
        Contour3DSettings.DEFAULT_DICT = value.kwargs
        Contour3DSettings.DEFAULTS = value
        return value

    @model_validator(mode="before")
    @classmethod
    def _debug_print(cls, data: dict[str, Any]) -> dict[str, Any]:
        MODULE_LOGGER.verbose(f"Creating {cls.__name__}: {data}")  # type: ignore[attr-defined]
        return data


def generate_template_file() -> None:
    model_config = EvaluationEnvelope.model_config
    EvaluationEnvelope.model_config = SettingsConfigDict(
        extra="ignore",
    )

    MODULE_LOGGER.info("Generating template files...")
    eval_settings = EvaluationEnvelope(
        experiments=[
            ExperimentSettings(
                start=datetime.now(ZoneInfo("Asia/Tokyo")),
                end=datetime.now(ZoneInfo("Europe/Berlin")),
                experiment_name="Example experiment",
                output_file_template="{title}-{figure_name}.png",
                retrieval_offsets={
                    FrameType.IPERF: timedelta(seconds=-200),
                    FrameType.WIRELESS: timedelta(seconds=150),
                    FrameType.UAV: timedelta(hours=2),
                },
            )
        ]
    )
    path = Path(__file__).parent / "eval.json.template"
    with open(path, "w") as file:
        json.dump(eval_settings.model_dump(mode="json"), file, indent=2)

    EvaluationEnvelope.model_config = model_config


####################################################################################################


def main():
    generate_template_file()

    MODULE_LOGGER.info("Start evaluation...")
    sns.set_style("darkgrid")
    sns.set_palette("colorblind")
    # Read settings
    eval_settings = EvaluationEnvelope()

    correlationsPath = Path(__file__).parent.parent / "output" / "correlations.json"
    if not correlationsPath.exists():
        correlations = {}
    else:
        with open(correlationsPath, "r") as file:
            correlations = json.load(file)

    # Evaluate experiments
    for experiment in eval_settings.experiments:
        MODULE_LOGGER.info(f"Evaluate experiment: {experiment.experiment_name}")
        experiment.plot()
        for result in experiment.calculate():
            print(result)
            if experiment.experiment_name not in correlations:
                correlations[experiment.experiment_name] = {}
            correlations[experiment.experiment_name][
                f"{experiment.rolling_window_period}s"
            ] = result.loc[result.index[-1], result.columns[0]]

    with open(correlationsPath, "w") as file:
        json.dump(correlations, file, indent=2)

    # plt.show()


if __name__ == "__main__":
    main()
