import json
from itertools import groupby
from pathlib import Path
from typing import Any, Callable, Optional

import matplotlib.pyplot as plt  # type: ignore[import]
import module_logging
import numpy as np
import pandas as pd  # type: ignore[import]
import seaborn as sns  # type: ignore[import]
from mpl_toolkits.mplot3d.axes3d import Axes3D  # type: ignore[import]
from pydantic import BaseModel, Field, model_serializer, model_validator
from pydantic_json_source import JsonConfigSettingsSource
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)
from scipy.spatial import distance  # type: ignore[import]

MODULE_LOGGER = module_logging.get_logger(module_logging.logging.INFO)


class MissionConfig(BaseModel):
    radius: float = Field(..., ge=0.0)
    numStops: int = Field(..., ge=0)
    altitude: float = Field(..., ge=0.0)


class MissionPoint(BaseModel):
    dlat: float
    dlon: float
    altitude: float = Field(..., ge=0.0)

    def to_ndarray(self) -> np.ndarray:
        return np.array([self.dlon, self.dlat, self.altitude])


class MissionConfigs(BaseModel):
    missions: list[MissionConfig]

    def append(self, mission: MissionConfig) -> None:
        self.missions.append(mission)

    def extend(self, other: "MissionConfigs") -> None:
        self.missions.extend(other.missions)


class MissionPoints(BaseModel):
    points: list[MissionPoint]

    def to_ndarray(self) -> np.ndarray:
        return np.array([point.to_ndarray() for point in self.points])

    def append(self, point: MissionPoint) -> None:
        self.points.append(point)

    def extend(self, other: "MissionPoints") -> None:
        self.points.extend(other.points)

    def keep(self, predicate: Callable[[MissionPoint], bool]) -> "MissionPoints":
        return MissionPoints(
            points=[point for point in self.points if predicate(point)]
        )

    def drop(self, predicate: Callable[[MissionPoint], bool]) -> "MissionPoints":
        return MissionPoints(
            points=[point for point in self.points if not predicate(point)]
        )


def generate_circular_3d_points(config: MissionConfig) -> MissionPoints:
    MODULE_LOGGER.debug(
        f"Generating {config.numStops} point{'s' if config.numStops > 1 else ''} at radius {config.radius} and height {config.altitude}"
    )
    theta = np.linspace(0, 2 * np.pi, config.numStops, endpoint=False)
    x = config.radius * np.cos(theta)
    y = config.radius * np.sin(theta)
    return MissionPoints(
        points=[
            MissionPoint(dlon=dlon, dlat=dlat, altitude=config.altitude)
            for dlon, dlat in zip(x, y)
        ]
    )


def calc_n_points_at_radius(radius: float, arc_segment_length: float) -> int:
    return np.round(2 * np.pi * radius / arc_segment_length)


def plot_reference_arcs(ax: Axes3D, radius: float, **kwargs):
    theta = np.linspace(0, np.pi, 50)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)

    ax.plot(x, y, zdir="y", **kwargs)
    ax.plot(x, y, zdir="x", **kwargs)


def calc_n_heights(
    radius: float, spacing: float, min_height: float, max_height: float
) -> np.ndarray:
    start_angle = np.arcsin(min_height / radius)
    end_angle = np.arcsin(max_height / radius)

    arc_length = (end_angle - start_angle) * radius

    n_heights = int(np.round(arc_length / spacing))
    MODULE_LOGGER.info(f"Generating {n_heights} heights")

    heights = np.linspace(min_height, max_height, n_heights, endpoint=True)
    return heights


def get_point_missions(
    radius: float,
    r_spacing: float,
    min_height: float,
    v_spacing: Optional[float] = None,
    max_height: Optional[float] = None,
) -> MissionConfigs:
    if max_height is None:
        max_height = radius

    if v_spacing is None:
        v_spacing = r_spacing

    heights = calc_n_heights(radius, v_spacing, min_height, max_height)
    MODULE_LOGGER.debug(f"Heights: {heights}")

    settings = MissionConfigs(missions=[])

    for height in heights:
        max_rad = np.sqrt(np.square(radius) - np.square(height))
        radii = np.arange(max_rad, 0, -r_spacing)
        if len(radii) == 0:
            radii = np.array([0])
        for r in radii[::-1]:
            n_points = int(calc_n_points_at_radius(r, r_spacing))
            n_points = max(n_points, 1)

            config = MissionConfig(radius=r, numStops=n_points, altitude=height)
            settings.append(config)

    return settings


def plot_point_missions(ax: Axes3D, settings: MissionConfigs, **kwargs) -> None:
    for config in settings.missions:
        points = generate_circular_3d_points(config)
        ax.scatter(*points.to_ndarray().T, **kwargs)


def get_path_missions(
    radius: float,
    r_spacing: float,
    min_height: float,
    v_spacing: float,
    max_height: float,
) -> MissionConfigs:
    heights = calc_n_heights(radius, v_spacing, min_height, max_height)
    MODULE_LOGGER.debug(f"Heights: {heights}")

    settings = MissionConfigs(missions=[])

    for height in heights:
        max_rad = np.sqrt(np.square(radius) - np.square(height))
        radii = np.arange(max_rad, 0, -r_spacing)
        for r in radii[::-1]:
            settings.append(MissionConfig(radius=r, numStops=0, altitude=height))

    return settings


def plot_path_missions(ax: Axes3D, settings: MissionConfigs, **kwargs) -> None:
    theta = np.linspace(0, 2 * np.pi, 50)

    for config in settings.missions:
        if config.radius == 0:
            ax.scatter(0, 0, config.altitude, **kwargs)
            continue

        x = config.radius * np.cos(theta)
        y = config.radius * np.sin(theta)
        ax.plot(x, y, config.altitude, **kwargs)


def order_points(settings: MissionConfigs) -> MissionPoints:
    missions = settings.missions.copy()
    all_points: MissionPoints = MissionPoints(points=[])
    all_points.extend(generate_circular_3d_points(missions.pop(0)))

    for i, (_, alt_missions) in enumerate(
        groupby(missions, lambda mission: mission.altitude)
    ):
        for mission in alt_missions if i % 2 == 0 else list(alt_missions)[::-1]:
            points = generate_circular_3d_points(mission)
            distances = distance.cdist(
                [all_points.points[-1].to_ndarray()], points.to_ndarray()
            )
            min_index = np.argmin(distances)
            all_points.points.extend(points.points[min_index:])
            all_points.points.extend(points.points[:min_index])

    return all_points


def plot_points_line(ax: Axes3D, points: MissionPoints, **kwargs) -> None:
    ax.plot(*points.to_ndarray().T, **kwargs)


def store_model(data: MissionConfigs, filename: str) -> None:
    path = Path(__file__).parent / Path(filename).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    MODULE_LOGGER.debug(f"Storing model to {path}")
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        json.dump(data.model_dump(mode="json"), f, indent=2)


class CoordSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=("../.env", "../.env.coords", ".env", ".env.coords"),
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        env_prefix="COORDS_",
        json_file="coords.json",
        json_file_encoding="utf-8",
        extra="ignore",
    )  # type: ignore[typeddict-unknown-key]

    radius: float = Field(default=100.0, ge=0.0)
    r_spacing: float = Field(
        default=25.0,
        gt=0.0,
        description="The radial spacing between points. If `v_spacing` is not specified, it will be the same as this.",
    )
    v_spacing: Optional[float] = Field(
        default=None,
        gt=0.0,
        description="The vertical spacing between points",
    )
    min_height: float = Field(
        default=0.0,
        ge=0.0,
    )
    max_height: Optional[float] = Field(
        default=None,
        gt=0.0,
    )

    generate_missions: bool = Field(
        False,
        description="Whether to generate seperate missions for each radius and height",
    )
    generate_points: bool = Field(
        False,
        description="Whether to generate all missions as points on a path in a single missions",
    )
    generate_paths: bool = Field(
        False,
        description="Whether to generate all missions as paths for HotPointMissions",
    )

    @model_validator(mode="after")
    def _model_validator(self) -> "CoordSettings":
        if self.v_spacing is None:
            self.v_spacing = self.r_spacing

        if self.max_height is None:
            self.max_height = self.radius

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


def generate_missions_points(
    ax: plt.Axes,
    radius: float,
    r_spacing: float,
    v_spacing: float,
    min_height: float,
    max_height: float,
    store_missions: bool,
    store_points: bool,
) -> None:
    MODULE_LOGGER.info("Generating points...")
    mission_configs = get_point_missions(
        radius=radius,
        r_spacing=r_spacing,
        v_spacing=v_spacing,
        min_height=min_height,
        max_height=max_height,
    )
    plot_point_missions(ax, mission_configs)

    all_points = MissionPoints(points=[])
    [
        all_points.extend(generate_circular_3d_points(mission))
        for mission in mission_configs.missions
    ]
    MODULE_LOGGER.info(f"Generated {len(all_points.points)} points")
    distances = distance.pdist(all_points.to_ndarray())
    MODULE_LOGGER.info(f"Min distance: {np.min(distances)}")
    MODULE_LOGGER.debug(f"Mean distance: {np.mean(distances)}")

    if store_missions:
        MODULE_LOGGER.info("Storing missions...")
        store_model(mission_configs, "../output/missions.jsonc")

    if not store_points:
        return

    ordered_points = order_points(mission_configs)
    filtered_points = ordered_points.drop(
        lambda point: 1.2 * point.dlon - 0.8 * point.dlat > 200.0
    )
    plot_points_line(ax, filtered_points)
    MODULE_LOGGER.info("Storing points...")
    store_model(filtered_points, "../output/points.jsonc")


def generate_paths(
    ax: plt.Axes,
    radius: float,
    r_spacing: float,
    v_spacing: float,
    min_height: float,
    max_height: float,
) -> None:
    MODULE_LOGGER.info("Generating paths...")
    mission_configs = get_path_missions(
        radius=radius,
        r_spacing=r_spacing,
        v_spacing=v_spacing,
        min_height=min_height,
        max_height=max_height,
    )
    plot_path_missions(ax, mission_configs)
    MODULE_LOGGER.info(f"Generated {len(mission_configs.missions)} paths")

    store_model(mission_configs, "../output/paths.jsonc")


def generate_template_file() -> None:
    model_config = CoordSettings.model_config
    CoordSettings.model_config = SettingsConfigDict(
        extra="ignore",
    )

    MODULE_LOGGER.info("Generating template file...")
    settings = CoordSettings()

    path = Path(__file__).parent / "coords.json.template"
    with open(path, "w") as file:
        json.dump(settings.model_dump(mode="json"), file, indent=2)

    CoordSettings.model_config = model_config


def main():
    generate_template_file()

    sns.set_style("darkgrid")
    sns.set_palette("colorblind")
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(10, 10))

    settings = CoordSettings()

    plot_reference_arcs(
        ax,
        radius=settings.radius,
        color="gray",
        linestyle=":",
    )

    if settings.generate_missions or settings.generate_points:
        generate_missions_points(
            ax=ax,
            radius=settings.radius,
            r_spacing=settings.r_spacing,
            v_spacing=settings.v_spacing,
            min_height=settings.min_height,
            max_height=settings.max_height,
            store_missions=settings.generate_missions,
            store_points=settings.generate_points,
        )

    if settings.generate_paths:
        generate_paths(
            ax=ax,
            radius=settings.radius,
            r_spacing=settings.r_spacing,
            v_spacing=settings.v_spacing,
            min_height=settings.min_height,
            max_height=settings.max_height,
        )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
