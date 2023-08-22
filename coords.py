import json
import logging
from itertools import groupby
from typing import Any, Callable, Optional

import matplotlib.pyplot as plt  # type: ignore[import]
import numpy as np
import pandas as pd  # type: ignore[import]
import seaborn as sns  # type: ignore[import]
from mpl_toolkits.mplot3d.axes3d import Axes3D  # type: ignore[import]
from pydantic import BaseModel, Field, model_serializer
from scipy.spatial import distance  # type: ignore[import]

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

_handler = logging.StreamHandler()
_handler.setLevel(logger.getEffectiveLevel())

_formatter = logging.Formatter(
    "%(asctime)s %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)

_handler.setFormatter(_formatter)
logger.addHandler(_handler)


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
    logger.debug(
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
    logger.info(f"Generating {n_heights} heights")

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
    logger.debug(f"Heights: {heights}")

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
    v_spacing: Optional[float] = None,
    max_height: Optional[float] = None,
) -> MissionConfigs:
    if max_height is None:
        max_height = radius

    if v_spacing is None:
        v_spacing = r_spacing

    heights = calc_n_heights(radius, v_spacing, min_height, max_height)
    logger.debug(f"Heights: {heights}")

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


def save_settings(settings: MissionConfigs, filename: str) -> None:
    with open(filename, "w") as f:
        json.dump(settings.model_dump(mode="json"), f, indent=2)


def save_points(points: MissionPoints, filename: str) -> None:
    with open(filename, "w") as f:
        json.dump(points.model_dump(mode="json"), f, indent=2)


def main():
    sns.set_style("darkgrid")
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(10, 10))

    radius = 250
    r_spacing = 75
    v_spacing = 50
    min_height = 3
    max_height = 150

    plot_reference_arcs(
        ax,
        radius=radius,
        color="gray",
        linestyle=":",
    )

    logger.info("Generating points...")
    settings = get_point_missions(
        radius=radius,
        r_spacing=r_spacing,
        v_spacing=v_spacing,
        min_height=min_height,
        max_height=max_height,
    )
    plot_point_missions(ax, settings)

    all_points = MissionPoints(points=[])
    [
        all_points.extend(generate_circular_3d_points(mission))
        for mission in settings.missions
    ]
    logger.info(f"Generated {len(all_points.points)} points")
    distances = distance.pdist(all_points.to_ndarray())
    logger.info(f"Min distance: {np.min(distances)}")
    logger.debug(f"Mean distance: {np.mean(distances)}")

    save_settings(settings, "missions.jsonc")

    ordered_points = order_points(settings)
    filtered_points = ordered_points.drop(
        lambda point: point.altitude == 3.0 and point.dlon > 100.0
    )
    plot_points_line(ax, filtered_points)
    save_points(ordered_points, "points.jsonc")

    # logger.info("Generating paths...")
    # settings = get_path_missions(
    #     radius=radius,
    #     r_spacing=r_spacing,
    #     v_spacing=v_spacing,
    #     min_height=min_height,
    #     max_height=max_height,
    # )
    # plot_path_missions(ax, settings)
    # logger.info(f"Generated {len(settings.missions)} paths")

    # save_settings(settings, "paths.jsonc")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
