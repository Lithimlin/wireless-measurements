import json
import logging
from typing import Any, Optional

import matplotlib.pyplot as plt  # type: ignore[import]
import numpy as np
import pandas as pd  # type: ignore[import]
import seaborn as sns  # type: ignore[import]
from mpl_toolkits.mplot3d.axes3d import Axes3D  # type: ignore[import]
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


def generate_circular_3d_points(
    radius: float, n_points: int, height: float
) -> np.ndarray:
    logger.debug(
        f"Generating {n_points} point{'s' if n_points > 1 else ''} at radius {radius} and height {height}"
    )
    theta = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    return np.vstack((x, y, np.ones(n_points) * height)).T


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


def plot_and_get_spaced_points(
    ax: Axes3D,
    radius: float,
    r_spacing: float,
    min_height: float,
    v_spacing: Optional[float] = None,
    max_height: Optional[float] = None,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    if max_height is None:
        max_height = radius

    if v_spacing is None:
        v_spacing = r_spacing

    heights = calc_n_heights(radius, v_spacing, min_height, max_height)
    logger.debug(f"Heights: {heights}")

    all_points = np.empty(shape=(0, 3))
    settings = np.empty(shape=(0, 3))

    for height in heights:
        max_rad = np.sqrt(np.square(radius) - np.square(height))
        radii = np.arange(max_rad, 0, -r_spacing)
        if len(radii) == 0:
            radii = np.array([0])
        for r in radii[::-1]:
            n_points = int(calc_n_points_at_radius(r, r_spacing))
            n_points = max(n_points, 1)

            points = generate_circular_3d_points(r, n_points, height)
            ax.scatter(*points.T, **kwargs)

            all_points = np.concatenate((all_points, points), axis=0)
            settings = np.append(settings, [[height, r, n_points]], axis=0)

    return all_points, settings


def plot_and_get_spaced_paths(
    ax: Axes3D,
    radius: float,
    r_spacing: float,
    min_height: float,
    v_spacing: Optional[float] = None,
    max_height: Optional[float] = None,
    **kwargs,
) -> np.ndarray:
    if max_height is None:
        max_height = radius

    if v_spacing is None:
        v_spacing = r_spacing

    heights = calc_n_heights(radius, v_spacing, min_height, max_height)
    logger.debug(f"Heights: {heights}")

    settings = np.empty(shape=(0, 2))

    for height in heights:
        if radius == height:
            ax.scatter(0, 0, height, **kwargs)
            continue

        max_rad = np.sqrt(np.square(radius) - np.square(height))
        radii = np.arange(max_rad, 0, -r_spacing)
        for r in radii[::-1]:
            theta = np.linspace(0, 2 * np.pi, 50)
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            ax.plot(x, y, height, **kwargs)

            settings = np.append(settings, [[height, r]], axis=0)

    return settings


def save_settings(settings: np.ndarray, filename: str) -> None:
    if settings.ndim == 2:
        settings = np.c_[settings, np.zeros(shape=(settings.shape[0]))]
    missions: list[dict] = [
        dict(altitude=setting[0], radius=setting[1], numStops=int(setting[2]))
        for setting in settings
    ]
    data = {"missions": missions}

    with open(filename, "w") as f:
        json.dump(data, f, indent=2)


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
    all_points, settings = plot_and_get_spaced_points(
        ax,
        radius=radius,
        r_spacing=r_spacing,
        v_spacing=v_spacing,
        min_height=min_height,
        max_height=max_height,
    )
    logger.info(f"Generated {all_points.shape[0]} points")
    distances = distance.pdist(all_points)
    logger.info(f"Min distance: {np.min(distances)}")
    logger.debug(f"Mean distance: {np.mean(distances)}")

    save_settings(settings, "missions.jsonc")

    logger.info("Generating paths...")
    settings = plot_and_get_spaced_paths(
        ax,
        radius=radius,
        r_spacing=r_spacing,
        v_spacing=v_spacing,
        min_height=min_height,
        max_height=max_height,
    )
    logger.info(f"Generated {settings.shape[0]} paths")

    save_settings(settings, "paths.jsonc")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
