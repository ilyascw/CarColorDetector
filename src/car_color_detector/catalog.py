"""Color catalog loading and RGB-to-name matching."""

from __future__ import annotations

import colorsys
import csv
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from car_color_detector.models import RgbColor


@dataclass(frozen=True, slots=True)
class ColorEntry:
    name: str
    description: str
    rgb: RgbColor


class ColorCatalog:
    """Maps RGB values to human-readable automotive color descriptions."""

    def __init__(self, entries: Iterable[ColorEntry]) -> None:
        self._entries = tuple(entries)
        if not self._entries:
            raise ValueError("Color catalog must contain at least one entry")

    @classmethod
    def from_csv(cls, path: str | Path) -> ColorCatalog:
        source = Path(path)
        try:
            with source.open(encoding="utf-8", newline="") as stream:
                reader = csv.DictReader(stream)
                required = {"name", "description", "hex"}
                if not reader.fieldnames or not required.issubset(reader.fieldnames):
                    raise ValueError(f"Color catalog must contain columns: {sorted(required)}")
                entries = [
                    ColorEntry(
                        name=row["name"].strip(),
                        description=row["description"].strip(),
                        rgb=hex_to_rgb(row["hex"]),
                    )
                    for row in reader
                ]
        except OSError as error:
            raise ValueError(f"Cannot read color catalog: {source}") from error
        return cls(entries)

    def nearest_name(self, rgb: RgbColor) -> str:
        """Return a neutral label or the nearest catalog color in HSV space."""

        red, green, blue = (channel / 255.0 for channel in rgb)
        hue, saturation, value = colorsys.rgb_to_hsv(red, green, blue)

        if saturation < 0.1:
            if value > 0.9:
                return "Белый"
            if value < 0.2:
                return "Чёрный"
            if value > 0.7:
                return "Светло-серый"
            if value > 0.4:
                return "Серый"
            return "Тёмно-серый"

        def distance(entry: ColorEntry) -> float:
            ref = tuple(channel / 255.0 for channel in entry.rgb)
            ref_hue, ref_saturation, ref_value = colorsys.rgb_to_hsv(*ref)
            hue_distance = min(abs(hue - ref_hue), 1 - abs(hue - ref_hue))
            return (
                hue_distance * 0.7
                + abs(saturation - ref_saturation) * 0.2
                + abs(value - ref_value) * 0.1
            )

        return min(self._entries, key=distance).description


def hex_to_rgb(value: str) -> RgbColor:
    normalized = value.strip().removeprefix("#")
    if len(normalized) != 6:
        raise ValueError(f"Invalid HEX color: {value!r}")
    try:
        channels = tuple(int(normalized[index : index + 2], 16) for index in (0, 2, 4))
    except ValueError as error:
        raise ValueError(f"Invalid HEX color: {value!r}") from error
    return channels[0], channels[1], channels[2]
