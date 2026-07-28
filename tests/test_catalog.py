from pathlib import Path

import pytest

from car_color_detector.catalog import ColorCatalog, ColorEntry, hex_to_rgb


def test_hex_to_rgb() -> None:
    assert hex_to_rgb("#12A0ff") == (18, 160, 255)


def test_hex_to_rgb_rejects_invalid_value() -> None:
    with pytest.raises(ValueError, match="Invalid HEX"):
        hex_to_rgb("#123")


def test_catalog_loads_csv_and_matches_color(tmp_path: Path) -> None:
    source = tmp_path / "colors.csv"
    source.write_text(
        "name,description,hex\nred,Красный,#C80000\nblue,Синий,#0000C8\n",
        encoding="utf-8",
    )
    catalog = ColorCatalog.from_csv(source)

    assert catalog.nearest_name((198, 5, 5)) == "Красный"


@pytest.mark.parametrize(
    ("rgb", "expected"),
    [
        ((245, 245, 245), "Белый"),
        ((20, 20, 20), "Чёрный"),
        ((130, 130, 130), "Серый"),
    ],
)
def test_catalog_uses_stable_neutral_labels(
    rgb: tuple[int, int, int],
    expected: str,
) -> None:
    catalog = ColorCatalog([ColorEntry("red", "Красный", (200, 0, 0))])

    assert catalog.nearest_name(rgb) == expected
