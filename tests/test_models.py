from car_color_detector.models import BoundingBox


def test_bounding_box_expands_and_clamps_to_image() -> None:
    bbox = BoundingBox(5, 5, 95, 45)

    assert bbox.expand(0.1, width=100, height=50) == BoundingBox(0, 1, 100, 49)


def test_bounding_box_never_has_negative_size_after_clamp() -> None:
    bbox = BoundingBox(120, 70, 10, 5).clamp(width=100, height=50)

    assert bbox == BoundingBox(100, 50, 100, 50)
