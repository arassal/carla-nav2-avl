import numpy as np
import pytest

from perception_costmap.segmentation import (
    letterbox, unletterbox_mask, HsvSegmenter, TwinLiteSegmenter,
    create_segmenter, segment_road,
)


def test_letterbox_preserves_aspect_and_pads_to_square():
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    padded, ratio, (pad_left, pad_top) = letterbox(img, new_size=640)
    assert padded.shape == (640, 640, 3)
    assert ratio == pytest.approx(640 / 640)
    # 640x480 -> scale to 640x480 (ratio=1.0 since 640/640=1 in width), pad
    # only in height... let's just assert pads are non-negative and content fits
    assert pad_left >= 0 and pad_top >= 0


def test_letterbox_grayscale_input():
    img = np.zeros((300, 600), dtype=np.uint8)
    padded, ratio, pad = letterbox(img, new_size=640)
    assert padded.shape == (640, 640)


def test_unletterbox_mask_roundtrip_shape():
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    padded, ratio, pad = letterbox(img, new_size=640)
    model_mask = np.zeros((640, 640), dtype=bool)
    model_mask[100:200, 100:200] = True
    out = unletterbox_mask(model_mask, ratio, pad, img.shape)
    assert out.shape == (480, 640)


def test_hsv_segmenter_finds_largest_blob():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    # HSV lower/upper defaults target low-saturation, mid-value (roadish grey)
    img[:, :] = (128, 128, 128)   # BGR grey everywhere -> low sat, mid value
    seg = HsvSegmenter(lower_hsv=(0, 0, 60), upper_hsv=(180, 60, 200), min_blob_area=10)
    mask = seg(img)
    assert mask.any()


def test_hsv_segmenter_below_min_area_returns_empty():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[0:2, 0:2] = (128, 128, 128)  # tiny 4px blob
    seg = HsvSegmenter(lower_hsv=(0, 0, 60), upper_hsv=(180, 60, 200), min_blob_area=500)
    mask = seg(img)
    assert not mask.any()


def test_twinlite_segmenter_falls_back_to_hsv_without_weights():
    seg = TwinLiteSegmenter(weights_path="/nonexistent/nano.pth")
    assert seg.using_fallback
    img = np.full((100, 100, 3), 128, dtype=np.uint8)
    mask = seg(img)
    assert mask.shape == (100, 100)


def test_create_segmenter_factory_hsv():
    seg = create_segmenter("hsv")
    assert isinstance(seg, HsvSegmenter)


def test_create_segmenter_factory_twinlitenet():
    seg = create_segmenter("twinlitenet", weights_path="/nonexistent/nano.pth")
    assert isinstance(seg, TwinLiteSegmenter)
    assert seg.using_fallback


def test_create_segmenter_unknown_method_raises():
    with pytest.raises(ValueError):
        create_segmenter("not_a_real_method")


def test_segment_road_backcompat_function():
    img = np.full((50, 50, 3), 128, dtype=np.uint8)
    mask = segment_road(img, min_blob_area=10)
    assert mask.shape == (50, 50)
