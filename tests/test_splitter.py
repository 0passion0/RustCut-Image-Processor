from io import BytesIO
import numpy as np
from PIL import Image
from smart_splitter import SmartSplitter


def test_bento_xycut():
    img = np.full((400, 600, 3), 255, dtype=np.uint8)
    img[:, :298] = [24, 52, 110]
    img[:198, 299:] = [215, 72, 65]
    img[200:, 299:] = [32, 145, 112]

    splitter = SmartSplitter(mode="xycut", sensitivity=0.65, bleed_trim=1)
    res = splitter.analyze(img)
    assert len(res.boxes) == 3
    assert res.boxes[0].height > 390


def test_contour_objects():
    img = np.full((950, 1500, 3), 248, dtype=np.uint8)
    img[70:350, 80:400] = [29, 78, 216]
    img[90:330, 540:920] = [4, 120, 87]
    img[65:375, 1060:1400] = [190, 18, 60]
    img[490:830, 170:630] = [15, 23, 42]
    img[470:840, 810:1320] = [217, 119, 6]

    splitter = SmartSplitter(mode="auto", sensitivity=0.7)
    res, blobs = splitter.split_to_bytes(img, fmt="png")
    assert res.detected_mode == "contour"
    assert len(res.boxes) == 5
    assert len(blobs) == 5
    assert abs(res.boxes[0].width - 320) <= 4


def test_white_gutter_collage():
    img = np.full((600, 800, 3), 255, dtype=np.uint8)
    img[20:280, 20:380] = [30, 64, 175]
    img[20:280, 420:780] = [16, 185, 129]
    img[320:580, 20:380] = [225, 29, 72]
    img[320:580, 420:780] = [245, 158, 11]

    splitter = SmartSplitter(mode="xycut", sensitivity=0.65, bleed_trim=2)
    res = splitter.analyze(img)
    assert len(res.boxes) == 4
    for box in res.boxes:
        assert abs(box.width - 360) <= 2
        assert abs(box.height - 260) <= 2


def test_bleed_decontamination():
    img = np.full((400, 600, 3), 252, dtype=np.uint8)
    img[:, :299] = [18, 36, 74]
    img[:, 299] = [213, 214, 217]  # 1px gray divider hairline
    img[:, 300:] = [250, 251, 253]

    splitter = SmartSplitter(mode="xycut", sensitivity=0.7, bleed_trim=2)
    res = splitter.analyze(img)
    assert len(res.boxes) == 2
    left_box, right_box = sorted(res.boxes, key=lambda b: b.x)
    assert left_box.x + left_box.width <= 299
    assert right_box.x >= 300


def test_uniform_size_modes():
    img = np.full((600, 900, 3), 255, dtype=np.uint8)
    img[20:220, 20:320] = [24, 52, 110]
    img[20:280, 360:760] = [215, 72, 65]
    img[320:540, 20:420] = [32, 145, 112]

    # 1. uniform_size=True + uniform_mode="pad" -> 所有输出切片尺寸一致等于最大切片尺寸
    pad_splitter = SmartSplitter(mode="auto", uniform_size=True, uniform_mode="pad")
    res_pad, blobs_pad = pad_splitter.split_to_bytes(img, fmt="png")
    assert res_pad.backend == "rust-rayon"
    assert len(blobs_pad) == len(res_pad.boxes)
    decoded_pad = [Image.open(BytesIO(b)).size for b in blobs_pad]
    assert len(set(decoded_pad)) == 1

    # 2. target_size=(256, 192) + uniform_mode="resize" -> 严格匹配指定宽高
    resize_splitter = SmartSplitter(mode="auto", target_size=(256, 192), uniform_mode="resize")
    _, blobs_resize = resize_splitter.split_to_bytes(img, fmt="png")
    assert all(Image.open(BytesIO(b)).size == (256, 192) for b in blobs_resize)

    # 3. uniform_size=True + uniform_mode="crop" -> 原图检测框自身即等宽等高
    crop_splitter = SmartSplitter(mode="auto", uniform_size=True, uniform_mode="crop")
    res_crop, blobs_crop = crop_splitter.split_to_bytes(img, fmt="png")
    crop_box_dims = {(b.width, b.height) for b in res_crop.boxes}
    assert len(crop_box_dims) == 1
    assert all(Image.open(BytesIO(b)).size == next(iter(crop_box_dims)) for b in blobs_crop)

