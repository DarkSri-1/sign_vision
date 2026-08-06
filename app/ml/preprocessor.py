import cv2
import numpy as np

try:
    from config import Config
except ImportError:
    Config = None

try:
    import mediapipe as mp
except ImportError:
    mp = None

IMG_SIZE = 64

# TF/Keras image_dataset_from_directory grayscale uses RGB luma (same as tf.image.rgb_to_grayscale)
_R = 0.2989
_G = 0.5870
_B = 0.1140

_hand_detector = None


def _use_hand_crop():
    if Config is None:
        return False
    return bool(getattr(Config, "USE_HAND_CROP_FOR_INFERENCE", False))


def _ensure_mediapipe():
    if mp is None:
        raise RuntimeError("MediaPipe is not installed.")


def _get_hand_detector():
    global _hand_detector
    _ensure_mediapipe()
    if _hand_detector is None:
        hands = mp.solutions.hands
        _hand_detector = hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    return _hand_detector


def center_square_crop(bgr):
    """Square crop from center (typical framing for static sign images)."""
    h, w = bgr.shape[:2]
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    return bgr[y0 : y0 + side, x0 : x0 + side]


def extract_hand_region_bgr(frame_bgr):
    """
    Returns (cropped_bgr, ok, message) where ok indicates hand found.
    """
    h, w = frame_bgr.shape[:2]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    detector = _get_hand_detector()
    res = detector.process(rgb)
    if not res.multi_hand_landmarks:
        return None, False, "No hand detected"
    lm = res.multi_hand_landmarks[0]
    xs = [p.x for p in lm.landmark]
    ys = [p.y for p in lm.landmark]
    pad = int(0.12 * max(w, h))
    x_min = int(max(0, min(xs) * w - pad))
    x_max = int(min(w, max(xs) * w + pad))
    y_min = int(max(0, min(ys) * h - pad))
    y_max = int(min(h, max(ys) * h + pad))
    if x_max <= x_min or y_max <= y_min:
        return None, False, "Invalid hand region"
    crop = frame_bgr[y_min:y_max, x_min:x_max]
    if crop.size == 0:
        return None, False, "Empty crop"
    return crop, True, "Hand detected"


def encode_for_model_sized(bgr, height, width, channels=1, pixel_range="0_1"):
    """
    Match TensorFlow training (streaming): bilinear resize, then either /255 or keep 0–255 float.
    channels=1: RGB luma; channels=3: RGB planes (for pretrained models expecting color).
    pixel_range: '0_1' (default, local training) or '0_255' when the model has Rescaling(1/255) inside.
    """
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    if channels == 1:
        gray = _R * rgb[:, :, 0] + _G * rgb[:, :, 1] + _B * rgb[:, :, 2]
        resized = cv2.resize(gray, (width, height), interpolation=cv2.INTER_LINEAR)
        if pixel_range == "0_255":
            norm = np.clip(resized, 0.0, 255.0)
        else:
            norm = resized / 255.0
        batch = np.expand_dims(np.expand_dims(norm.astype(np.float32), axis=-1), axis=0)
    elif channels == 3:
        resized = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_LINEAR)
        if pixel_range == "0_255":
            norm = np.clip(resized, 0.0, 255.0)
        else:
            norm = resized / 255.0
        batch = np.expand_dims(norm.astype(np.float32), axis=0)
    else:
        raise ValueError(f"Unsupported channel count for inference: {channels}")
    return batch


def encode_for_model(bgr):
    """Default 64×64 grayscale batch (matches project training)."""
    return encode_for_model_sized(bgr, IMG_SIZE, IMG_SIZE, 1)


def preprocess_for_model(image_bgr):
    """Alias for encode_for_model (single region already cropped)."""
    return encode_for_model(image_bgr)


def preprocess_uploaded_file(path, input_hwc=None, pixel_range="0_1"):
    img = cv2.imread(path)
    if img is None:
        raise ValueError("Could not read image file.")

    h, w, c = (input_hwc if input_hwc is not None else (IMG_SIZE, IMG_SIZE, 1))

    if _use_hand_crop():
        crop, ok, msg = extract_hand_region_bgr(img)
        if ok:
            return encode_for_model_sized(crop, h, w, c, pixel_range=pixel_range), True, msg
        sq = center_square_crop(img)
        return (
            encode_for_model_sized(sq, h, w, c, pixel_range=pixel_range),
            False,
            f"{msg} — using center crop (full image)",
        )

    sq = center_square_crop(img)
    return encode_for_model_sized(sq, h, w, c, pixel_range=pixel_range), True, "Full frame (matches training)"


def preprocess_frame_bytes(jpeg_bytes, input_hwc=None, pixel_range="0_1"):
    arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Invalid image data.")

    h, w, c = (input_hwc if input_hwc is not None else (IMG_SIZE, IMG_SIZE, 1))

    if _use_hand_crop():
        crop, ok, msg = extract_hand_region_bgr(frame)
        if not ok:
            return None, False, msg
        batch = encode_for_model_sized(crop, h, w, c, pixel_range=pixel_range)
        return batch, True, msg

    sq = center_square_crop(frame)
    batch = encode_for_model_sized(sq, h, w, c, pixel_range=pixel_range)
    return batch, True, "Frame processed (center crop, matches training)"
