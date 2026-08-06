import os
import json

try:
    import tensorflow as tf
    from tensorflow import keras
except ImportError:
    keras = None

import numpy as np

# Keras image_dataset_from_directory on Kaggle "asl_alphabet" uses lexicographic folder order:
# A–Z, then del, nothing, space (29 classes). Many public ASL.h5 checkpoints use this.
_KAGGLE_ASL_29_CLASS_NAMES = [
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "J",
    "K",
    "L",
    "M",
    "N",
    "O",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "U",
    "V",
    "W",
    "X",
    "Y",
    "Z",
    "del",
    "nothing",
    "space",
]
_KAGGLE_ASL_26_CLASS_NAMES = [chr(65 + i) for i in range(26)]


def _builtin_asl_class_names(num_classes):
    if num_classes == len(_KAGGLE_ASL_29_CLASS_NAMES):
        return list(_KAGGLE_ASL_29_CLASS_NAMES)
    if num_classes == len(_KAGGLE_ASL_26_CLASS_NAMES):
        return list(_KAGGLE_ASL_26_CLASS_NAMES)
    return None


def _list_dataset_class_names(dataset_root):
    if not dataset_root or not os.path.isdir(dataset_root):
        return []
    return sorted(
        d
        for d in os.listdir(dataset_root)
        if os.path.isdir(os.path.join(dataset_root, d)) and not d.startswith(".")
    )


def _flatten_nested_layers(layer, seen, out):
    """Depth-first list of layers (handles nested Sequential/Functional submodels)."""
    if layer is None:
        return
    lid = id(layer)
    if lid in seen:
        return
    seen.add(lid)
    out.append(layer)
    inner = getattr(layer, "layers", None)
    if inner:
        for child in inner:
            _flatten_nested_layers(child, seen, out)


def _all_keras_layers(model):
    if model is None or not hasattr(model, "layers"):
        return []
    seen, out = set(), []
    for top in model.layers:
        _flatten_nested_layers(top, seen, out)
    return out


def _layer_applies_inv255_rescaling(layer):
    """
    True if layer is Keras Rescaling with scale ~ 1/255 (model expects raw 0–255 pixels).
    """
    if layer is None:
        return False
    name = layer.__class__.__name__
    if name != "Rescaling":
        return False
    scale = getattr(layer, "scale", None)
    if scale is None:
        return False
    try:
        s = float(np.asarray(scale).reshape(-1)[0])
    except (TypeError, ValueError):
        return False
    target = 1.0 / 255.0
    return abs(s - target) <= 0.001


def infer_input_pixel_range(model):
    """
    Return '0_255' if the model applies Rescaling(1/255) internally (common in Kaggle notebooks);
    otherwise '0_1' (pixels already normalized before the first conv / dense).
    """
    if model is None:
        return "0_1"
    for lay in _all_keras_layers(model):
        if _layer_applies_inv255_rescaling(lay):
            return "0_255"
    return "0_1"


def meta_input_pixel_range(model, meta):
    """Respect saved meta override; otherwise infer from the loaded model."""
    meta = meta or {}
    v = meta.get("input_pixel_range")
    if v in ("0_1", "0_255"):
        return v
    return infer_input_pixel_range(model)


def model_input_hwc(model):
    """Return (height, width, channels) for a Keras model's first input."""
    if model is None:
        return 64, 64, 1
    try:
        sh = model.input_shape
        if isinstance(sh, list):
            sh = sh[0] if sh else None
        if sh and len(sh) >= 4:
            h, w, c = int(sh[1]), int(sh[2]), int(sh[3])
            if h > 0 and w > 0 and c in (1, 3):
                return h, w, c
    except (TypeError, ValueError, IndexError):
        pass
    return 64, 64, 1


def load_model_bundle(model_path, meta_path):
    if keras is None or not os.path.isfile(model_path):
        return None, None
    try:
        model = keras.models.load_model(model_path, compile=False)
    except Exception:
        return None, None
    meta = {}
    if meta_path and os.path.isfile(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    return model, meta


def resolve_inference_paths(cfg):
    pre = cfg.get("PRETRAINED_MODEL_PATH")
    if pre and os.path.isfile(pre):
        return pre, cfg.get("PRETRAINED_META_PATH"), "pretrained"
    return cfg["MODEL_PATH"], cfg["MODEL_META_PATH"], "trained"


def prepare_meta_for_inference(model, meta, dataset_root):
    """class_names + input_pixel match the model; prefer standard ASL order over dataset folders."""
    meta = dict(meta or {})
    try:
        n_out = int(model.output_shape[-1])
    except (TypeError, ValueError, IndexError):
        meta["input_pixel_range"] = meta_input_pixel_range(model, meta)
        return meta
    names = meta.get("class_names")
    if not (isinstance(names, list) and len(names) == n_out):
        builtin = _builtin_asl_class_names(n_out)
        if builtin is not None:
            meta["class_names"] = builtin
        else:
            fallback = _list_dataset_class_names(dataset_root)
            if len(fallback) == n_out:
                meta["class_names"] = fallback
            else:
                meta["class_names"] = [str(i) for i in range(n_out)]
    meta["input_pixel_range"] = meta_input_pixel_range(model, meta)
    return meta


def load_inference_model(app):
    """
    Prefer admin-uploaded pretrained_asl.h5 when present; else local sign_model.keras.
    Returns (model, meta, source) where source is 'pretrained' or 'trained'.
    """
    mp, meta_path, source = resolve_inference_paths(app.config)
    model, meta = load_model_bundle(mp, meta_path)
    if model is None:
        return None, None, source
    meta = prepare_meta_for_inference(model, meta, app.config.get("DATASET_ROOT"))
    return model, meta, source


def predict_batch(model, meta, batch_nhwc):
    """batch_nhwc: numpy (1,64,64,1) float32"""
    if model is None:
        return None, 0.0
    preds = model.predict(batch_nhwc, verbose=0)
    idx = int(np.argmax(preds[0]))
    conf = float(preds[0][idx])
    names = meta.get("class_names") if meta else None
    if not names or idx >= len(names):
        return str(idx), conf
    return names[idx], conf
