import time
from flask import Blueprint, request, jsonify, current_app
from flask_login import login_required, current_user
from datetime import datetime
from app.helpers import log_prediction_db
from app.ml.predictor import load_inference_model, model_input_hwc, predict_batch
from app.ml.preprocessor import preprocess_frame_bytes
from app import db


bp = Blueprint("api", __name__)

_log_times = {}


def _should_log(uid, interval=1.8):
    now = time.time()
    last = _log_times.get(uid, 0)
    if now - last < interval:
        return False
    _log_times[uid] = now
    return True


@bp.route("/predict-frame", methods=["POST"])
@login_required
def predict_frame():
    if not request.data:
        return jsonify(ok=False, error="Empty body"), 400
    model, meta, _ = load_inference_model(current_app)
    if model is None:
        return jsonify(ok=False, error="No model available. Train a model or upload a pretrained .h5 in Admin."), 400
    hwc = model_input_hwc(model)
    px = (meta or {}).get("input_pixel_range")
    if px not in ("0_1", "0_255"):
        px = "0_1"
    t0 = time.perf_counter()
    try:
        batch, ok, status_msg = preprocess_frame_bytes(request.data, input_hwc=hwc, pixel_range=px)
    except ValueError as e:
        return jsonify(ok=False, error=str(e)), 400
    if not ok or batch is None:
        return jsonify(
            ok=True,
            hand_detected=False,
            status=status_msg,
            sign=None,
            confidence=0.0,
        )
    sign, conf = predict_batch(model, meta, batch)
    elapsed = (time.perf_counter() - t0) * 1000
    if _should_log(current_user.id):
        log_prediction_db(
            current_user.id,
            sign,
            conf,
            frame_ms=elapsed,
            extra=status_msg,
        )
    current_user.last_activity = datetime.utcnow()
    db.session.commit()
    return jsonify(
        ok=True,
        hand_detected=True,
        status="Prediction in progress",
        sign=sign,
        confidence=round(conf, 4),
        frame_ms=round(elapsed, 2),
    )


@bp.route("/classes")
@login_required
def classes():
    from app.helpers import dataset_class_counts

    c = dataset_class_counts(current_app.config["DATASET_ROOT"])
    return jsonify(classes=list(c.keys()), counts=c)
