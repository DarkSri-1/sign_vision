import os
import uuid
from datetime import datetime
import shutil

from flask import (
    Blueprint,
    render_template,
    redirect,
    url_for,
    flash,
    request,
    current_app,
    Response,
    send_from_directory,
    abort,
)
from flask_login import login_required, current_user
from werkzeug.utils import secure_filename
from sqlalchemy import func, desc
from app import db
from app.models import User, RecognitionHistory, PredictionLog
from app.forms import ProfileForm, HistorySearchForm, RecognitionNoteForm
from app.helpers import dataset_class_counts, model_exists, export_history_csv, build_pdf_report
from app.ml.predictor import (
    load_inference_model,
    model_input_hwc,
    predict_batch,
    resolve_inference_paths,
)
from app.ml.preprocessor import preprocess_uploaded_file

bp = Blueprint("main", __name__)


@bp.route("/")
def landing():
    return render_template("landing.html")


@bp.route("/dashboard")
@login_required
def dashboard():
    uid = current_user.id
    total = RecognitionHistory.query.filter_by(user_id=uid).count()
    last = (
        RecognitionHistory.query.filter_by(user_id=uid)
        .order_by(desc(RecognitionHistory.created_at))
        .first()
    )
    recent = (
        RecognitionHistory.query.filter_by(user_id=uid)
        .order_by(desc(RecognitionHistory.created_at))
        .limit(8)
        .all()
    )
    avg_conf = db.session.query(func.avg(RecognitionHistory.confidence)).filter(
        RecognitionHistory.user_id == uid
    ).scalar()
    today = datetime.utcnow().date()
    today_count = RecognitionHistory.query.filter(
        RecognitionHistory.user_id == uid,
        RecognitionHistory.created_at >= datetime.combine(today, datetime.min.time()),
    ).count()
    top_signs = (
        db.session.query(
            RecognitionHistory.predicted_sign,
            func.count(RecognitionHistory.id).label("cnt"),
        )
        .filter(RecognitionHistory.user_id == uid)
        .group_by(RecognitionHistory.predicted_sign)
        .order_by(desc("cnt"))
        .limit(6)
        .all()
    )
    webcam_count = RecognitionHistory.query.filter_by(user_id=uid, source="webcam").count()
    upload_count = RecognitionHistory.query.filter_by(user_id=uid, source="upload").count()
    classes = dataset_class_counts(current_app.config["DATASET_ROOT"])
    has_model = model_exists(current_app)
    _, _, inference_source = resolve_inference_paths(current_app.config)
    inference_label = (
        "Pretrained model (admin upload)" if inference_source == "pretrained" else "Locally trained model"
    )
    return render_template(
        "user/dashboard.html",
        total_attempts=total,
        last_prediction=last,
        recent=recent,
        avg_confidence=float(avg_conf or 0),
        class_counts=classes,
        has_model=has_model,
        today_count=today_count,
        top_signs=top_signs,
        webcam_count=webcam_count,
        upload_count=upload_count,
        inference_label=inference_label,
    )


@bp.route("/profile", methods=["GET", "POST"])
@login_required
def profile():
    form = ProfileForm(obj=current_user)
    if form.validate_on_submit():
        other = User.query.filter(
            User.username == form.username.data.strip(), User.id != current_user.id
        ).first()
        if other:
            flash("Username already taken.", "danger")
            return render_template("user/profile.html", form=form)
        current_user.full_name = form.full_name.data.strip()
        current_user.username = form.username.data.strip()
        if form.new_password.data:
            if not form.current_password.data or not current_user.check_password(
                form.current_password.data
            ):
                flash("Current password required and must be correct to set a new password.", "danger")
                return render_template("user/profile.html", form=form)
            current_user.set_password(form.new_password.data)
        current_user.last_activity = datetime.utcnow()
        db.session.commit()
        flash("Profile updated.", "success")
        return redirect(url_for("main.profile"))
    activity_count = RecognitionHistory.query.filter_by(user_id=current_user.id).count()
    return render_template(
        "user/profile.html",
        form=form,
        activity_count=activity_count,
    )


@bp.route("/recognition")
@login_required
def recognition():
    classes = list(dataset_class_counts(current_app.config["DATASET_ROOT"]).keys())
    has_model = model_exists(current_app)
    meta = {}
    if has_model:
        _, meta, _ = load_inference_model(current_app)
    return render_template(
        "user/recognition.html",
        classes=classes,
        has_model=has_model,
        class_names=meta.get("class_names", classes) if meta else classes,
    )


def _apply_history_filters(query, form):
    q = query
    if form.sign.data and form.sign.data.strip():
        q = q.filter(
            RecognitionHistory.predicted_sign.ilike(f"%{form.sign.data.strip()}%")
        )
    if form.source.data:
        q = q.filter(RecognitionHistory.source == form.source.data)
    if form.date_from.data:
        try:
            d = datetime.strptime(form.date_from.data.strip(), "%Y-%m-%d")
            q = q.filter(RecognitionHistory.created_at >= d)
        except ValueError:
            flash("Invalid from date. Use YYYY-MM-DD.", "warning")
    if form.date_to.data:
        try:
            d = datetime.strptime(form.date_to.data.strip(), "%Y-%m-%d")
            from datetime import timedelta

            q = q.filter(RecognitionHistory.created_at < d + timedelta(days=1))
        except ValueError:
            flash("Invalid to date. Use YYYY-MM-DD.", "warning")
    return q


@bp.route("/history", methods=["GET"])
@login_required
def history():
    form = HistorySearchForm(formdata=request.args)
    q = RecognitionHistory.query.filter_by(user_id=current_user.id)
    if request.args and form.validate():
        q = _apply_history_filters(q, form)
    rows = q.order_by(desc(RecognitionHistory.created_at)).limit(500).all()
    return render_template("user/history.html", form=form, rows=rows)


@bp.route("/history/<int:hid>")
@login_required
def history_detail(hid):
    r = RecognitionHistory.query.get_or_404(hid)
    if r.user_id != current_user.id:
        flash("You do not have access to this record.", "danger")
        return redirect(url_for("main.history"))
    note_form = RecognitionNoteForm(obj=r)
    has_upload_image = bool(
        r.source_image_relpath and r.source == "upload"
    )
    return render_template(
        "user/history_detail.html",
        record=r,
        note_form=note_form,
        has_upload_image=has_upload_image,
    )


@bp.route("/history/<int:hid>/image")
@login_required
def history_upload_image(hid):
    r = RecognitionHistory.query.get_or_404(hid)
    if r.user_id != current_user.id:
        abort(403)
    if not r.source_image_relpath or r.source != "upload":
        abort(404)
    filename = os.path.basename(r.source_image_relpath)
    if filename != f"{hid}{os.path.splitext(filename)[1]}":
        abort(404)
    directory = os.path.join(current_app.config["HISTORY_UPLOAD_DIR"], str(r.user_id))
    path = os.path.join(directory, filename)
    if not os.path.isfile(path):
        abort(404)
    return send_from_directory(directory, filename)


@bp.route("/history/<int:hid>/note", methods=["POST"])
@login_required
def history_update_note(hid):
    r = RecognitionHistory.query.get_or_404(hid)
    if r.user_id != current_user.id:
        flash("Unauthorized.", "danger")
        return redirect(url_for("main.history"))
    form = RecognitionNoteForm()
    if form.validate_on_submit():
        r.user_note = (form.user_note.data or "").strip() or None
        db.session.commit()
        flash("Note saved.", "success")
    else:
        flash("Could not save note. Check length (max 500 characters).", "warning")
    return redirect(url_for("main.history_detail", hid=hid))


@bp.route("/history/bulk-delete", methods=["POST"])
@login_required
def history_bulk_delete():
    raw_ids = request.form.getlist("ids")
    try:
        ids = [int(x) for x in raw_ids if str(x).strip().isdigit()]
    except ValueError:
        ids = []
    if not ids:
        flash("No records selected.", "warning")
        return redirect(url_for("main.history"))
    q = RecognitionHistory.query.filter(
        RecognitionHistory.user_id == current_user.id,
        RecognitionHistory.id.in_(ids),
    )
    n = q.delete(synchronize_session=False)
    db.session.commit()
    flash(f"Deleted {n} record(s).", "info")
    return redirect(url_for("main.history"))


@bp.route("/history/delete/<int:hid>", methods=["POST"])
@login_required
def history_delete_one(hid):
    r = RecognitionHistory.query.get_or_404(hid)
    if r.user_id != current_user.id:
        flash("Unauthorized.", "danger")
        return redirect(url_for("main.history"))
    db.session.delete(r)
    db.session.commit()
    flash("Record deleted.", "info")
    return redirect(url_for("main.history"))


@bp.route("/history/clear", methods=["POST"])
@login_required
def history_clear():
    RecognitionHistory.query.filter_by(user_id=current_user.id).delete()
    db.session.commit()
    flash("All history cleared.", "info")
    return redirect(url_for("main.history"))


@bp.route("/history/export.csv")
@login_required
def history_export_csv():
    rows = (
        RecognitionHistory.query.filter_by(user_id=current_user.id)
        .order_by(desc(RecognitionHistory.created_at))
        .all()
    )
    data = export_history_csv(rows)
    return Response(
        data,
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment;filename=recognition_history.csv"},
    )


@bp.route("/history/export.pdf")
@login_required
def history_export_pdf():
    rows = (
        RecognitionHistory.query.filter_by(user_id=current_user.id)
        .order_by(desc(RecognitionHistory.created_at))
        .limit(200)
        .all()
    )
    lines = [
        f"User: {current_user.username}",
        f"Exported: {datetime.utcnow().isoformat()}",
        "",
    ]
    for r in rows:
        lines.append(
            f"{r.created_at} | {r.predicted_sign} | {r.confidence:.2%}"
        )
    pdf = build_pdf_report("Recognition History", lines)
    if pdf is None:
        flash("PDF export unavailable (install reportlab).", "warning")
        return redirect(url_for("main.history"))
    return Response(
        pdf,
        mimetype="application/pdf",
        headers={"Content-Disposition": "attachment;filename=history.pdf"},
    )


@bp.route("/help")
def help_page():
    return render_template("help.html")


@bp.route("/evaluate")
@login_required
def evaluate():
    import json

    app = current_app
    _, _, source = resolve_inference_paths(app.config)
    meta_path = app.config["MODEL_META_PATH"]

    if source == "pretrained":
        model, inf_meta, _ = load_inference_model(app)
        if model is None:
            flash("No model available.", "warning")
            return redirect(url_for("main.dashboard"))
        meta = dict(inf_meta or {})
        meta["using_pretrained_inference"] = True
        if os.path.isfile(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                local = json.load(f)
            meta["local_training_note"] = (
                "Figures below include your last local training run (sign_model.keras) where available; "
                "live recognition uses the uploaded pretrained weights."
            )
            for k in ("test_accuracy", "samples_per_class", "classification_report"):
                if k not in meta or meta.get(k) in (None, {}, []):
                    if local.get(k) is not None:
                        meta[k] = local.get(k)
        else:
            meta.setdefault("test_accuracy", None)
            meta.setdefault("samples_per_class", {})
            meta.setdefault("classification_report", {})
            meta["pretrained_only_note"] = (
                "Only the uploaded pretrained model is present. Train locally to see accuracy and a full "
                "classification report for sign_model.keras."
            )
        return render_template("user/evaluate.html", meta=meta)

    if not os.path.isfile(meta_path):
        flash("Train a model first to see evaluation metrics.", "warning")
        return redirect(url_for("main.dashboard"))
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return render_template("user/evaluate.html", meta=meta)


@bp.route("/upload-predict", methods=["POST"])
@login_required
def upload_predict():
    if "file" not in request.files:
        flash("No file.", "danger")
        return redirect(url_for("main.recognition"))
    f = request.files["file"]
    if not f.filename:
        flash("Empty filename.", "danger")
        return redirect(url_for("main.recognition"))
    name = secure_filename(f.filename)
    ext = os.path.splitext(name)[1].lower()
    if ext not in (".png", ".jpg", ".jpeg", ".webp", ".bmp"):
        flash("Invalid image type.", "danger")
        return redirect(url_for("main.recognition"))
    uid = uuid.uuid4().hex
    path = os.path.join(current_app.config["UPLOAD_FOLDER"], uid + ext)
    f.save(path)
    model, meta, _ = load_inference_model(current_app)
    if model is None:
        flash("No model found. Train a model or upload a pretrained .h5 in Admin.", "danger")
        return redirect(url_for("main.recognition"))
    try:
        hwc = model_input_hwc(model)
        px = (meta or {}).get("input_pixel_range")
        if px not in ("0_1", "0_255"):
            px = "0_1"
        batch, hand_ok, msg = preprocess_uploaded_file(path, input_hwc=hwc, pixel_range=px)
    except ValueError as e:
        flash(str(e), "danger")
        return redirect(url_for("main.recognition"))
    sign, conf = predict_batch(model, meta, batch)

    rh = RecognitionHistory(
        user_id=current_user.id,
        predicted_sign=sign,
        confidence=conf,
        source="upload",
    )
    db.session.add(rh)
    db.session.flush()
    rid = rh.id
    user_dir = os.path.join(current_app.config["HISTORY_UPLOAD_DIR"], str(current_user.id))
    os.makedirs(user_dir, exist_ok=True)
    final_name = f"{rid}{ext}"
    final_abs = os.path.join(user_dir, final_name)
    try:
        shutil.copy2(path, final_abs)
        rh.source_image_relpath = f"{current_user.id}/{final_name}"
    except OSError:
        pass
    db.session.add(
        PredictionLog(
            user_id=current_user.id,
            predicted_class=sign,
            confidence=conf,
            frame_time_ms=None,
            extra=f"upload:{hand_ok}:{msg}",
        )
    )
    db.session.commit()
    try:
        os.remove(path)
    except OSError:
        pass

    flash(f"Predicted: {sign} ({conf:.1%}). {msg}", "success" if hand_ok else "warning")
    return redirect(url_for("main.recognition"))


@bp.route("/save-capture", methods=["POST"])
@login_required
def save_capture():
    raw = request.get_data()
    if not raw:
        flash("No image data.", "danger")
        return redirect(url_for("main.recognition"))
    fn = f"capture_{current_user.id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.jpg"
    path = os.path.join(current_app.config["CAPTURE_SAVE_DIR"], fn)
    with open(path, "wb") as out:
        out.write(raw)
    flash(f"Saved: {fn}", "success")
    return redirect(url_for("main.recognition"))
