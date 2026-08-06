import os
import json
import shutil
from datetime import datetime, timedelta
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
from flask_login import current_user
from sqlalchemy import func, desc
from werkzeug.utils import secure_filename
from app import db
from app.models import User, RecognitionHistory, PredictionLog
from app.forms import (
    TrainForm,
    AdminUserCreateForm,
    AdminUserEditForm,
    AdminHistoryFilterForm,
    AdminLogFilterForm,
    AdminDatasetRenameForm,
    AdminRecognitionNoteForm,
)
from app.helpers import dataset_class_counts, export_history_csv, build_pdf_report, model_exists
from app.ml.trainer import train_and_save
from app.ml.predictor import load_model_bundle, resolve_inference_paths, infer_input_pixel_range

bp = Blueprint("admin", __name__)


def _parse_class_names_field(text):
    if not text or not str(text).strip():
        return None
    out = []
    for line in str(text).replace(",", "\n").split("\n"):
        s = line.strip()
        if s:
            out.append(s)
    return out or None


def _count_admins():
    return User.query.filter_by(is_admin=True).count()


def _delete_history_image_file(app, record):
    if not record or not getattr(record, "source_image_relpath", None):
        return
    try:
        base = app.config["HISTORY_UPLOAD_DIR"]
        rel = record.source_image_relpath.replace("/", os.sep)
        path = os.path.join(base, rel)
        if os.path.isfile(path):
            os.remove(path)
    except OSError:
        pass


def _apply_admin_history_filters(query, form):
    q = query
    uid_raw = (form.user_id.data or "").strip()
    if uid_raw.isdigit():
        q = q.filter(RecognitionHistory.user_id == int(uid_raw))
    if form.sign.data and form.sign.data.strip():
        q = q.filter(RecognitionHistory.predicted_sign.ilike(f"%{form.sign.data.strip()}%"))
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
            q = q.filter(RecognitionHistory.created_at < d + timedelta(days=1))
        except ValueError:
            flash("Invalid to date. Use YYYY-MM-DD.", "warning")
    return q


@bp.before_request
def _require_admin():
    from flask import abort

    if not current_user.is_authenticated:
        return redirect(url_for("auth.login", next=request.path))
    if not current_user.is_admin:
        abort(403)


@bp.route("/")
def dashboard():
    total_users = User.query.count()
    active_users = User.query.filter_by(is_active=True).count()
    total_preds = RecognitionHistory.query.count()
    log_count = PredictionLog.query.count()
    today = datetime.utcnow().date()
    daily = RecognitionHistory.query.filter(
        RecognitionHistory.created_at >= datetime.combine(today, datetime.min.time())
    ).count()
    top = (
        db.session.query(
            RecognitionHistory.predicted_sign,
            func.count(RecognitionHistory.id).label("c"),
        )
        .group_by(RecognitionHistory.predicted_sign)
        .order_by(desc("c"))
        .limit(8)
        .all()
    )
    recent_users = User.query.order_by(desc(User.created_at)).limit(8).all()
    recent_recs = (
        RecognitionHistory.query.order_by(desc(RecognitionHistory.created_at)).limit(12).all()
    )
    counts = dataset_class_counts(current_app.config["DATASET_ROOT"])
    has_model = model_exists(current_app)
    _, _, inference_source = resolve_inference_paths(current_app.config)
    inference_label = (
        "Pretrained (ASL.h5)" if inference_source == "pretrained" else "Local sign_model.keras"
    )
    return render_template(
        "admin/dashboard.html",
        total_users=total_users,
        active_users=active_users,
        total_preds=total_preds,
        daily=daily,
        top_signs=top,
        recent_users=recent_users,
        recent_recs=recent_recs,
        dataset_classes=len(counts),
        total_dataset_images=sum(counts.values()),
        log_count=log_count,
        has_model=has_model,
        inference_label=inference_label,
    )


@bp.route("/users")
def users():
    rows = User.query.order_by(User.id).all()
    return render_template("admin/users.html", rows=rows)


@bp.route("/users/new", methods=["GET", "POST"])
def user_create():
    form = AdminUserCreateForm()
    if form.validate_on_submit():
        email = form.email.data.strip().lower()
        username = form.username.data.strip()
        if User.query.filter_by(email=email).first():
            flash("Email already registered.", "danger")
            return render_template("admin/user_create.html", form=form)
        if User.query.filter_by(username=username).first():
            flash("Username already taken.", "danger")
            return render_template("admin/user_create.html", form=form)
        u = User(
            full_name=form.full_name.data.strip(),
            email=email,
            username=username,
            is_admin=bool(form.is_admin.data),
            is_active=True,
        )
        u.set_password(form.password.data)
        db.session.add(u)
        db.session.commit()
        flash("User created successfully.", "success")
        return redirect(url_for("admin.user_detail", uid=u.id))
    return render_template("admin/user_create.html", form=form)


@bp.route("/users/<int:uid>")
def user_detail(uid):
    u = User.query.get_or_404(uid)
    rec_count = RecognitionHistory.query.filter_by(user_id=u.id).count()
    last_rec = (
        RecognitionHistory.query.filter_by(user_id=u.id)
        .order_by(desc(RecognitionHistory.created_at))
        .first()
    )
    return render_template(
        "admin/user_detail.html",
        user=u,
        rec_count=rec_count,
        last_recognition=last_rec,
    )


@bp.route("/users/<int:uid>/edit", methods=["GET", "POST"])
def user_edit(uid):
    u = User.query.get_or_404(uid)
    form = AdminUserEditForm()
    if request.method == "GET":
        form.full_name.data = u.full_name
        form.email.data = u.email
        form.username.data = u.username
        form.is_admin.data = u.is_admin
        form.is_active.data = u.is_active
    if form.validate_on_submit():
        email = form.email.data.strip().lower()
        username = form.username.data.strip()
        if User.query.filter(User.email == email, User.id != u.id).first():
            flash("Email already in use.", "danger")
            return render_template("admin/user_edit.html", form=form, user=u)
        if User.query.filter(User.username == username, User.id != u.id).first():
            flash("Username already taken.", "danger")
            return render_template("admin/user_edit.html", form=form, user=u)
        if not form.is_admin.data and u.is_admin and _count_admins() <= 1:
            flash("Cannot remove administrator role from the only admin account.", "danger")
            return render_template("admin/user_edit.html", form=form, user=u)
        if not form.is_active.data and u.id == current_user.id:
            flash("You cannot deactivate your own account here.", "warning")
            return render_template("admin/user_edit.html", form=form, user=u)
        u.full_name = form.full_name.data.strip()
        u.email = email
        u.username = username
        if u.id != current_user.id:
            u.is_admin = bool(form.is_admin.data)
            u.is_active = bool(form.is_active.data)
        else:
            u.is_admin = True
            u.is_active = True
        if form.new_password.data:
            u.set_password(form.new_password.data)
        db.session.commit()
        flash("User updated.", "success")
        return redirect(url_for("admin.user_detail", uid=u.id))
    return render_template("admin/user_edit.html", form=form, user=u)


@bp.route("/users/toggle/<int:uid>", methods=["POST"])
def user_toggle(uid):
    u = User.query.get_or_404(uid)
    if u.id == current_user.id:
        flash("Cannot deactivate yourself.", "warning")
        return redirect(url_for("admin.users"))
    u.is_active = not u.is_active
    db.session.commit()
    flash("User status updated.", "success")
    return redirect(url_for("admin.users"))


@bp.route("/users/delete/<int:uid>", methods=["POST"])
def user_delete(uid):
    u = User.query.get_or_404(uid)
    if u.id == current_user.id:
        flash("Cannot delete yourself.", "warning")
        return redirect(url_for("admin.users"))
    if u.is_admin and _count_admins() <= 1:
        flash("Cannot delete the only administrator account.", "warning")
        return redirect(url_for("admin.users"))
    for r in RecognitionHistory.query.filter_by(user_id=u.id).all():
        _delete_history_image_file(current_app, r)
    RecognitionHistory.query.filter_by(user_id=u.id).delete()
    PredictionLog.query.filter_by(user_id=u.id).delete()
    db.session.delete(u)
    db.session.commit()
    flash("User deleted.", "info")
    return redirect(url_for("admin.users"))


@bp.route("/dataset")
def dataset():
    counts = dataset_class_counts(current_app.config["DATASET_ROOT"])
    rename_form = AdminDatasetRenameForm()
    return render_template(
        "admin/dataset.html",
        counts=counts,
        rename_form=rename_form,
    )


@bp.route("/dataset/upload", methods=["POST"])
def dataset_upload():
    cls = request.form.get("class_name", "").strip()
    if not cls:
        flash("Invalid class name.", "danger")
        return redirect(url_for("admin.dataset"))
    folder = os.path.join(current_app.config["DATASET_ROOT"], cls)
    os.makedirs(folder, exist_ok=True)
    files = request.files.getlist("images")
    n = 0
    for f in files:
        if not f or not f.filename:
            continue
        fn = secure_filename(f.filename)
        if not fn.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp")):
            continue
        path = os.path.join(folder, fn)
        f.save(path)
        n += 1
    flash(f"Uploaded {n} image(s) to class '{cls}'.", "success")
    return redirect(url_for("admin.dataset"))


@bp.route("/dataset/add-class", methods=["POST"])
def dataset_add_class():
    name = request.form.get("class_name", "").strip()
    name = secure_filename(name) or name.replace(" ", "_")
    if not name:
        flash("Class name required.", "danger")
        return redirect(url_for("admin.dataset"))
    path = os.path.join(current_app.config["DATASET_ROOT"], name)
    os.makedirs(path, exist_ok=True)
    flash(f"Class '{name}' created.", "success")
    return redirect(url_for("admin.dataset"))


@bp.route("/dataset/class/rename", methods=["POST"])
def dataset_class_rename():
    form = AdminDatasetRenameForm()
    if not form.validate_on_submit():
        flash("Invalid rename data.", "danger")
        return redirect(url_for("admin.dataset"))
    old = secure_filename(form.old_name.data.strip())
    new_raw = form.new_name.data.strip()
    new = secure_filename(new_raw) or new_raw.replace(" ", "_")
    if not old or not new:
        flash("Both names required.", "danger")
        return redirect(url_for("admin.dataset"))
    root = current_app.config["DATASET_ROOT"]
    op = os.path.join(root, old)
    np = os.path.join(root, new)
    if not os.path.isdir(op):
        flash("Source class folder not found.", "danger")
    elif os.path.exists(np):
        flash("A folder with the new name already exists.", "danger")
    else:
        os.rename(op, np)
        flash(f"Renamed class folder '{old}' → '{new}'.", "success")
    return redirect(url_for("admin.dataset"))


@bp.route("/dataset/class/delete", methods=["POST"])
def dataset_class_delete():
    name = secure_filename(request.form.get("class_name", "").strip())
    if not name:
        flash("Class name required.", "danger")
        return redirect(url_for("admin.dataset"))
    root = current_app.config["DATASET_ROOT"]
    path = os.path.join(root, name)
    if os.path.isdir(path):
        shutil.rmtree(path)
        flash(f"Deleted class '{name}' and all images inside it.", "info")
    else:
        flash("Class folder not found.", "warning")
    return redirect(url_for("admin.dataset"))


@bp.route("/dataset/image/delete")
def dataset_image_delete():
    cls = request.args.get("cls")
    fn = request.args.get("fn")
    if not cls or not fn:
        flash("Invalid parameters.", "danger")
        return redirect(url_for("admin.dataset"))
    cls = secure_filename(cls)
    fn = secure_filename(fn)
    path = os.path.join(current_app.config["DATASET_ROOT"], cls, fn)
    if os.path.isfile(path):
        os.remove(path)
        flash("Image removed.", "info")
    return redirect(url_for("admin.dataset_browse", class_name=cls))


@bp.route("/dataset/<class_name>")
def dataset_browse(class_name):
    class_name = secure_filename(class_name)
    folder = os.path.join(current_app.config["DATASET_ROOT"], class_name)
    if not os.path.isdir(folder):
        flash("Class not found.", "danger")
        return redirect(url_for("admin.dataset"))
    files = [
        f
        for f in os.listdir(folder)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp"))
    ]
    return render_template(
        "admin/dataset_browse.html", class_name=class_name, files=files
    )


@bp.route("/dataset/file/<class_name>/<path:filename>")
def dataset_file(class_name, filename):
    class_name = secure_filename(class_name)
    folder = os.path.join(current_app.config["DATASET_ROOT"], class_name)
    return send_from_directory(folder, filename)


@bp.route("/train", methods=["GET", "POST"])
def train():
    form = TrainForm()
    if request.method == "POST" and request.form.get("action") == "train":
        epochs = 20
        raw = (request.form.get("epochs") or "").strip()
        if raw:
            try:
                epochs = max(5, int(raw))
            except ValueError:
                pass
        ok, msg, meta = train_and_save(
            current_app.config["DATASET_ROOT"],
            current_app.config["MODEL_PATH"],
            current_app.config["MODEL_META_PATH"],
            epochs=epochs,
            batch_size=64,
        )
        if ok:
            flash(msg, "success")
        else:
            flash(msg, "danger")
        return redirect(url_for("admin.train"))
    meta = {}
    if os.path.isfile(current_app.config["MODEL_META_PATH"]):
        with open(current_app.config["MODEL_META_PATH"], "r", encoding="utf-8") as f:
            meta = json.load(f)
    counts = dataset_class_counts(current_app.config["DATASET_ROOT"])
    pre_path = current_app.config["PRETRAINED_MODEL_PATH"]
    pre_meta_path = current_app.config["PRETRAINED_META_PATH"]
    pretrained_loaded = os.path.isfile(pre_path)
    pre_meta = {}
    if os.path.isfile(pre_meta_path):
        with open(pre_meta_path, "r", encoding="utf-8") as f:
            pre_meta = json.load(f)
    return render_template(
        "admin/train.html",
        form=form,
        meta=meta,
        counts=counts,
        pretrained_loaded=pretrained_loaded,
        pre_meta=pre_meta,
    )


@bp.route("/model/pretrained", methods=["POST"])
def pretrained_upload():
    if request.form.get("action") == "remove":
        for key in ("PRETRAINED_MODEL_PATH", "PRETRAINED_META_PATH"):
            p = current_app.config.get(key)
            if p and os.path.isfile(p):
                try:
                    os.remove(p)
                except OSError:
                    pass
        flash(
            "Pretrained model removed. Predictions will use the locally trained sign_model.keras if present.",
            "info",
        )
        return redirect(url_for("admin.train"))

    f = request.files.get("model_file")
    if not f or not f.filename:
        flash("Choose a model file (.h5 or .keras).", "danger")
        return redirect(url_for("admin.train"))
    raw_name = secure_filename(f.filename)
    if not raw_name.lower().endswith((".h5", ".keras")):
        flash("Only .h5 or .keras files are accepted.", "danger")
        return redirect(url_for("admin.train"))

    dest = current_app.config["PRETRAINED_MODEL_PATH"]
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    f.save(dest)

    model, _ = load_model_bundle(dest, None)
    meta_path = current_app.config["PRETRAINED_META_PATH"]
    labels_to_save = None
    parsed = _parse_class_names_field(request.form.get("class_names", ""))
    scale_choice = (request.form.get("input_pixel_range") or "auto").strip().lower()
    if scale_choice == "0_1":
        det_range = "0_1"
    elif scale_choice == "0_255":
        det_range = "0_255"
    else:
        det_range = None  # detect after load

    if model is None:
        flash(
            "File was saved, but TensorFlow could not load it. Replace it with a valid Keras .h5/.keras file.",
            "danger",
        )
        if os.path.isfile(meta_path):
            try:
                os.remove(meta_path)
            except OSError:
                pass
        return redirect(url_for("admin.train"))

    if det_range is None:
        det_range = infer_input_pixel_range(model)

    if parsed:
        try:
            n_out = int(model.output_shape[-1])
            if len(parsed) != n_out:
                flash(
                    f"Model has {n_out} outputs but {len(parsed)} labels were given. "
                    "Labels were not saved; use one label per line in softmax order, or leave blank for dataset folder names.",
                    "warning",
                )
            else:
                labels_to_save = parsed
        except (TypeError, ValueError, IndexError):
            flash("Could not read model output size; class labels were not saved.", "warning")

    meta_out = {"source": "pretrained_upload", "input_pixel_range": det_range}
    if labels_to_save:
        meta_out["class_names"] = labels_to_save
    with open(meta_path, "w", encoding="utf-8") as out:
        json.dump(meta_out, out, indent=2)

    hint = (
        "pixels 0–255 (matches internal Rescaling 1/255 — typical Kaggle ASL notebooks)"
        if det_range == "0_255"
        else "normalized 0–1 before conv layers (matches this app’s local training)"
    )
    flash(
        f"Pretrained model is active. Input scaling: {det_range} ({hint}). "
        "If results still look random, set class labels (29 lines A–Z + del, nothing, space) and/or try the other scaling on re-upload.",
        "success",
    )
    return redirect(url_for("admin.train"))


@bp.route("/history", methods=["GET"])
def history():
    form = AdminHistoryFilterForm(formdata=request.args)
    q = RecognitionHistory.query
    if request.args and form.validate():
        q = _apply_admin_history_filters(q, form)
    rows = q.order_by(desc(RecognitionHistory.created_at)).limit(500).all()
    return render_template("admin/history.html", rows=rows, form=form)


@bp.route("/history/<int:rid>")
def history_record(rid):
    r = RecognitionHistory.query.get_or_404(rid)
    note_form = AdminRecognitionNoteForm(obj=r)
    has_upload_image = bool(r.source_image_relpath and r.source == "upload")
    return render_template(
        "admin/history_detail.html",
        record=r,
        note_form=note_form,
        has_upload_image=has_upload_image,
    )


@bp.route("/history/<int:rid>/note", methods=["POST"])
def history_record_note(rid):
    r = RecognitionHistory.query.get_or_404(rid)
    form = AdminRecognitionNoteForm()
    if form.validate_on_submit():
        r.user_note = (form.user_note.data or "").strip() or None
        db.session.commit()
        flash("Note saved.", "success")
    else:
        flash("Could not save note.", "warning")
    return redirect(url_for("admin.history_record", rid=rid))


@bp.route("/history/<int:rid>/delete", methods=["POST"])
def history_record_delete(rid):
    r = RecognitionHistory.query.get_or_404(rid)
    _delete_history_image_file(current_app, r)
    db.session.delete(r)
    db.session.commit()
    flash("Recognition record deleted.", "info")
    return redirect(url_for("admin.history"))


@bp.route("/history/<int:rid>/image")
def history_record_image(rid):
    r = RecognitionHistory.query.get_or_404(rid)
    if not r.source_image_relpath or r.source != "upload":
        abort(404)
    filename = os.path.basename(r.source_image_relpath)
    if filename != f"{rid}{os.path.splitext(filename)[1]}":
        abort(404)
    directory = os.path.join(current_app.config["HISTORY_UPLOAD_DIR"], str(r.user_id))
    if not os.path.isfile(os.path.join(directory, filename)):
        abort(404)
    return send_from_directory(directory, filename)


@bp.route("/history/bulk-delete", methods=["POST"])
def history_bulk_delete():
    raw_ids = request.form.getlist("ids")
    try:
        ids = [int(x) for x in raw_ids if str(x).strip().isdigit()]
    except ValueError:
        ids = []
    if not ids:
        flash("No records selected.", "warning")
        return redirect(url_for("admin.history"))
    rows = RecognitionHistory.query.filter(RecognitionHistory.id.in_(ids)).all()
    for r in rows:
        _delete_history_image_file(current_app, r)
    RecognitionHistory.query.filter(RecognitionHistory.id.in_(ids)).delete(synchronize_session=False)
    db.session.commit()
    flash(f"Deleted {len(ids)} recognition record(s).", "info")
    return redirect(url_for("admin.history"))


@bp.route("/history/export.csv")
def history_export_csv():
    rows = RecognitionHistory.query.order_by(desc(RecognitionHistory.created_at)).all()
    data = export_history_csv(rows)
    return Response(
        data,
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment;filename=admin_recognition_history.csv"},
    )


@bp.route("/history/export.pdf")
def history_export_pdf():
    rows = RecognitionHistory.query.order_by(desc(RecognitionHistory.created_at)).limit(300).all()
    lines = [f"Total records: {len(rows)}", ""]
    for r in rows:
        lines.append(
            f"{r.created_at} | user:{r.user_id} | {r.predicted_sign} | {r.confidence:.2%}"
        )
    pdf = build_pdf_report("Admin — Recognition History", lines)
    if pdf is None:
        flash("PDF export unavailable.", "warning")
        return redirect(url_for("admin.history"))
    return Response(
        pdf,
        mimetype="application/pdf",
        headers={"Content-Disposition": "attachment;filename=admin_history.pdf"},
    )


@bp.route("/logs", methods=["GET"])
def logs():
    form = AdminLogFilterForm(formdata=request.args)
    q = PredictionLog.query
    if request.args and form.validate():
        uid_raw = (form.user_id.data or "").strip()
        if uid_raw.isdigit():
            q = q.filter(PredictionLog.user_id == int(uid_raw))
    rows = q.order_by(desc(PredictionLog.created_at)).limit(500).all()
    return render_template("admin/logs.html", rows=rows, form=form)


@bp.route("/logs/delete/<int:lid>", methods=["POST"])
def log_delete(lid):
    log = PredictionLog.query.get_or_404(lid)
    db.session.delete(log)
    db.session.commit()
    flash("Debug log row deleted.", "info")
    uid = request.form.get("user_id") or request.args.get("user_id")
    if uid:
        return redirect(url_for("admin.logs", user_id=uid))
    return redirect(url_for("admin.logs"))


@bp.route("/logs/clear", methods=["POST"])
def logs_clear():
    n = PredictionLog.query.count()
    PredictionLog.query.delete(synchronize_session=False)
    db.session.commit()
    flash(f"Cleared {n} prediction debug log row(s).", "info")
    return redirect(url_for("admin.logs"))
