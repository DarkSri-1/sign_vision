import os
import csv
import io
from datetime import datetime
def dataset_class_counts(dataset_root):
    out = {}
    if not os.path.isdir(dataset_root):
        return out
    for name in sorted(os.listdir(dataset_root)):
        path = os.path.join(dataset_root, name)
        if not os.path.isdir(path) or name.startswith("."):
            continue
        n = sum(
            1
            for f in os.listdir(path)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp"))
        )
        out[name] = n
    return out


def model_exists(app):
    cfg = app.config
    if os.path.isfile(cfg.get("PRETRAINED_MODEL_PATH", "")):
        return True
    return os.path.isfile(cfg["MODEL_PATH"])


def log_prediction_db(user_id, sign, confidence, frame_ms=None, extra=None, source="webcam"):
    from app import db
    from app.models import RecognitionHistory, PredictionLog

    rh = RecognitionHistory(
        user_id=user_id,
        predicted_sign=sign,
        confidence=confidence,
        source=source,
    )
    db.session.add(rh)
    pl = PredictionLog(
        user_id=user_id,
        predicted_class=sign,
        confidence=confidence,
        frame_time_ms=frame_ms,
        extra=extra or "",
    )
    db.session.add(pl)
    db.session.commit()
    return rh.id


def export_history_csv(rows):
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(
        [
            "id",
            "user_id",
            "predicted_sign",
            "confidence",
            "date_time",
            "source",
            "user_note",
            "source_image_relpath",
        ]
    )
    for r in rows:
        w.writerow(
            [
                r.id,
                r.user_id,
                r.predicted_sign,
                r.confidence,
                r.created_at.isoformat() if r.created_at else "",
                getattr(r, "source", ""),
                getattr(r, "user_note", "") or "",
                getattr(r, "source_image_relpath", "") or "",
            ]
        )
    return buf.getvalue()


def build_pdf_report(title, lines):
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas
    except ImportError:
        return None

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    w, h = letter
    y = h - 50
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y, title)
    y -= 30
    c.setFont("Helvetica", 10)
    for line in lines:
        if y < 50:
            c.showPage()
            y = h - 50
        c.drawString(50, y, line[:120])
        y -= 14
    c.save()
    buf.seek(0)
    return buf.read()
