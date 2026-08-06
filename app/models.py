from datetime import datetime
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from app import db


class User(UserMixin, db.Model):
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    full_name = db.Column(db.String(120), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    username = db.Column(db.String(64), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(256), nullable=False)
    is_admin = db.Column(db.Boolean, default=False)
    is_active = db.Column(db.Boolean, default=True)
    last_login = db.Column(db.DateTime)
    last_activity = db.Column(db.DateTime)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    recognitions = db.relationship("RecognitionHistory", backref="user", lazy="dynamic")
    prediction_logs = db.relationship("PredictionLog", backref="user", lazy="dynamic")

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)


class RecognitionHistory(db.Model):
    __tablename__ = "recognition_history"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, index=True)
    predicted_sign = db.Column(db.String(64), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    source = db.Column(db.String(32), default="webcam")
    user_note = db.Column(db.String(512), nullable=True)
    # Relative path under HISTORY_UPLOAD_DIR, e.g. "42/33.jpg" (user_id / record_id.ext)
    source_image_relpath = db.Column(db.String(512), nullable=True)


class PredictionLog(db.Model):
    __tablename__ = "prediction_logs"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=True, index=True)
    predicted_class = db.Column(db.String(64))
    confidence = db.Column(db.Float)
    frame_time_ms = db.Column(db.Float)
    extra = db.Column(db.String(256))
    created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)
