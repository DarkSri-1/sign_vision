import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))


class Config:
    """
    Inference must match training. Default dataset (ASL-style) uses full-frame images;
    set SIGNVISION_USE_HAND_CROP=1 only if you trained on MediaPipe hand crops.
    """

    SECRET_KEY = os.environ.get("SECRET_KEY") or "dev-change-me-in-production-sign-lang-2026"
    USE_HAND_CROP_FOR_INFERENCE = os.environ.get("SIGNVISION_USE_HAND_CROP", "").lower() in (
        "1",
        "true",
        "yes",
    )
    SQLALCHEMY_DATABASE_URI = os.environ.get("DATABASE_URL") or f"sqlite:///{os.path.join(BASE_DIR, 'instance', 'sign_language.db')}"
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
    DATASET_ROOT = os.path.join(BASE_DIR, "data", "dataset")
    MODEL_DIR = os.path.join(BASE_DIR, "trained_model")
    MODEL_PATH = os.path.join(MODEL_DIR, "sign_model.keras")
    MODEL_META_PATH = os.path.join(MODEL_DIR, "model_meta.json")
    # Admin-uploaded pretrained weights (preferred at inference when present)
    PRETRAINED_MODEL_PATH = os.path.join(MODEL_DIR, "pretrained_asl.h5")
    PRETRAINED_META_PATH = os.path.join(MODEL_DIR, "pretrained_model_meta.json")
    CAPTURE_SAVE_DIR = os.path.join(BASE_DIR, "saved_captures")
    # Persisted copies of user upload predictions (relative keys on RecognitionHistory)
    HISTORY_UPLOAD_DIR = os.path.join(BASE_DIR, "instance", "history_uploads")
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024
    WTF_CSRF_ENABLED = True
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = "Lax"
