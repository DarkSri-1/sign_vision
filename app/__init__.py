import os
from flask import Flask
from flask_login import LoginManager
from flask_sqlalchemy import SQLAlchemy
from flask_wtf.csrf import CSRFProtect
from config import Config

db = SQLAlchemy()
login_manager = LoginManager()
login_manager.login_view = "auth.login"
login_manager.login_message_category = "warning"
csrf = CSRFProtect()


def _sqlite_upgrade_recognition_history_columns(app):
    uri = app.config.get("SQLALCHEMY_DATABASE_URI") or ""
    if not uri.startswith("sqlite:"):
        return
    try:
        from sqlalchemy import inspect, text

        inspector = inspect(db.engine)
        if not inspector.has_table("recognition_history"):
            return
        cols = {c["name"] for c in inspector.get_columns("recognition_history")}
        with db.engine.begin() as conn:
            if "user_note" not in cols:
                conn.execute(
                    text("ALTER TABLE recognition_history ADD COLUMN user_note VARCHAR(512)")
                )
        inspector = inspect(db.engine)
        cols = {c["name"] for c in inspector.get_columns("recognition_history")}
        with db.engine.begin() as conn:
            if "source_image_relpath" not in cols:
                conn.execute(
                    text(
                        "ALTER TABLE recognition_history ADD COLUMN source_image_relpath VARCHAR(512)"
                    )
                )
    except Exception:
        pass


def create_app(config_class=Config):
    app = Flask(__name__, static_folder="../static", template_folder="../templates")
    app.config.from_object(config_class)

    csrf.init_app(app)
    os.makedirs(os.path.join(os.path.dirname(__file__), "..", "instance"), exist_ok=True)
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    os.makedirs(app.config["DATASET_ROOT"], exist_ok=True)
    os.makedirs(app.config["MODEL_DIR"], exist_ok=True)
    os.makedirs(app.config["CAPTURE_SAVE_DIR"], exist_ok=True)
    os.makedirs(app.config["HISTORY_UPLOAD_DIR"], exist_ok=True)

    db.init_app(app)
    login_manager.init_app(app)

    from app.models import User

    @login_manager.user_loader
    def load_user(user_id):
        return db.session.get(User, int(user_id))

    from app.blueprints.auth import bp as auth_bp
    from app.blueprints.main import bp as main_bp
    from app.blueprints.admin import bp as admin_bp
    from app.blueprints.api import bp as api_bp

    app.register_blueprint(auth_bp)
    app.register_blueprint(main_bp)
    app.register_blueprint(admin_bp, url_prefix="/admin")
    app.register_blueprint(api_bp, url_prefix="/api")

    with app.app_context():
        db.create_all()
        _sqlite_upgrade_recognition_history_columns(app)

    @app.context_processor
    def inject_csrf():
        from flask_wtf.csrf import generate_csrf
        from datetime import datetime

        return dict(csrf_token=generate_csrf, utcnow=datetime.utcnow)

    return app
