"""
Seed the database with 8 sample users and demo recognition history.
Creates synthetic training images under data/dataset/ for quick model training.

Usage:
  python seed_db.py           # skip if users already exist
  python seed_db.py --force   # drop all tables and re-seed
"""
import os
import sys

ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, ROOT)

import random
from datetime import datetime, timedelta

import numpy as np
import cv2

from app import create_app, db
from app.models import User, RecognitionHistory, PredictionLog


CLASSES = ["A", "B", "C", "Hello", "Thanks", "Yes", "No"]


def _write_dummy_images(dataset_root, classes, per_class=6):
    os.makedirs(dataset_root, exist_ok=True)
    rng = np.random.default_rng(42)
    for cls in classes:
        folder = os.path.join(dataset_root, cls)
        os.makedirs(folder, exist_ok=True)
        for i in range(per_class):
            path = os.path.join(folder, f"sample_{i:02d}.png")
            if os.path.isfile(path):
                continue
            img = (rng.random((200, 200, 3)) * 255).astype(np.uint8)
            noise = int(hash(cls) % 40)
            img[:, :, 0] = np.clip(img[:, :, 0].astype(int) + noise, 0, 255).astype(np.uint8)
            cv2.imwrite(path, img)


def seed_users():
    users_data = [
        ("System Admin", "admin@signvision.local", "admin", "Admin@123", True),
        ("Alice Kumar", "alice@example.com", "alice", "User@123", False),
        ("Bob Singh", "bob@example.com", "bob", "User@123", False),
        ("Carol Dsouza", "carol@example.com", "carol", "User@123", False),
        ("David Lee", "david@example.com", "david", "User@123", False),
        ("Elena Rao", "elena@example.com", "elena", "User@123", False),
        ("Farhan Ali", "farhan@example.com", "farhan", "User@123", False),
        ("Grace Paul", "grace@example.com", "grace", "User@123", False),
    ]
    created = []
    for full_name, email, username, password, is_admin in users_data:
        u = User(
            full_name=full_name,
            email=email.lower(),
            username=username,
            is_admin=is_admin,
            is_active=True,
            last_login=datetime.utcnow() - timedelta(hours=random.randint(1, 48)),
            last_activity=datetime.utcnow() - timedelta(hours=random.randint(0, 12)),
        )
        u.set_password(password)
        db.session.add(u)
        created.append(u)
    db.session.flush()
    return created


def seed_history(users):
    signs = CLASSES + ["A", "B", "Hello"]
    for _ in range(12):
        u = random.choice([x for x in users if not x.is_admin] or users)
        r = RecognitionHistory(
            user_id=u.id,
            predicted_sign=random.choice(signs),
            confidence=random.uniform(0.55, 0.99),
            created_at=datetime.utcnow() - timedelta(days=random.randint(0, 14), hours=random.randint(0, 23)),
            source=random.choice(["webcam", "upload"]),
        )
        db.session.add(r)
    for _ in range(5):
        pl = PredictionLog(
            user_id=random.choice(users).id,
            predicted_class=random.choice(signs),
            confidence=random.uniform(0.6, 0.98),
            frame_time_ms=random.uniform(15, 90),
            extra="seed",
        )
        db.session.add(pl)


def main():
    force = "--force" in sys.argv
    app = create_app()
    with app.app_context():
        if force:
            db.drop_all()
            db.create_all()
        elif User.query.first():
            print("Database already contains users. Use --force to reset and re-seed.")
            return

        dataset_root = app.config["DATASET_ROOT"]
        _write_dummy_images(dataset_root, CLASSES, per_class=6)
        print(f"Dummy dataset images under: {dataset_root}")

        users = seed_users()
        seed_history(users)
        db.session.commit()
        print("Seeded 8 users (admin / alice ... grace). Default password for non-admin: User@123")
        print("Admin login: username admin, password Admin@123")
        print("Train the model from Admin / Train model after seeding.")


if __name__ == "__main__":
    main()
