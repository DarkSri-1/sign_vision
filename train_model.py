"""
Train the sign classifier from images in data/dataset/<class_name>/.

Large datasets use a TensorFlow streaming pipeline (no full RAM load).

Usage:
  python train_model.py
  python train_model.py --epochs 25 --batch-size 64
"""
import argparse
import os
import sys

ROOT = os.path.abspath(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def main():
    parser = argparse.ArgumentParser(description="Train SignVision CNN on local dataset")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs (minimum 5 enforced in trainer)")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for streaming training (large datasets)",
    )
    args = parser.parse_args()

    from app import create_app
    from app.ml.trainer import train_and_save

    app = create_app()
    with app.app_context():
        ok, msg, meta = train_and_save(
            app.config["DATASET_ROOT"],
            app.config["MODEL_PATH"],
            app.config["MODEL_META_PATH"],
            epochs=args.epochs,
            batch_size=args.batch_size,
        )
        print(msg)
        if ok and meta:
            print("Validation accuracy:", round(meta.get("test_accuracy", 0) * 100, 2), "%")
            print("Classes:", len(meta.get("class_names", [])))
            print("Model:", app.config["MODEL_PATH"])
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
