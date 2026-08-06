import os
import json
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
except ImportError:
    tf = None
    keras = None
    layers = None

from app.ml.preprocessor import IMG_SIZE, encode_for_model, center_square_crop

# Use TensorFlow pipeline above this many images (avoids loading all images into RAM)
STREAMING_THRESHOLD = 2000


def _list_classes(dataset_root):
    if not os.path.isdir(dataset_root):
        return []
    return sorted(
        d
        for d in os.listdir(dataset_root)
        if os.path.isdir(os.path.join(dataset_root, d)) and not d.startswith(".")
    )


def _count_images(dataset_root):
    n = 0
    for cls in _list_classes(dataset_root):
        folder = os.path.join(dataset_root, cls)
        try:
            for fn in os.listdir(folder):
                if fn.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp")):
                    n += 1
        except OSError:
            pass
    return n


def _samples_per_class(dataset_root, class_names):
    counts = {}
    for cls in class_names:
        folder = os.path.join(dataset_root, cls)
        if not os.path.isdir(folder):
            counts[cls] = 0
            continue
        counts[cls] = sum(
            1
            for fn in os.listdir(folder)
            if fn.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp"))
        )
    return counts


def _load_dataset_in_memory(dataset_root):
    import cv2

    X, y, class_names = [], [], []
    classes = _list_classes(dataset_root)
    if not classes:
        return None, None, []
    name_to_idx = {n: i for i, n in enumerate(classes)}
    for cls in classes:
        folder = os.path.join(dataset_root, cls)
        for fn in os.listdir(folder):
            if fn.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp")):
                path = os.path.join(folder, fn)
                img = cv2.imread(path)
                if img is None:
                    continue
                sq = center_square_crop(img)
                feat = encode_for_model(sq)[0]
                X.append(feat)
                y.append(name_to_idx[cls])
    if not X:
        return None, None, classes
    X = np.array(X)
    y = np.array(y)
    return X, y, classes


def _epoch_console_logger(total_epochs):
    """ASCII-only progress so Windows consoles do not hang on unicode progress bars."""

    def on_epoch_end(epoch, logs):
        logs = logs or {}
        va = logs.get("val_accuracy")
        if va is None:
            va = logs.get("val_acc", 0.0)
        print(
            f"  Epoch {epoch + 1}/{total_epochs} | "
            f"loss={float(logs.get('loss', 0)):.4f} acc={float(logs.get('accuracy', 0)):.4f} | "
            f"val_loss={float(logs.get('val_loss', 0)):.4f} val_acc={float(va):.4f}",
            flush=True,
        )

    return on_epoch_end


def build_model(num_classes):
    model = keras.Sequential(
        [
            layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1)),
            layers.Conv2D(32, 3, activation="relu", padding="same"),
            layers.MaxPooling2D(2),
            layers.Conv2D(64, 3, activation="relu", padding="same"),
            layers.MaxPooling2D(2),
            layers.Conv2D(128, 3, activation="relu", padding="same"),
            layers.MaxPooling2D(2),
            layers.Dropout(0.25),
            layers.Flatten(),
            layers.Dense(256, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def _train_streaming(dataset_root, model_path, meta_path, epochs, batch_size, progress_cb=None):
    counts_map = _samples_per_class(dataset_root, _list_classes(dataset_root))
    total = sum(counts_map.values())
    if total < 2:
        return False, "Need at least 2 images across classes to train.", None

    try:
        train_ds = tf.keras.utils.image_dataset_from_directory(
            dataset_root,
            validation_split=0.2,
            subset="training",
            seed=42,
            image_size=(IMG_SIZE, IMG_SIZE),
            batch_size=batch_size,
            color_mode="grayscale",
        )
        val_ds = tf.keras.utils.image_dataset_from_directory(
            dataset_root,
            validation_split=0.2,
            subset="validation",
            seed=42,
            image_size=(IMG_SIZE, IMG_SIZE),
            batch_size=batch_size,
            color_mode="grayscale",
        )
    except ValueError as e:
        return False, f"Could not load dataset: {e}", None

    class_names = list(train_ds.class_names)
    num_classes = len(class_names)
    if num_classes < 1:
        return False, "No classes.", None

    norm = layers.Rescaling(1.0 / 255.0)
    train_ds = train_ds.map(lambda x, y: (norm(x), y), num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.map(lambda x, y: (norm(x), y), num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

    model = build_model(num_classes)
    ep = max(5, int(epochs))
    train_n = max(1, int(round(total * 0.8)))
    steps_per_epoch = max(1, (train_n + batch_size - 1) // batch_size)
    print(
        f"\nTraining started: ~{train_n} train images, ~{steps_per_epoch} steps/epoch, "
        f"batch_size={batch_size}, epochs={ep}.",
        flush=True,
    )
    print(
        "On CPU with a large dataset each epoch can take many minutes. "
        "This is normal — watch for 'Epoch N/...' lines below.\n",
        flush=True,
    )

    callbacks = [
        keras.callbacks.LambdaCallback(on_epoch_end=_epoch_console_logger(ep)),
    ]
    if progress_cb:
        callbacks.append(
            keras.callbacks.LambdaCallback(
                on_epoch_end=lambda e, logs, cb=progress_cb, ep_tot=ep: cb(
                    e + 1, ep_tot, (logs or {}).get("accuracy", 0)
                )
            )
        )

    model.fit(train_ds, validation_data=val_ds, epochs=ep, verbose=0, callbacks=callbacks)

    print("\nComputing metrics on full validation set (please wait)...", flush=True)
    y_true = []
    y_pred = []
    for batch_x, batch_y in val_ds:
        preds = model.predict(batch_x, verbose=0)
        y_pred.extend(np.argmax(preds, axis=1).tolist())
        y_true.extend(batch_y.numpy().tolist())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    labels_all = list(range(num_classes))
    acc = float(accuracy_score(y_true, y_pred))
    report = classification_report(
        y_true,
        y_pred,
        labels=labels_all,
        target_names=class_names,
        zero_division=0,
        output_dict=True,
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels_all).tolist()

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(model_path)

    meta = {
        "class_names": class_names,
        "samples_per_class": counts_map,
        "total_images": total,
        "training_mode": "streaming",
        "test_accuracy": acc,
        "classification_report": report,
        "confusion_matrix": cm,
        "epochs_trained": ep,
        "batch_size": batch_size,
        "img_size": IMG_SIZE,
        "input_pixel_range": "0_1",
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return True, "Training completed.", meta


def _train_in_memory(dataset_root, model_path, meta_path, epochs, progress_cb=None):
    from sklearn.model_selection import train_test_split

    X, y, class_names = _load_dataset_in_memory(dataset_root)
    if X is None or len(class_names) == 0:
        return False, "No dataset classes found. Add folders under data/dataset/.", None
    if len(X) < 2:
        return False, "Need at least 2 images across classes to train.", None

    num_classes = len(class_names)
    stratify = y if len(np.unique(y)) > 1 and min(np.bincount(y)) >= 2 else None
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=stratify
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    counts = {c: 0 for c in class_names}
    for yi in y:
        counts[class_names[int(yi)]] += 1

    model = build_model(num_classes)
    ep = max(5, int(epochs))
    bs = min(32, len(X_train))
    print(
        f"\nTraining started (in-memory): {len(X_train)} train samples, batch_size={bs}, epochs={ep}.\n",
        flush=True,
    )
    callbacks = [
        keras.callbacks.LambdaCallback(on_epoch_end=_epoch_console_logger(ep)),
    ]
    if progress_cb:
        callbacks.append(
            keras.callbacks.LambdaCallback(
                on_epoch_end=lambda e, l, cb=progress_cb, ep_tot=ep: cb(
                    e + 1, ep_tot, (l or {}).get("accuracy", 0)
                )
            )
        )

    model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=ep,
        batch_size=bs,
        verbose=0,
        callbacks=callbacks,
    )

    print("\nComputing final predictions for metrics...", flush=True)
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    acc = float(accuracy_score(y_test, y_pred))
    report = classification_report(
        y_test, y_pred, target_names=class_names, zero_division=0, output_dict=True
    )
    cm = confusion_matrix(y_test, y_pred).tolist()

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(model_path)

    meta = {
        "class_names": class_names,
        "samples_per_class": counts,
        "total_images": int(len(X)),
        "training_mode": "in_memory",
        "test_accuracy": acc,
        "classification_report": report,
        "confusion_matrix": cm,
        "epochs_trained": ep,
        "img_size": IMG_SIZE,
        "input_pixel_range": "0_1",
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return True, "Training completed.", meta


def train_and_save(
    dataset_root, model_path, meta_path, epochs=25, progress_cb=None, batch_size=64
):
    if keras is None:
        return False, "TensorFlow/Keras is not available.", None

    total = _count_images(dataset_root)
    if total == 0:
        return False, "No images found under data/dataset/.", None

    if total > STREAMING_THRESHOLD:
        return _train_streaming(
            dataset_root, model_path, meta_path, epochs, batch_size, progress_cb
        )
    return _train_in_memory(dataset_root, model_path, meta_path, epochs, progress_cb)
