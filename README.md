# SignVision AI — Sign Language Recognition (Python + Flask)

A full-stack web application for **sign language gesture recognition** using a **local SQLite database**, **Flask** authentication, an **admin console**, and a **Keras CNN** trained on images stored under `data/dataset/`. All processing runs on your machine (no external recognition API).

---

## Features

### Authentication & users
- **User signup** — full name, email, username, password, confirm password; checks for duplicate email/username; passwords stored hashed (Werkzeug).
- **Login / logout** — sign in with **username or email**; session via Flask-Login; redirect to dashboard on success; error messages on failure.
- **Forgot / reset password** — verify **email + username**, then set a new password (local flow, no mail server).
- **Profile** — view account info; edit name, username, password (current password required to change password); **last login** shown on dashboard/profile context.

### User application
- **Dashboard** — total recognition attempts, last prediction, average confidence, shortcuts (recognition, history, accuracy, help).
- **Live webcam recognition** — start/stop camera; **MediaPipe** hand region detection; preprocessing (resize 64×64, grayscale, blur, normalize); **real-time predictions** with **confidence**; status messages (hand detected / no hand / prediction in progress).
- **Manual image upload** — predict from a file; same preprocessing pipeline.
- **Text output & sentence building** — append predicted sign, space, delete last character, clear text, clear prediction display.
- **Save frame** — save current JPEG to `saved_captures/`.
- **Supported classes** — list of dataset class names on the recognition page.
- **Recognition history** — stored per user (sign, confidence, date/time, source); **search** by date range and sign; **delete** one row or **clear all** (with confirmation); **export CSV / PDF**.
- **Accuracy / evaluation** — test accuracy, per-class stats, classification report, confusion matrix (from `trained_model/model_meta.json`).
- **Help page** — how to log in, train, use camera, and upload images.

### Admin
- **Admin dashboard** (sidebar layout) — totals: users, active users, predictions, today’s usage; top signs; recent users.
- **User management** — list users; **activate/deactivate**; **delete** user (and related history); cannot delete/deactivate self destructively where blocked in code.
- **Dataset management** — create class folders; **upload** multiple images per class; **browse** class; **delete** images; **sample counts** per class.
- **Train / retrain model** — trains on `data/dataset/`, saves `trained_model/sign_model.keras` and `model_meta.json`.
- **Global history & logs** — view all recognition rows; **export CSV/PDF**; **prediction debug logs** (frame time, confidence, etc.).
- **Statistics** — aggregates suitable for demos (totals, top signs, daily count).

### AI / ML
- **Hand region** — MediaPipe Hands; crop and preprocess for the model.
- **Model** — convolutional neural network (Keras); **softmax** over class folders.
- **Training**
  - **Small datasets** (≤ 2000 images total): load in memory with OpenCV (with Gaussian blur).
  - **Large datasets** (&gt; 2000 images): **TensorFlow `image_dataset_from_directory`** streaming (fits **tens of thousands** of images without loading everything into RAM).
- **Saved model** — load at prediction time; clear message if missing.

### Security & quality
- **CSRF** protection (Flask-WTF); **login required** for user/admin areas; **admin-only** routes.
- **Validation** on forms; **error handling** for camera, missing model, bad upload, empty fields.
- **Responsive** Bootstrap-based UI with custom styling (landing + admin sidebar).

---

## Requirements

- **Python 3.10+** (3.12 tested)
- Dependencies in `requirements.txt` (Flask, SQLAlchemy, TensorFlow CPU, MediaPipe, OpenCV, scikit-learn, ReportLab, etc.)

---

## Installation

```bash
cd "path/to/Sign Language Recognition using Python with AI"
pip install -r requirements.txt
```

---

## Database seed (optional)

Creates **8 demo users** (including **admin**), sample history rows, and small synthetic images for quick tests:

```bash
python seed_db.py --force
```

**Default accounts after seed**

| Role  | Username | Password   |
|-------|----------|------------|
| Admin | `admin`  | `Admin@123` |
| User  | `alice` … | `User@123` |

Skip `--force` if you already have users and do not want to reset the database.

---

## How to run the application

```bash
python run.py
```

Open **http://127.0.0.1:5000** in your browser.

- **Users**: register or sign in with a seeded account.
- **Admins**: sign in as `admin` → use **Admin** in the navbar for dataset, training, and reports.

---

## Dataset layout

Put training images in **one folder per class** (folder name = label), for example:

```text
data/dataset/
  A/
  B/
  ...
  Hello/
  Thanks/
  Yes/
  No/
```

Supported file extensions: **`.png` `.jpg` `.jpeg` `.webp` `.bmp`**

The model learns **folder names** as class names (sorted alphabetically by Keras).

---

## Train the model

Training writes:

- `trained_model/sign_model.keras` — trained Keras model  
- `trained_model/model_meta.json` — validation accuracy, class names, sample counts, classification report, confusion matrix  

### Option A — Command line (recommended for long runs)

```bash
python train_model.py
```

With custom settings:

```bash
python train_model.py --epochs 25 --batch-size 64
```

- **`--epochs`** — number of epochs (the trainer enforces a **minimum of 5** epochs internally).
- **`--batch-size`** — used for **large** datasets (streaming mode). Lower if you run out of GPU/CPU memory; **64** is a reasonable default.

**Note:** With **many images** (e.g. tens of thousands), training can take **a long time on CPU**. Use fewer epochs first to verify, then increase for better accuracy.

### Option B — Admin UI

1. Sign in as **admin**.  
2. Go to **Admin → Train model**.  
3. Set **epochs** (optional) and click **Train & save model**.

---

## After training

1. Sign in as any user.  
2. Open **Recognition** → **Start camera** (allow browser access) or use **upload image**.  
3. Check **History** and **Accuracy** for logged results and metrics.

---

## Configuration

Key paths in `config.py`:

| Setting | Purpose |
|--------|---------|
| `DATASET_ROOT` | `data/dataset/` |
| `MODEL_PATH` | `trained_model/sign_model.keras` |
| `MODEL_META_PATH` | `trained_model/model_meta.json` |
| SQLite DB | `instance/sign_language.db` |

Set `SECRET_KEY` in environment for production.

### Inference vs training (important)

Training uses TensorFlow’s pipeline: **RGB-based grayscale**, **bilinear-style resize**, **no blur**, and **full image** (or center-cropped square for uploads/webcam). Inference uses the **same rules** so predictions match validation accuracy.

- **Default (recommended for most ASL / fingerspelling folders):** webcam and upload use a **center square crop** of the frame, then the same encoding as training. **Hand detection (MediaPipe) is off** so results match your dataset images.
- **Optional hand crop:** if you trained only on tight hand regions from a camera, start the app with:
  - **Windows (PowerShell):** `$env:SIGNVISION_USE_HAND_CROP="1"; python run.py`
  - **Linux/macOS:** `SIGNVISION_USE_HAND_CROP=1 python run.py`

After changing this variable, **restart the server**.

---

## Project structure (summary)

| Path | Description |
|------|-------------|
| `run.py` | Start Flask server |
| `train_model.py` | CLI training script |
| `seed_db.py` | Seed users + demo data |
| `app/` | Application package (blueprints, models, ML) |
| `static/` | CSS, JS |
| `templates/` | HTML templates |
| `data/dataset/` | Class folders for training images |
| `trained_model/` | Saved model + metadata |
| `uploads/` | Temporary uploads |
| `saved_captures/` | Saved webcam frames |

---

## Troubleshooting

- **No trained model** — Train via `train_model.py` or Admin → Train model.  
- **Wrong predictions despite high validation accuracy** — Inference now matches TensorFlow training (grayscale, resize, no extra blur). **Center your sign** in the camera; the app uses a **center square crop** like typical dataset images. For MediaPipe hand mode only, set `SIGNVISION_USE_HAND_CROP=1` and restart.  
- **Camera not working** — Browser permissions, HTTPS on some hosts, or no webcam.  
- **“No hand detected”** — Only if `SIGNVISION_USE_HAND_CROP=1`. Otherwise framing/lighting for a centered sign.  
- **Training looks “stuck” after “Using … files for validation”** — Training was running with no visible progress. The script now prints `Epoch 1/20 | …` after each epoch. On CPU, **one epoch can take 10–30+ minutes** with ~55k images; wait for those lines or reduce data/epochs for a quicker test.
- **Training slow** — Normal on CPU with large datasets; reduce `--epochs` or `--batch-size` if needed.  
- **Out of memory** — Use streaming mode (automatic over 2000 images); reduce `--batch-size`.

---

## License

Use this project for academic/educational purposes as appropriate for your institution.
