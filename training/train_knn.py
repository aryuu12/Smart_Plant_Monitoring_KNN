from pathlib import Path
import sys
import warnings

import joblib
import numpy as np
import pandas as pd

from PIL import Image, UnidentifiedImageError

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    StratifiedKFold
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# Memastikan folder utama proyek dapat dikenali Python
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from training.features import extract_features, FEATURE_NAMES


# =====================================================
# 1. KONFIGURASI
# =====================================================

DATASET_DIR = BASE_DIR / "dataset"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "model_knn_kentang.joblib"

FEATURES_CSV = OUTPUT_DIR / "dataset_fitur.csv"
PREDICTIONS_CSV = OUTPUT_DIR / "prediksi_test_knn.csv"
METRICS_CSV = OUTPUT_DIR / "evaluasi_knn.csv"
REPORT_CSV = OUTPUT_DIR / "classification_report_knn.csv"
CONFUSION_CSV = OUTPUT_DIR / "confusion_matrix_knn.csv"
SPLIT_CSV = OUTPUT_DIR / "pembagian_dataset.csv"

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

RANDOM_STATE = 42


# =====================================================
# 2. MEMBACA DATASET DAN EKSTRAKSI FITUR
# =====================================================

def load_dataset():

    X = []
    y = []
    filenames = []

    if not DATASET_DIR.exists():
        raise FileNotFoundError(
            f"Folder dataset tidak ditemukan: {DATASET_DIR}"
        )

    class_folders = sorted([
        folder for folder in DATASET_DIR.iterdir()
        if folder.is_dir()
    ])

    if not class_folders:
        raise ValueError("Tidak ditemukan folder kelas dalam dataset.")

    print("=" * 60)
    print("EKSTRAKSI FITUR DATASET DAUN KENTANG")
    print("=" * 60)

    for class_folder in class_folders:

        class_name = class_folder.name

        image_files = sorted([
            file for file in class_folder.rglob("*")
            if file.is_file()
            and file.suffix.lower() in VALID_EXTENSIONS
        ])

        print(f"\nKelas: {class_name}")
        print(f"Jumlah gambar: {len(image_files)}")

        for image_path in image_files:

            try:
                with Image.open(image_path) as img:
                    image_rgb = np.asarray(img.convert("RGB"))

                features = extract_features(image_rgb)

                if len(features) != len(FEATURE_NAMES):
                    raise ValueError(
                        "Jumlah fitur tidak sesuai dengan FEATURE_NAMES."
                    )

                if not np.all(np.isfinite(features)):
                    raise ValueError("Ditemukan fitur NaN atau infinity.")

                X.append(features)
                y.append(class_name)
                filenames.append(
                    str(image_path.relative_to(BASE_DIR))
                )

            except (
                UnidentifiedImageError,
                OSError,
                ValueError,
                TypeError
            ) as error:

                warnings.warn(
                    f"Gambar dilewati: {image_path} | {error}"
                )

    if not X:
        raise ValueError("Tidak ada gambar valid yang berhasil diproses.")

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)

    feature_df = pd.DataFrame(X, columns=FEATURE_NAMES)
    feature_df.insert(0, "class_label", y)
    feature_df.insert(0, "filename", filenames)

    feature_df.to_csv(FEATURES_CSV, index=False)

    print("\nEkstraksi fitur selesai.")
    print(f"Jumlah gambar valid: {len(X)}")
    print(f"Jumlah fitur: {len(FEATURE_NAMES)}")
    print(f"File fitur: {FEATURES_CSV}")

    return X, y, filenames


# =====================================================
# 3. MEMBAGI DATASET
# =====================================================

def split_dataset(X, y, filenames):

    class_counts = pd.Series(y).value_counts()

    print("\nDistribusi kelas:")
    print(class_counts)

    # Minimal 10 gambar per kelas untuk pembagian
    # train, validation, dan test secara stratified
    if class_counts.min() < 10:
        raise ValueError(
            "Setiap kelas membutuhkan minimal 10 gambar "
            "untuk pembagian dataset 70:15:15. "
            "Tambahkan data atau sesuaikan strategi pembagian."
        )

    # 70% training, 30% sementara
    X_train, X_temp, y_train, y_temp, \
        file_train, file_temp = train_test_split(
            X,
            y,
            filenames,
            test_size=0.30,
            random_state=RANDOM_STATE,
            stratify=y
        )

    # Membagi sisa 30% menjadi 15% validation dan 15% test
    X_val, X_test, y_val, y_test, \
        file_val, file_test = train_test_split(
            X_temp,
            y_temp,
            file_temp,
            test_size=0.50,
            random_state=RANDOM_STATE,
            stratify=y_temp
        )

    print("\nPembagian dataset:")
    print(f"Training   : {len(X_train)} gambar")
    print(f"Validation : {len(X_val)} gambar")
    print(f"Testing    : {len(X_test)} gambar")

    split_df = pd.DataFrame({
        "filename": (
            list(file_train) +
            list(file_val) +
            list(file_test)
        ),
        "split": (
            ["train"] * len(file_train) +
            ["validation"] * len(file_val) +
            ["test"] * len(file_test)
        )
    })

    split_df.to_csv(SPLIT_CSV, index=False)

    return (
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        file_test
    )


# =====================================================
# 4. OPTIMASI PARAMETER KNN
# =====================================================

def optimize_knn(X_train, y_train):

    print("\n" + "=" * 60)
    print("OPTIMASI PARAMETER KNN")
    print("=" * 60)

    class_counts = pd.Series(y_train).value_counts()

    # Jumlah fold CV tidak boleh melebihi jumlah
    # sampel kelas terkecil pada data training.
    n_splits = min(5, int(class_counts.min()))

    if n_splits < 2:
        raise ValueError(
            "Jumlah sampel per kelas tidak mencukupi untuk cross-validation."
        )

    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    # Memastikan nilai K tidak melebihi jumlah sampel
    # training pada setiap fold.
    min_fold_train = len(y_train) - int(
        np.ceil(len(y_train) / n_splits)
    )

    candidate_k = [
        k for k in [3, 5, 7, 9, 11]
        if k <= min_fold_train
    ]

    if not candidate_k:
        raise ValueError(
            "Jumlah data training terlalu sedikit untuk optimasi KNN."
        )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier())
    ])

    param_grid = {
        "knn__n_neighbors": candidate_k,
        "knn__weights": ["uniform", "distance"],
        "knn__metric": ["euclidean", "manhattan"]
    }

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="f1_macro",
        cv=cv,
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train, y_train)

    print("\nParameter terbaik:")
    print(grid_search.best_params_)

    print("\nNilai F1-macro terbaik dari CV:")
    print(f"{grid_search.best_score_:.4f}")

    return grid_search.best_estimator_, grid_search.best_params_


# =====================================================
# 5. EVALUASI MODEL
# =====================================================

def evaluate_model(model, X, y, dataset_name):

    predictions = model.predict(X)

    metrics = {
        "dataset": dataset_name,
        "accuracy": accuracy_score(y, predictions),
        "precision_macro": precision_score(
            y, predictions,
            average="macro",
            zero_division=0
        ),
        "recall_macro": recall_score(
            y, predictions,
            average="macro",
            zero_division=0
        ),
        "f1_macro": f1_score(
            y, predictions,
            average="macro",
            zero_division=0
        )
    }

    return metrics, predictions


# =====================================================
# 6. PROGRAM UTAMA
# =====================================================

def main():

    # Ekstraksi fitur
    X, y, filenames = load_dataset()

    # Pembagian dataset
    (
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        file_test
    ) = split_dataset(X, y, filenames)

    # Optimasi parameter menggunakan data training
    best_model, best_params = optimize_knn(
        X_train,
        y_train
    )

    # Evaluasi validation
    val_metrics, _ = evaluate_model(
        best_model,
        X_val,
        y_val,
        "Validation"
    )

    # Evaluasi test
    test_metrics, test_predictions = evaluate_model(
        best_model,
        X_test,
        y_test,
        "Test"
    )

    # =================================================
    # 7. MENYIMPAN MODEL
    # =================================================

    joblib.dump(best_model, MODEL_PATH)

    print("\nModel berhasil disimpan:")
    print(MODEL_PATH)

    # =================================================
    # 8. MENYIMPAN HASIL EVALUASI
    # =================================================

    metrics_df = pd.DataFrame([
        val_metrics,
        test_metrics
    ])

    metrics_df.to_csv(METRICS_CSV, index=False)

    print("\nHasil evaluasi:")
    print(metrics_df.to_string(index=False))

    # Classification report pada data test
    class_labels = sorted(np.unique(y))

    report = classification_report(
        y_test,
        test_predictions,
        labels=class_labels,
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(REPORT_CSV)

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        test_predictions,
        labels=class_labels,
        zero_division=0
    ))

    # Confusion matrix pada data test
    cm = confusion_matrix(
        y_test,
        test_predictions,
        labels=class_labels
    )

    cm_df = pd.DataFrame(
        cm,
        index=class_labels,
        columns=class_labels
    )

    cm_df.to_csv(CONFUSION_CSV)

    print("\nConfusion Matrix:")
    print(cm_df)

    # Prediksi setiap gambar pada data test
    prediction_df = pd.DataFrame({
        "filename": file_test,
        "actual_label": y_test,
        "predicted_label": test_predictions,
        "correct": y_test == test_predictions
    })

    prediction_df.to_csv(
        PREDICTIONS_CSV,
        index=False
    )

    print("\nSemua proses selesai.")
    print("\nFile hasil:")
    print(f"Model              : {MODEL_PATH}")
    print(f"Dataset fitur      : {FEATURES_CSV}")
    print(f"Evaluasi           : {METRICS_CSV}")
    print(f"Classification     : {REPORT_CSV}")
    print(f"Confusion matrix   : {CONFUSION_CSV}")
    print(f"Prediksi test      : {PREDICTIONS_CSV}")
    print(f"Pembagian dataset  : {SPLIT_CSV}")


if __name__ == "__main__":
    main()