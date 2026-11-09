import sys
from pathlib import Path

# Pastikan direktori utama proyek masuk ke jalur pencarian modul Python.
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import streamlit as st
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import json

from datetime import datetime
from PIL import Image
from io import BytesIO
from training.features import extract_features, FEATURE_NAMES

st.set_page_config(
    page_title="Smart Plant Monitoring",
    page_icon="🌱",
    layout="wide"
)

OUTPUT_DIR = BASE_DIR / "outputsApp"
MASK_DIR = OUTPUT_DIR / "masks"
CSV_FILE = OUTPUT_DIR / "hasil_monitoring.csv"
MODEL_PATH = BASE_DIR / "models" / "model_knn_kentang.joblib"
METRICS_PATH = BASE_DIR / "outputs" / "evaluasi_knn.csv"
REPORT_PATH = BASE_DIR / "outputs" / "classification_report_knn.csv"
CONFUSION_PATH = BASE_DIR / "outputs" / "confusion_matrix_knn.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MASK_DIR.mkdir(parents=True, exist_ok=True)


def segmentasi_hijau(image_rgb, lower_hsv, upper_hsv, kernel_size=5):
    """Segmentasi kandidat warna hijau dengan HSV."""
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(
        hsv,
        np.array(lower_hsv, dtype=np.uint8),
        np.array(upper_hsv, dtype=np.uint8)
    )
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    green_pixels = cv2.countNonZero(mask)
    total_pixels = mask.size
    percentage = green_pixels / total_pixels * 100 if total_pixels else 0.0
    segmented = cv2.bitwise_and(image_rgb, image_rgb, mask=mask)

    return hsv, mask, segmented, total_pixels, green_pixels, percentage


@st.cache_resource
def load_model(model_path_string, modified_time):
    # modified_time menjadi bagian cache key agar model yang baru dilatih dimuat ulang.
    return joblib.load(model_path_string)


def save_history(record):
    new_df = pd.DataFrame([record])
    if CSV_FILE.exists():
        old_df = pd.read_csv(CSV_FILE)
        all_df = pd.concat([old_df, new_df], ignore_index=True)
    else:
        all_df = new_df
    all_df.to_csv(CSV_FILE, index=False, encoding="utf-8-sig")
    return all_df


def safe_name(value):
    value = "".join(
        char for char in str(value).strip()
        if char.isalnum() or char in ("-", "_")
    )
    return value or "tanaman"


st.title("🌱 Smart Plant Monitoring — Daun Kentang")
st.write(
    """
    Aplikasi ini mempertahankan monitoring segmentasi warna HSV dan menambahkan
    klasifikasi menggunakan **K-Nearest Neighbors (KNN)**. Model KNN menggunakan
    gabungan fitur warna HSV dan tekstur GLCM. Label prediksi mengikuti nama
    folder kelas asli dataset saat pelatihan.
    """
)
st.divider()

st.sidebar.header("Pengaturan Segmentasi HSV")
h_min = st.sidebar.slider("Hue minimum", 0, 179, 25)
h_max = st.sidebar.slider("Hue maksimum", 0, 179, 95)
s_min = st.sidebar.slider("Saturation minimum", 0, 255, 35)
s_max = st.sidebar.slider("Saturation maksimum", 0, 255, 255)
v_min = st.sidebar.slider("Value minimum", 0, 255, 30)
v_max = st.sidebar.slider("Value maksimum", 0, 255, 255)
kernel_size = st.sidebar.select_slider(
    "Ukuran kernel morfologi",
    options=[3, 5, 7, 9],
    value=5
)

if h_min > h_max or s_min > s_max or v_min > v_max:
    st.sidebar.error("Nilai minimum tidak boleh lebih besar dari nilai maksimum.")
    st.stop()

lower_hsv = [h_min, s_min, v_min]
upper_hsv = [h_max, s_max, v_max]

knn_model = None

if MODEL_PATH.exists():
    try:
        model_artifact = load_model(
            str(MODEL_PATH),
            MODEL_PATH.stat().st_mtime
        )

        # Model disimpan sebagai Pipeline secara langsung
        knn_model = model_artifact

        # Mengambil label kelas dari model
        class_names = knn_model.classes_

        st.sidebar.success("Model KNN berhasil dimuat!")
        st.sidebar.caption("Fitur: HSV + GLCM")

        st.sidebar.write("Label dataset:")
        for class_name in class_names:
            st.sidebar.write(f"- {class_name}")

        st.sidebar.write("Parameter KNN:")
        st.sidebar.write(f"K: {knn_model.named_steps['knn'].n_neighbors}")
        st.sidebar.write(f"Metric: {knn_model.named_steps['knn'].metric}")
        st.sidebar.write(f"Weight: {knn_model.named_steps['knn'].weights}")

    except Exception as error:
        st.sidebar.error(
            f"Model tidak dapat dimuat: {repr(error)}"
        )

else:
    st.sidebar.warning(
        "Model belum tersedia. Jalankan training terlebih dahulu."
    )

st.subheader("1. Input Data Tanaman")
col_a, col_b = st.columns(2)
with col_a:
    plant_id = st.text_input("ID Tanaman", value="P001")
with col_b:
    location = st.text_input("Lokasi Pengamatan", value="Kebun Percobaan")

uploaded_file = st.file_uploader(
    "Unggah foto daun kentang",
    type=["jpg", "jpeg", "png", "bmp", "webp"]
)

if uploaded_file is not None:
    try:
        pil_image = Image.open(
            BytesIO(uploaded_file.getvalue())
        ).convert("RGB")
        image_rgb = np.array(pil_image)

        hsv, mask, segmented, total_pixels, green_pixels, green_percentage = (
            segmentasi_hijau(
                image_rgb,
                lower_hsv,
                upper_hsv,
                kernel_size
            )
        )

        feature_vector = np.asarray(
            extract_features(image_rgb), dtype=np.float64
        ).reshape(1, -1)

        prediction = None
        voting_df = None
        feature_df = pd.DataFrame({
            "Nama fitur": FEATURE_NAMES,
            "Nilai": feature_vector[0]
        })

        if knn_model is not None:
            prediction = str(knn_model.predict(feature_vector)[0])
            if hasattr(knn_model, "predict_proba"):
                voting = knn_model.predict_proba(feature_vector)[0]
                voting_df = pd.DataFrame({
                    "Label kelas": knn_model.classes_,
                    "Proporsi voting (%)": np.round(voting * 100, 2)
                }).sort_values(
                    "Proporsi voting (%)",
                    ascending=False
                )

        st.subheader("2. Hasil Pengolahan Citra")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(image_rgb, caption="Gambar asli", use_container_width=True)
        with col2:
            st.image(mask, caption="Mask segmentasi HSV", use_container_width=True)
        with col3:
            st.image(segmented, caption="Hasil segmentasi", use_container_width=True)

        st.subheader("3. Hasil Monitoring HSV")
        m1, m2, m3 = st.columns(3)
        m1.metric("Total piksel", f"{total_pixels:,}")
        m2.metric("Piksel hijau", f"{green_pixels:,}")
        m3.metric("Persentase piksel hijau", f"{green_percentage:.2f}%")
        st.progress(min(max(green_percentage / 100, 0.0), 1.0))
        st.caption(
            "Persentase ini adalah proporsi piksel yang lolos masker HSV, "
            "bukan otomatis persentase luas daun atau tingkat kesehatan."
        )

        st.subheader("4. Hasil Klasifikasi KNN")
        if prediction is not None:
            st.success(f"Label prediksi sesuai dataset: {prediction}")
            if voting_df is not None:
                st.write("Proporsi voting tetangga:")
                st.dataframe(voting_df, use_container_width=True, hide_index=True)
                st.caption(
                    "Proporsi voting bukan probabilitas biologis atau tingkat "
                    "kepastian penyakit yang telah dikalibrasi."
                )
        else:
            st.warning(
                "Model KNN belum tersedia. Audit dataset dan latih model terlebih dahulu."
            )

        with st.expander("Lihat nilai fitur HSV dan GLCM"):
            st.dataframe(feature_df, use_container_width=True, hide_index=True)

        st.warning(
            "Prediksi merupakan klasifikasi visual berdasarkan label dataset, "
            "bukan diagnosis pasti penyebab penyakit."
        )

        safe_plant_id = safe_name(plant_id)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        mask_filename = f"{safe_plant_id}_{timestamp}_mask.png"
        mask_path = MASK_DIR / mask_filename
        cv2.imwrite(str(mask_path), mask)

        mask_buffer = BytesIO()
        Image.fromarray(mask).save(mask_buffer, format="PNG")
        st.download_button(
            "Unduh mask segmentasi",
            data=mask_buffer.getvalue(),
            file_name=f"{safe_plant_id}_mask.png",
            mime="image/png"
        )

        segmented_buffer = BytesIO()
        Image.fromarray(segmented).save(segmented_buffer, format="PNG")
        st.download_button(
            "Unduh hasil segmentasi",
            data=segmented_buffer.getvalue(),
            file_name=f"{safe_plant_id}_segmentasi.png",
            mime="image/png"
        )

        record = {
            "waktu_pemrosesan": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "id_tanaman": plant_id.strip(),
            "lokasi": location.strip(),
            "nama_file": uploaded_file.name,
            "total_piksel": int(total_pixels),
            "piksel_hijau": int(green_pixels),
            "persentase_hijau": round(float(green_percentage), 2),
            "prediksi_knn": prediction if prediction is not None else "Model belum tersedia",
            "hsv_bawah": str(lower_hsv),
            "hsv_atas": str(upper_hsv),
            "ukuran_kernel": int(kernel_size),
            "file_mask": str(mask_path.relative_to(BASE_DIR))
        }
        record.update({
            name: round(float(value), 6)
            for name, value in zip(FEATURE_NAMES, feature_vector[0])
        })

        if "last_saved_upload" not in st.session_state:
            st.session_state.last_saved_upload = None

        upload_key = (
            uploaded_file.name,
            len(uploaded_file.getvalue()),
            plant_id.strip(),
            location.strip(),
            prediction
        )

        if st.button("Simpan hasil monitoring dan klasifikasi", type="primary"):
            if st.session_state.last_saved_upload == upload_key:
                st.info("Hasil ini sudah disimpan pada sesi saat ini.")
            else:
                save_history(record)
                st.session_state.last_saved_upload = upload_key
                st.success("Hasil berhasil disimpan ke riwayat CSV.")

    except Exception as error:
        st.error(f"Terjadi kesalahan saat memproses gambar: {error}")

st.divider()
st.subheader("5. Riwayat Monitoring")

if CSV_FILE.exists():
    history = pd.read_csv(CSV_FILE)
    if not history.empty:
        st.dataframe(history, use_container_width=True, hide_index=True)
        st.download_button(
            "Unduh riwayat CSV",
            data=history.to_csv(index=False).encode("utf-8-sig"),
            file_name="hasil_monitoring.csv",
            mime="text/csv"
        )

        if "persentase_hijau" in history.columns:
            chart_data = history.copy()
            chart_data["persentase_hijau"] = pd.to_numeric(
                chart_data["persentase_hijau"],
                errors="coerce"
            )
            chart_data = chart_data.dropna(subset=["persentase_hijau"])
            if not chart_data.empty:
                labels = (
                    chart_data["id_tanaman"].astype(str)
                    + " - "
                    + chart_data["nama_file"].astype(str)
                )
                fig, ax = plt.subplots(figsize=(11, 4))
                ax.bar(labels, chart_data["persentase_hijau"])
                ax.set_title("Persentase Piksel yang Terdeteksi Hijau")
                ax.set_xlabel("Tanaman / gambar")
                ax.set_ylabel("Persentase (%)")
                ax.tick_params(axis="x", labelrotation=45)
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

        if "prediksi_knn" in history.columns:
            st.subheader("Distribusi Prediksi KNN")
            counts = history["prediksi_knn"].value_counts()
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.bar(counts.index.astype(str), counts.values)
            ax.set_xlabel("Label kelas dataset")
            ax.set_ylabel("Jumlah prediksi")
            ax.tick_params(axis="x", labelrotation=35)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
    else:
        st.info("Belum ada data monitoring.")
else:
    st.info("Belum ada riwayat monitoring.")

st.divider()
with st.expander("Metrik evaluasi model KNN"):
    # Nama file disesuaikan dengan keluaran train_knn.py.
    if METRICS_PATH.exists():
        metrics_df = pd.read_csv(METRICS_PATH)

        if "dataset" in metrics_df.columns:
            test_rows = metrics_df[
                metrics_df["dataset"].astype(str).str.lower() == "test"
            ]
        else:
            test_rows = pd.DataFrame()

        if not test_rows.empty:
            test_metrics = test_rows.iloc[0]
            metric_cols = st.columns(4)
            metric_cols[0].metric(
                "Test accuracy", f"{float(test_metrics['accuracy']):.4f}"
            )
            metric_cols[1].metric(
                "Test precision (macro)",
                f"{float(test_metrics['precision_macro']):.4f}"
            )
            metric_cols[2].metric(
                "Test recall (macro)",
                f"{float(test_metrics['recall_macro']):.4f}"
            )
            metric_cols[3].metric(
                "Test macro-F1", f"{float(test_metrics['f1_macro']):.4f}"
            )
        else:
            st.info("Baris evaluasi Test tidak ditemukan di evaluasi_knn.csv.")

        st.write("Tabel evaluasi:")
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)
    else:
        st.info(
            "Metrik belum tersedia. Jalankan "
            "'python -m training.train_knn' terlebih dahulu."
        )

    if REPORT_PATH.exists():
        st.write("Classification report:")
        report_df = pd.read_csv(REPORT_PATH, index_col=0)
        st.dataframe(report_df, use_container_width=True)

    if CONFUSION_PATH.exists():
        st.write("Confusion matrix data test:")
        confusion_df = pd.read_csv(CONFUSION_PATH, index_col=0)
        st.dataframe(confusion_df, use_container_width=True)

        fig, ax = plt.subplots(figsize=(7, 5))
        image_plot = ax.imshow(confusion_df.values, interpolation="nearest")
        ax.set_title("Confusion Matrix — Data Test")
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("Actual label")
        ax.set_xticks(range(len(confusion_df.columns)))
        ax.set_xticklabels(confusion_df.columns, rotation=35, ha="right")
        ax.set_yticks(range(len(confusion_df.index)))
        ax.set_yticklabels(confusion_df.index)

        for row in range(confusion_df.shape[0]):
            for col in range(confusion_df.shape[1]):
                ax.text(
                    col, row, str(confusion_df.iloc[row, col]),
                    ha="center", va="center"
                )

        fig.colorbar(image_plot, ax=ax)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

st.caption("Smart Plant Monitoring | OpenCV + HSV + GLCM + KNN + Streamlit")