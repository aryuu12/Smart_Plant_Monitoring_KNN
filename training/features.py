
import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops


# =====================================================
# 1. DAFTAR NAMA FITUR
# =====================================================

FEATURE_NAMES = [
    "h_mean",
    "h_std",
    "s_mean",
    "s_std",
    "v_mean",
    "v_std",
    "green_ratio",
    "yellow_ratio",
    "brown_ratio",
    "glcm_contrast",
    "glcm_dissimilarity",
    "glcm_homogeneity",
    "glcm_energy",
    "glcm_correlation",
    "glcm_asm"
]


# =====================================================
# 2. FUNGSI EKSTRAKSI FITUR
# =====================================================

def extract_features(image_rgb):

    """
    Mengekstraksi fitur warna HSV dan tekstur GLCM
    dari citra daun kentang.

    Input:
        image_rgb : citra RGB dalam bentuk NumPy array

    Output:
        feature_vector : array berisi 15 fitur numerik
    """

    # -------------------------------------------------
    # A. VALIDASI GAMBAR
    # -------------------------------------------------

    if image_rgb is None or image_rgb.size == 0:
        raise ValueError("Citra kosong atau tidak valid.")

    image_rgb = np.asarray(
        image_rgb,
        dtype=np.uint8
    )

    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("Citra harus memiliki tiga kanal RGB.")

    # -------------------------------------------------
    # B. KONVERSI RGB KE HSV
    # -------------------------------------------------

    hsv = cv2.cvtColor(
        image_rgb,
        cv2.COLOR_RGB2HSV
    )

    h, s, v = cv2.split(hsv)

    # -------------------------------------------------
    # C. FITUR STATISTIK HSV
    # -------------------------------------------------

    h_mean = np.mean(h)
    h_std = np.std(h)

    s_mean = np.mean(s)
    s_std = np.std(s)

    v_mean = np.mean(v)
    v_std = np.std(v)

    # -------------------------------------------------
    # D. SEGMENTASI KANDIDAT WARNA
    # -------------------------------------------------

    green_mask = cv2.inRange(
        hsv,
        np.array([35, 35, 30], dtype=np.uint8),
        np.array([85, 255, 255], dtype=np.uint8)
    )

    yellow_mask = cv2.inRange(
        hsv,
        np.array([20, 35, 30], dtype=np.uint8),
        np.array([34, 255, 255], dtype=np.uint8)
    )

    brown_mask = cv2.inRange(
        hsv,
        np.array([5, 40, 20], dtype=np.uint8),
        np.array([19, 255, 220], dtype=np.uint8)
    )

    total_pixels = hsv.shape[0] * hsv.shape[1]

    green_ratio = (
        np.count_nonzero(green_mask) / total_pixels
    )

    yellow_ratio = (
        np.count_nonzero(yellow_mask) / total_pixels
    )

    brown_ratio = (
        np.count_nonzero(brown_mask) / total_pixels
    )

    # -------------------------------------------------
    # E. KONVERSI RGB KE GRAYSCALE
    # -------------------------------------------------

    gray = cv2.cvtColor(
        image_rgb,
        cv2.COLOR_RGB2GRAY
    )

    # Resize agar ukuran citra seragam
    gray = cv2.resize(
        gray,
        (128, 128),
        interpolation=cv2.INTER_AREA
    )

    # Kuantisasi menjadi 32 tingkat keabuan
    gray_quantized = (gray // 8).astype(np.uint8)

    # -------------------------------------------------
    # F. MEMBANGUN MATRIKS GLCM
    # -------------------------------------------------

    glcm = graycomatrix(
        gray_quantized,
        distances=[1, 2],
        angles=[
            0,
            np.pi / 4,
            np.pi / 2,
            3 * np.pi / 4
        ],
        levels=32,
        symmetric=True,
        normed=True
    )

    # -------------------------------------------------
    # G. EKSTRAKSI FITUR TEKSTUR GLCM
    # -------------------------------------------------

    contrast = np.mean(
        graycoprops(glcm, "contrast")
    )

    dissimilarity = np.mean(
        graycoprops(glcm, "dissimilarity")
    )

    homogeneity = np.mean(
        graycoprops(glcm, "homogeneity")
    )

    energy = np.mean(
        graycoprops(glcm, "energy")
    )

    correlation = np.mean(
        graycoprops(glcm, "correlation")
    )

    asm = np.mean(
        graycoprops(glcm, "ASM")
    )

    # -------------------------------------------------
    # H. MENGGABUNGKAN SEMUA FITUR
    # -------------------------------------------------

    feature_vector = np.array(
        [
            h_mean,
            h_std,
            s_mean,
            s_std,
            v_mean,
            v_std,
            green_ratio,
            yellow_ratio,
            brown_ratio,
            contrast,
            dissimilarity,
            homogeneity,
            energy,
            correlation,
            asm
        ],
        dtype=np.float64
    )

    return feature_vector
