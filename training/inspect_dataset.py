
from pathlib import Path
import pandas as pd
from PIL import Image, UnidentifiedImageError

# =====================================================
# 1. KONFIGURASI DIREKTORI
# =====================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_DIR = BASE_DIR / "dataset"
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_OUTPUT = OUTPUT_DIR / "audit_dataset.csv"

# Ekstensi gambar yang diperbolehkan
VALID_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp"
}


# =====================================================
# 2. FUNGSI AUDIT DATASET
# =====================================================

def inspect_dataset():

    if not DATASET_DIR.exists():
        print("Folder dataset tidak ditemukan!")
        print(f"Lokasi: {DATASET_DIR}")
        return

    results = []

    class_folders = sorted([
        folder for folder in DATASET_DIR.iterdir()
        if folder.is_dir()
    ])

    if not class_folders:
        print("Tidak ditemukan folder kelas dalam dataset.")
        return

    total_images = 0
    total_valid = 0
    total_invalid = 0

    print("=" * 60)
    print("AUDIT DATASET CITRA DAUN KENTANG")
    print("=" * 60)

    # =================================================
    # 3. PEMERIKSAAN SETIAP KELAS
    # =================================================

    for class_folder in class_folders:

        class_name = class_folder.name

        image_files = [
            file for file in class_folder.rglob("*")
            if file.is_file()
            and file.suffix.lower() in VALID_EXTENSIONS
        ]

        class_valid = 0
        class_invalid = 0

        print(f"\nKelas: {class_name}")
        print(f"Jumlah file gambar: {len(image_files)}")

        # =============================================
        # 4. PEMERIKSAAN SETIAP GAMBAR
        # =============================================

        for image_path in image_files:

            total_images += 1

            try:
                with Image.open(image_path) as img:

                    width, height = img.size
                    image_format = img.format
                    image_mode = img.mode

                    # Memastikan gambar dapat dibaca
                    img.verify()

                class_valid += 1
                total_valid += 1

                status = "Valid"
                error_message = ""

            except (UnidentifiedImageError, OSError, ValueError) as error:

                width = None
                height = None
                image_format = None
                image_mode = None

                class_invalid += 1
                total_invalid += 1

                status = "Invalid"
                error_message = str(error)

            results.append({
                "filename": image_path.name,
                "filepath": str(image_path.relative_to(BASE_DIR)),
                "class_label": class_name,
                "width": width,
                "height": height,
                "format": image_format,
                "mode": image_mode,
                "status": status,
                "error": error_message
            })

        print(f"Gambar valid   : {class_valid}")
        print(f"Gambar invalid : {class_invalid}")

    # =================================================
    # 5. SIMPAN HASIL AUDIT
    # =================================================

    df = pd.DataFrame(results)

    df.to_csv(CSV_OUTPUT, index=False)

    # =================================================
    # 6. RINGKASAN DATASET
    # =================================================

    print("\n" + "=" * 60)
    print("RINGKASAN HASIL AUDIT")
    print("=" * 60)

    print(f"Jumlah kelas       : {len(class_folders)}")
    print(f"Total gambar       : {total_images}")
    print(f"Gambar valid       : {total_valid}")
    print(f"Gambar invalid     : {total_invalid}")

    if total_images > 0:
        percentage_valid = total_valid / total_images * 100
        print(f"Persentase valid   : {percentage_valid:.2f}%")

    print("\nJumlah gambar per kelas:")

    if not df.empty:
        summary = df.groupby(
            ["class_label", "status"]
        ).size().unstack(fill_value=0)

        print(summary)

    print("\nHasil audit disimpan di:")
    print(CSV_OUTPUT)


# =====================================================
# 7. PROGRAM UTAMA
# =====================================================

if __name__ == "__main__":
    inspect_dataset()
