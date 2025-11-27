import cv2
import os

os.system("cls")

def process_base_images():
    print("=== MULAI LANGKAH 1: Konversi Grayscale ===")
    
    # --- KONFIGURASI PATH ---
    # Input: File gambar di folder yang sama dengan script ini
    files_input = {'landscape': 'landscape.jpeg', 'portrait': 'portrait.jpeg'}
    base_output_dir = 'output'

    for category, filename in files_input.items():
        # 1. Cek Input
        if not os.path.exists(filename):
            print(f"[ERROR] File input '{filename}' tidak ditemukan di folder ini.")
            print(f"        Mohon siapkan file '{filename}' sebelum menjalankan script ini.")
            continue

        # 2. Siapkan Folder Output
        # Output Step 1: output/<category>/original/
        save_dir = os.path.join(base_output_dir, category, 'original')
        os.makedirs(save_dir, exist_ok=True)

        # 3. Baca Gambar
        img_color = cv2.imread(filename)
        if img_color is None:
            print(f"[ERROR] Gagal membaca '{filename}'. Pastikan format gambar valid.")
            continue

        # 4. Proses Grayscale
        img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

        # 5. Simpan Gambar
        path_color = os.path.join(save_dir, 'color.jpg')
        path_gray = os.path.join(save_dir, 'gray.jpg')

        cv2.imwrite(path_color, img_color)
        cv2.imwrite(path_gray, img_gray)

        print(f"[OK] {category}: Disimpan di -> {save_dir}")

    print("=== LANGKAH 1 SELESAI ===\n")

if __name__ == "__main__":
    process_base_images()