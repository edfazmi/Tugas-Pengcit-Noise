import cv2
import numpy as np
import os
import csv
from numpy.lib.stride_tricks import as_strided

os.system("cls")

def get_windows(image, kernel_size):
    """
    Fungsi helper untuk sliding window manual (tanpa loop lambat).
    FIXED: Memperbaiki dimensi view_shape untuk citra berwarna.
    """
    pad = kernel_size // 2
    
    if len(image.shape) == 3: # === CITRA BERWARNA ===
        # Padding (H, W, C) -> hanya padding H dan W
        img_pad = np.pad(image, ((pad, pad), (pad, pad), (0, 0)), mode='edge')
        
        # Shape output yang diinginkan: (H, W, K, K, 3)
        # H, W = geser per pixel gambar
        # K, K = geser per pixel window
        # 3    = channel warna
        sub_shape = (kernel_size, kernel_size, 3)
        view_shape = (image.shape[0], image.shape[1]) + sub_shape # (H, W, K, K, 3)
        
        # Strides:
        # 1. Pindah baris gambar (row stride)
        # 2. Pindah kolom gambar (col stride)
        # 3. Pindah baris window (row stride)
        # 4. Pindah kolom window (col stride)
        # 5. Pindah channel (channel stride)
        strides = img_pad.strides[:2] + img_pad.strides
        
        windows = as_strided(img_pad, shape=view_shape, strides=strides)
        
        # Flatten window 3x3 menjadi 9 tetangga: (H, W, 9, 3)
        return windows.reshape(image.shape[0], image.shape[1], -1, 3)
        
    else: # === CITRA GRAYSCALE ===
        img_pad = np.pad(image, ((pad, pad), (pad, pad)), mode='edge')
        
        sub_shape = (kernel_size, kernel_size)
        view_shape = (image.shape[0], image.shape[1]) + sub_shape
        
        strides = img_pad.strides + img_pad.strides
        windows = as_strided(img_pad, shape=view_shape, strides=strides)
        
        # Flatten window 3x3 menjadi 9 tetangga: (H, W, 9)
        return windows.reshape(image.shape[0], image.shape[1], -1)

def manual_filter(image, filter_type, kernel_size=3):
    windows = get_windows(image, kernel_size)
    
    # Axis 2 adalah axis neighbor (9 pixel tetangga)
    eval_axis = 2
    
    # Casting ke float untuk perhitungan rata-rata agar presisi, lalu balik ke uint8
    if filter_type == 'mean':
        output = np.mean(windows, axis=eval_axis)
    elif filter_type == 'median':
        output = np.median(windows, axis=eval_axis)
    elif filter_type == 'min':
        output = np.min(windows, axis=eval_axis)
    elif filter_type == 'max':
        output = np.max(windows, axis=eval_axis)
    
    return output.astype(np.uint8)

def calculate_mse(original, processed):
    # Konversi ke float untuk menghindari overflow saat pengurangan
    err = np.sum((original.astype("float") - processed.astype("float")) ** 2)
    err /= float(original.size) # Menggunakan size (H*W*C) agar adil
    return err

def process_filtering_and_eval():
    print("=== MULAI LANGKAH 3: Filtering & Evaluasi ===")
    
    base_output_dir = 'output'
    categories = ['landscape', 'portrait']
    
    # Setup CSV Output
    csv_path = os.path.join(base_output_dir, 'evaluasi_mse.csv')
    csv_file = open(csv_path, 'w', newline='')
    writer = csv.writer(csv_file)
    writer.writerow(['Category', 'Base_Image', 'Noise_Params', 'Filter_Type', 'MSE_Score'])

    filters = ['mean', 'median', 'min', 'max']

    for category in categories:
        # --- KONFIGURASI PATH ---
        noisy_dir = os.path.join(base_output_dir, category, 'noise')
        clean_dir = os.path.join(base_output_dir, category, 'original')
        filter_base_dir = os.path.join(base_output_dir, category, 'filter')

        if not os.path.exists(noisy_dir):
            print(f"[SKIP] Folder noise '{noisy_dir}' tidak ditemukan.")
            continue
            
        print(f"--- Memproses Kategori: {category} ---")
        noisy_files = sorted([f for f in os.listdir(noisy_dir) if f.endswith('.jpg')])

        for f_noisy in noisy_files:
            # 1. Load Noise Image
            img_noisy = cv2.imread(os.path.join(noisy_dir, f_noisy))
            
            # 2. Cari Gambar Asli (Referensi)
            base_type = f_noisy.split('_')[0] # 'color' atau 'gray'
            ref_filename = f"{base_type}.jpg"
            clean_path = os.path.join(clean_dir, ref_filename)
            
            if not os.path.exists(clean_path):
                print(f"[WARN] Referensi '{ref_filename}' tidak ditemukan. Skip.")
                continue

            img_clean = cv2.imread(clean_path)
            
            # Pastikan format grayscale sesuai
            is_gray = 'gray' in base_type
            if is_gray:
                if len(img_noisy.shape) == 3:
                    img_noisy = cv2.cvtColor(img_noisy, cv2.COLOR_BGR2GRAY)
                if len(img_clean.shape) == 3:
                    img_clean = cv2.cvtColor(img_clean, cv2.COLOR_BGR2GRAY)

            print(f"  > Processing: {f_noisy}")

            # 3. Terapkan 4 Filter
            for f_type in filters:
                try:
                    img_result = manual_filter(img_noisy, f_type, kernel_size=3)
                    mse_val = calculate_mse(img_clean, img_result)
                    
                    # 4. Simpan Output
                    save_dir = os.path.join(filter_base_dir, f_type)
                    os.makedirs(save_dir, exist_ok=True)
                    
                    out_name = f"{os.path.splitext(f_noisy)[0]}_{f_type}.jpg"
                    cv2.imwrite(os.path.join(save_dir, out_name), img_result)
                    
                    # 5. Catat CSV
                    noise_params = "_".join(f_noisy.split('_')[1:]).replace('.jpg', '')
                    writer.writerow([category, base_type, noise_params, f_type, f"{mse_val:.4f}"])
                except Exception as e:
                    print(f"    [ERROR] Gagal memproses filter {f_type}: {e}")

    csv_file.close()
    print(f"\n=== LANGKAH 3 SELESAI ===")
    print(f"File evaluasi tersimpan di: {csv_path}")

if __name__ == "__main__":
    process_filtering_and_eval()