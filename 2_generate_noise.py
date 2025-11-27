import cv2
import numpy as np
import os

os.system("cls")

def add_salt_pepper_noise(image, prob):
    output = np.copy(image)
    if len(image.shape) == 3: # Color
        h, w, c = image.shape
        rnd = np.random.rand(h, w)
        salt_mask = rnd < (prob / 2)
        output[salt_mask] = 255
        pepper_mask = rnd > (1 - prob / 2)
        output[pepper_mask] = 0
    else: # Grayscale
        h, w = image.shape
        rnd = np.random.rand(h, w)
        output[rnd < (prob / 2)] = 255
        output[rnd > (1 - prob / 2)] = 0
    return output

def add_gaussian_noise(image, mean, sigma):
    row, col = image.shape[:2]
    if len(image.shape) == 3:
        ch = image.shape[2]
        gauss = np.random.normal(mean, sigma, (row, col, ch))
    else:
        gauss = np.random.normal(mean, sigma, (row, col))
    noisy = image + gauss
    noisy = np.clip(noisy, 0, 255)
    return noisy.astype(np.uint8)

def process_noise():
    print("=== MULAI LANGKAH 2: Generate Noise ===")
    
    base_output_dir = 'output'
    categories = ['landscape', 'portrait']
    
    # Parameter Noise
    sp_levels = {'lvl1': 0.02, 'lvl2': 0.05}
    gauss_levels = {'lvl1': (0, 15), 'lvl2': (0, 40)}

    for category in categories:
        # --- KONFIGURASI PATH ---
        # Input dari Langkah 1: output/<category>/original
        input_dir = os.path.join(base_output_dir, category, 'original')
        # Output Langkah 2: output/<category>/noise
        output_dir = os.path.join(base_output_dir, category, 'noise')
        
        # Cek apakah langkah 1 sudah dijalankan
        if not os.path.exists(input_dir):
            print(f"[SKIP] Folder input '{input_dir}' tidak ditemukan.")
            print("       Silakan jalankan '1_convert_grayscale.py' terlebih dahulu.")
            continue
            
        os.makedirs(output_dir, exist_ok=True)
        
        # Proses file dari folder original
        found_files = False
        for base_filename in ['color.jpg', 'gray.jpg']:
            full_input_path = os.path.join(input_dir, base_filename)
            
            if not os.path.exists(full_input_path):
                continue
            
            found_files = True
            img = cv2.imread(full_input_path)
            
            # Pastikan channel benar
            is_gray_file = 'gray' in base_filename
            if is_gray_file:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            img_type = os.path.splitext(base_filename)[0]

            # Generate Salt & Pepper
            for lvl, prob in sp_levels.items():
                noisy = add_salt_pepper_noise(img, prob)
                fname = f"{img_type}_SP_{lvl}.jpg"
                cv2.imwrite(os.path.join(output_dir, fname), noisy)

            # Generate Gaussian
            for lvl, (mean, sigma) in gauss_levels.items():
                noisy = add_gaussian_noise(img, mean, sigma)
                fname = f"{img_type}_Gauss_{lvl}.jpg"
                cv2.imwrite(os.path.join(output_dir, fname), noisy)
        
        if found_files:
            print(f"[OK] {category}: Noise generated di -> {output_dir}")
        else:
            print(f"[WARN] {category}: Tidak ada file gambar di folder original.")

    print("=== LANGKAH 2 SELESAI ===\n")

if __name__ == "__main__":
    process_noise()