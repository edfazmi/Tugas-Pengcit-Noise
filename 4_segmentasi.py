import cv2
import numpy as np
import os
import math

def manual_convolution(image, kernel_x, kernel_y, name="Method"):
    img_float = image.astype(np.float32)
    
    grad_x = cv2.filter2D(img_float, -1, kernel_x)
    grad_y = cv2.filter2D(img_float, -1, kernel_y)
    
    magnitude = np.sqrt(np.square(grad_x) + np.square(grad_y))
    
    max_val = magnitude.max()
    if max_val > 0:
        magnitude = (magnitude / max_val) * 255
    else:
        magnitude = magnitude * 0
        
    result_image = np.uint8(magnitude)
    return result_image

def main():
    # Folder induk
    base_input_folder = 'input_segmentasi'
    base_output_folder = 'output_segmentasi'
    
    # Format: (Path Folder Input Lengkap, Nama Sub-Folder Output)
    # Input path digabung: input_segmentasi/portrait dan input_segmentasi/landscape
    folder_pairs = [
        (os.path.join(base_input_folder, "portrait"), "portrait"),
        (os.path.join(base_input_folder, "landscape"), "landscape")
    ]

    kernels = {
        "Roberts": {
            "x": np.array([[1, 0], 
                           [0, -1]], dtype=np.float32),
            "y": np.array([[0, 1], 
                           [-1, 0]], dtype=np.float32)
        },
        "Prewitt": {
            "x": np.array([[-1, 0, 1], 
                           [-1, 0, 1], 
                           [-1, 0, 1]], dtype=np.float32),
            "y": np.array([[-1, -1, -1], 
                           [0, 0, 0], 
                           [1, 1, 1]], dtype=np.float32)
        },
        "Sobel": {
            "x": np.array([[-1, 0, 1], 
                           [-2, 0, 2], 
                           [-1, 0, 1]], dtype=np.float32),
            "y": np.array([[-1, -2, -1], 
                           [0, 0, 0], 
                           [1, 2, 1]], dtype=np.float32)
        },
        "Frei-Chen": {
            "x": np.array([[-1, 0, 1], 
                           [-np.sqrt(2), 0, np.sqrt(2)], 
                           [-1, 0, 1]], dtype=np.float32),
            "y": np.array([[-1, -np.sqrt(2), -1], 
                           [0, 0, 0], 
                           [1, np.sqrt(2), 1]], dtype=np.float32)
        }
    }

    print("[INFO] Memulai proses segmentasi...")

    # Cek apakah folder induk input ada
    if not os.path.exists(base_input_folder):
        print(f"[ERROR] Folder induk '{base_input_folder}' tidak ditemukan!")
        return

    for input_dir, output_subdir_name in folder_pairs:
        full_output_dir = os.path.join(base_output_folder, output_subdir_name)
        
        if not os.path.exists(full_output_dir):
            os.makedirs(full_output_dir)
            
        if not os.path.exists(input_dir):
            print(f"[WARNING] Sub-folder input tidak ditemukan: {input_dir}")
            continue
            
        files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        
        print(f"\nMemproses kategori: {output_subdir_name}")
        print(f"Sumber: {input_dir} | Tujuan: {full_output_dir}")
        
        if not files:
            print(f"  [INFO] Tidak ada file gambar di {input_dir}")
            continue

        for file_name in files:
            file_path = os.path.join(input_dir, file_name)
            original_img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            
            if original_img is None:
                continue
                
            print(f"  - Mengolah: {file_name}")
            
            for method_name, k_data in kernels.items():
                segmented_img = manual_convolution(
                    original_img, 
                    k_data["x"], 
                    k_data["y"], 
                    method_name
                )
                
                name_no_ext = os.path.splitext(file_name)[0]
                out_name = f"{name_no_ext}_{output_subdir_name}_{method_name}.png"
                out_path = os.path.join(full_output_dir, out_name)
                
                cv2.imwrite(out_path, segmented_img)

    print("\n[SELESAI] Cek folder output.")

if __name__ == "__main__":
    main()