import os
import cv2

# --- AYARLAR ---
input_dir = r"C:\Users\ofsaa\Desktop\mechacademy\mechademy\local\local_sim_photo\00800-TEEsavR23oF"     # Girdi klasörü
output_dir = r"C:\Users\ofsaa\Desktop\mechacademy\mechademy\local\local_sim_photo_720"   # Çıktı klasörü
new_width, new_height = 720, 360                      # Yeni boyutlar

# Çıktı klasörü yoksa oluştur
os.makedirs(output_dir, exist_ok=True)

# Desteklenen uzantılar
valid_exts = [".png", ".jpg", ".jpeg"]

# Klasördeki tüm dosyaları işle
for filename in os.listdir(input_dir):
    file_ext = os.path.splitext(filename)[1].lower()
    if file_ext in valid_exts:
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        # Görseli yükle
        img = cv2.imread(input_path)
        if img is None:
            print(f"Atlandı (yüklenemedi): {filename}")
            continue

        # Yeniden boyutlandır
        resized = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)

        # Kaydet
        cv2.imwrite(output_path, resized)
        print(f"Kaydedildi: {output_path}")
