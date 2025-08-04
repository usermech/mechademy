import cv2
import os
import glob

# Görsellerin bulunduğu klasör
image_folder = r'C:\Users\ofsaa\Desktop\photoso\sim_navigation_output'  # örnek: './frames'

# Çıktı videosu ismi
video_name = r'C:\Users\ofsaa\Desktop\mechacademy\mechademy\local\map_navigation\trajectory_video.mp4'

# Bütün jpg dosyalarını al, sıralı olarak
images = sorted(glob.glob(os.path.join(image_folder, '*.png')))

# İlk görüntüden boyutları al
frame = cv2.imread(images[0])
height, width, layers = frame.shape
size = (width, height)

# VideoWriter objesi oluştur
out = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), 15, size)  # 10 = FPS

# Tüm görselleri sırayla ekle
for i in range(len(images)):
    img = cv2.imread(images[i])
    out.write(img)

out.release()
print("✅ Video oluşturuldu:", video_name)
