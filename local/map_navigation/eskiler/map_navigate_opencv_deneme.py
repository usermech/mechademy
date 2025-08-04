import cv2
import numpy as np
import json

# --- Ayarlar ---
scale = 0.5  # %50 küçült
animation_delay = 50  # ms
alpha = 0.8
beta = -30

# --- Dosya yolları ---
npofmap = r"C:\Users\ofsaa\Desktop\mechacademy\mechademy\local\top_down_map_last.npy"
json_list = r"C:\Users\ofsaa\Desktop\mechacademy\mechademy\local\lists.json"

# --- Veri yükle ---
topdown_map = np.load(npofmap)
topdown_map = cv2.convertScaleAbs(topdown_map, alpha=alpha, beta=beta)

# dtype düzelt → OpenCV uyumlu hale getir
if topdown_map.dtype != np.uint8:  
    topdown_map = topdown_map.astype(np.uint8)

# Renkleri tersine çevir (iç ve dış kısımların renklerini değiştir)
topdown_map = 255 - topdown_map  # <--- BU SATIRI EKLEYİN

# Tek kanal ise 3 kanala dönüştür (BGR)
if len(topdown_map.shape) == 2:
    topdown_map = cv2.cvtColor(topdown_map, cv2.COLOR_GRAY2BGR)

# Yeniden boyutlandır
new_size = (int(topdown_map.shape[1] * scale), int(topdown_map.shape[0] * scale))
topdown_map = cv2.resize(topdown_map, new_size, interpolation=cv2.INTER_AREA)

# Bellek uyumlu hale getir
topdown_map = np.ascontiguousarray(topdown_map)

# JSON verilerini al
with open(json_list, "r") as f:
    data = json.load(f)

poses = data["poses"]
trajectory = data["trajectory"]


# Koordinatları ölçekle
def scale_point(p):
    return (int(p[0] * scale), int(p[1] * scale))

scaled_poses = [scale_point(p) for p in poses]
scaled_traj_xy = [scaled_poses[i] for i in trajectory]


########output_path = r"C:\Users\ofsaa\Desktop\mechacademy\mechademy\local\map_navigation\output_video_inner_light_.mp4"
########fps = int(1000 / animation_delay)  # delay'den FPS hesapla
########fourcc = cv2.VideoWriter_fourcc(*'mp4v')
########video_writer = cv2.VideoWriter(output_path, fourcc, fps, new_size)

# --- Sürekli animasyon döngüsü ---
while True:
    frame = topdown_map.copy()

    # Tüm node'ları çiz (kırmızı noktalar)
    for x, y in scaled_poses:
        cv2.circle(frame, (x, y), 3, (0, 0, 200), -1)

    # Planlanan yeşil çizgi
    for i in range(len(scaled_traj_xy) - 1):
        cv2.line(frame, scaled_traj_xy[i], scaled_traj_xy[i+1], (50, 200, 30), 7)

    for x, y in scaled_traj_xy:
        cv2.circle(frame, (x, y), 4, (0, 200, 0), -1)

    # Başlangıç/bitiş noktası
    cv2.rectangle(frame, (scaled_traj_xy[0][0]-5, scaled_traj_xy[0][1]-5),
                  (scaled_traj_xy[0][0]+5, scaled_traj_xy[0][1]+5), (144, 0, 3), 2)
    cv2.rectangle(frame, (scaled_traj_xy[-1][0]-5, scaled_traj_xy[-1][1]-5),
                  (scaled_traj_xy[-1][0]+5, scaled_traj_xy[-1][1]+5), (0, 200, 0), 2)

    # Mavi çizgiyi adım adım çiz
    for i in range(1, len(scaled_traj_xy)):
        pt1 = scaled_traj_xy[i - 1]
        pt2 = scaled_traj_xy[i]
        cv2.line(frame, pt1, pt2, (144, 0, 3), 7)  # mavi

        ########video_writer.write(frame)  # videoya yaz

        cv2.imshow("Looped Navigation", frame)
        key = cv2.waitKey(animation_delay) & 0xFF
        if key == 27:  # ESC ile çık
            cv2.destroyAllWindows()
            exit()

    # Son çizim bitince 1 saniye beklet
    if cv2.waitKey(1000) & 0xFF == 27:
        break


########video_writer.release()
########cv2.destroyAllWindows()