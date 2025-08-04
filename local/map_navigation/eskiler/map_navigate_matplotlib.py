import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import json
from matplotlib.animation import FuncAnimation, FFMpegWriter

# --- Veri Yükleme ---
npofmap = r"C:\Users\ofsaa\Desktop\mechacademy\mechademy\local\top_down_map_last.npy"
topdown_map = np.load(npofmap)

json_list = r"C:\Users\ofsaa\Desktop\mechacademy\mechademy\local\lists.json"
with open(json_list, "r") as f:
    data = json.load(f)


poses = data["poses"]  # Tüm olası node pozisyonları
trajectory = data["trajectory"]  # Planlanan path

# --- Noktaları ayır ---

# Tüm node'lar → kırmızı noktalar
all_x, all_y = zip(*poses)

# Planlanan path → yeşil çizgi ve yeşil noktalar
planned_xy = [poses[i] for i in trajectory]
planned_x, planned_y = zip(*planned_xy)

# Estimated trajectory → animasyonla ilerleyecek (başta boş)
est_x, est_y = planned_x, planned_y  # Şimdilik estimated = planned

# --- Grafik Ayarları ---
fig, ax = plt.subplots()
ax.imshow(topdown_map)

# 1. Tüm node'ları kırmızı nokta olarak göster
ax.scatter(all_x, all_y, c='red', s=10, label='All Navigable Nodes')

# 2. Planlanan path'i yeşil çizgiyle çiz + noktaları yeşil yap
ax.plot(planned_x, planned_y, 'g-', linewidth=2.5, label='Planned Path')
ax.scatter(planned_x, planned_y, c='green', s=20, label='Planned Nodes')

# 3. Estimated trajectory (animasyonla mavi çizgi)
traj_line, = ax.plot([], [], 'b-', linewidth=2.5, label='Estimated Trajectory')

# Başlangıç noktası → mavi kare
ax.scatter(planned_x[0], planned_y[0], c='blue', marker='s', s=60, label='Start Node')

# Bitiş noktası → yeşil kare
ax.scatter(planned_x[-1], planned_y[-1], c='green', marker='s', s=60, label='Goal Node')


# --- Animasyon Fonksiyonu ---
def update(frame):
    traj_line.set_data(est_x[:frame+1], est_y[:frame+1])
    return traj_line,

# --- Animasyonu başlat ---
ani = FuncAnimation(fig, update, frames=len(est_x), interval=100, blit=True)
ax.legend(loc = 4, handlelength= 0.5, handleheight = 0.3)
plt.show()


ani.save("trajectory_animation.gif", writer='pillow', fps=10)



