import cv2
import os
import numpy as np
import requests
import threading
from requests.auth import HTTPDigestAuth

class ThetaLivePreview:
    def __init__(self, save_dir="./snapshots"):
        # Camera settings
        self.url = "http://192.168.68.55/osc/commands/execute"
        self.auth = HTTPDigestAuth("THETAYR20102028", "20102028")
        self.payload = {"name": "camera.getLivePreview"}
        self.headers = {"Content-Type": "application/json;charset=utf-8"}

        # Image saving
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.frame = None
        self.running = True
        self.frame_counter = 0

        # Start preview stream
        self.thread = threading.Thread(target=self.stream_preview)
        self.thread.daemon = True
        self.thread.start()

    def stream_preview(self):
        print("Connecting to Theta live preview...")
        response = requests.post(self.url, auth=self.auth, json=self.payload,
                                 headers=self.headers, stream=True)

        if response.status_code != 200:
            print(f"Failed to start live preview: {response.status_code}")
            self.running = False
            return

        bytes_ = bytes()

        for chunk in response.iter_content(chunk_size=1024):
            if not self.running:
                break
            bytes_ += chunk
            a = bytes_.find(b'\xff\xd8')
            b = bytes_.find(b'\xff\xd9')
            if a != -1 and b != -1:
                jpg = bytes_[a:b+2]
                bytes_ = bytes_[b+2:]
                img = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                if img is not None:
                    self.frame = img

    def save_snapshot(self):
        if self.frame is not None:
            filename = os.path.join(self.save_dir, f"snapshot_{self.frame_counter:05d}.jpg")
            self.frame_counter += 1
            cv2.imwrite(filename, self.frame)
            print(f"Snapshot saved: {filename}")
        else:
            print("No frame available yet. Try again shortly.")

    def stop(self):
        self.running = False


if __name__ == "__main__":
    theta = ThetaLivePreview()

    print("Press Enter to take a snapshot. Type 'q' and Enter to quit.")

    try:
        while True:
            user_input = input()
            if user_input.strip().lower() == "q":
                break
            theta.save_snapshot()
    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        theta.stop()
        print("Exited cleanly.")
