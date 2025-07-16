import paramiko
import numpy as np
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
import os
from PIL import Image 


def parse_arguments():
    available_envs = ["00800-TEEsavR23oF", "00802-wcojb4TFT35",
                      "00803-k1cupFYWXJ6", "00808-y9hTuugGdiq",]
    
    parser = argparse.ArgumentParser(
        description="SSH and run script on remote server",
        formatter_class=argparse.RawTextHelpFormatter)
    
    parser.add_argument(
        "--env",
        type=str,
        default="00800-TEEsavR23oF",
        help="Name of the scene.\nAvailable options: " + ", ".join(available_envs)
    )
    return parser.parse_args()


def get_ssh_client(hostname, port, username, password):
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(hostname, port, username, password)
    return client


def fetch_remote_file(sftp, remote_path, local_path):
    sftp.get(str(remote_path), local_path)


def load_map(local_map_path):
    map_array = np.load(local_map_path).astype(int)
    recolor_map = np.array([[255, 255, 255], [128, 128, 128], [0, 0, 0]])
    return recolor_map[map_array]


def display_image(img, title="Image", axis_on=False):
    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    plt.title(title)
    plt.axis('on' if axis_on else 'off')
    plt.show()
    plt.close()


def is_valid_position(x, y, shape, mask):
    return (0 <= x < shape[1] and 0 <= y < shape[0] and mask[y, x])


def run_remote_script(client, conda_env, script_path, x, y, env):
    conda_init = "source /home/romer/miniconda3/etc/profile.d/conda.sh"
    run_cmd = f"conda activate {conda_env} && python {script_path} --pos {y},{x} --env {env}"
    full_cmd = f"bash -c '{conda_init} && {run_cmd}'"
    stdin, stdout, stderr = client.exec_command(full_cmd)
    exit_status = stdout.channel.recv_exit_status()
    return exit_status, stderr.read().decode()

def load_coordinates_from_txt(txt_path):
    coords = []
    with open(txt_path, 'r') as file:
        for line in file:
            try:
                x, y = map(int, line.strip().split(','))
                coords.append((x, y))
            except:
                continue
    return coords


def main():
    args = parse_arguments()

    CONFIG = {
        "hostname": "192.168.68.71",
        "port": 22,
        "username": "romer",
        "password": "123",
        "conda_env": "habitat",
        "habitat_path": Path("/home/romer/umut/GibsonData/habitat-lab/my_scripts")
    }

    # KULLANICIDAN txt ve output klasör yolu al
    txt_path = r"C:\Users\ofsaa\Desktop\mechacademy\mechademy\local\coords.txt"
    output_dir = r"C:\Users\ofsaa\Desktop\mechacademy\mechademy\local\cıktı"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    coordinates = load_coordinates_from_txt(txt_path)

    remote_script = CONFIG["habitat_path"] / "oneformer" / "explore_hm3d.py"
    remote_image_path = CONFIG["habitat_path"] / "oneformer" / "image.npy"
    remote_map_path = CONFIG["habitat_path"] / "maps" / (args.env + ".npy")
    local_image_path = "./image.npy"
    local_map_path = "./map.npy"

    client = get_ssh_client(CONFIG["hostname"], CONFIG["port"], CONFIG["username"], CONFIG["password"])

    with client.open_sftp() as sftp:
        fetch_remote_file(sftp, remote_map_path.as_posix(), local_map_path)

    top_down_map = load_map(local_map_path)
    free_space_mask = np.all(top_down_map == [128, 128, 128], axis=-1)

    for x, y in coordinates:
        print(f"Running at position ({x},{y})...")

        if not is_valid_position(x, y, top_down_map.shape, free_space_mask):
            print(f"Skipping invalid position: ({x}, {y})")
            continue

        status, error_msg = run_remote_script(client, CONFIG["conda_env"], remote_script.as_posix(), x, y, args.env)
        if status != 0:
            print(f"Remote script failed at ({x}, {y}): {error_msg}")
            continue

        with client.open_sftp() as sftp:
            fetch_remote_file(sftp, remote_image_path.as_posix(), local_image_path)

        image = np.load(local_image_path)
        if image.dtype == np.float32 or image.max() <= 1.0:
          image = (image * 255).astype(np.uint8)

# Kaydet
        img_pil = Image.fromarray(image)
        img_pil.save(os.path.join(output_dir, f"pos_{x}_{y}.png"))
        print(f"Saved image at ({x}, {y}) to {output_dir}/pos_{x}_{y}.png")

    client.close()
    os.remove(local_image_path)
    os.remove(local_map_path)


if __name__ == "__main__":
    main()
