import paramiko
import numpy as np
from pathlib import Path
import pickle
from MapClass import SemanticMap
import argparse
import os
import gzip

def parse_arguments():    
    parser = argparse.ArgumentParser(
        description="SSH and run script on remote server",
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "--env",
        type=str,
        default="00800-TEEsavR23oF",
        help="Name of the scene.\nAvailable options: " + ", ".join(["00800-TEEsavR23oF", "00802-wcojb4TFT35",
                                                                     "00803-k1cupFYWXJ6", "00808-y9hTuugGdiq"])
    )
    return parser.parse_args()


def get_ssh_client(hostname, port, username):
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(hostname, port, username)
    return client

def send_local_file(sftp, local_path, remote_path):
    print(f"Sending {local_path} to {str(remote_path)}")
    sftp.put(local_path, str(remote_path))

def fetch_remote_file(sftp, remote_path, local_path):
    sftp.get(str(remote_path), local_path)

def run_remote_script(client, conda_env, script_path):
    conda_init = "source /home/romer/miniconda3/etc/profile.d/conda.sh"
    run_cmd = f"conda activate {conda_env} && python {script_path}"
    full_cmd = f"bash -c '{conda_init} && {run_cmd}'"
    stdin, stdout, stderr = client.exec_command(full_cmd)
    exit_status = stdout.channel.recv_exit_status()
    return exit_status, stderr.read().decode()


def main():
    args = parse_arguments()
    # Configuration
    CONFIG = {
        "hostname": "144.122.128.224",
        "port": 22,
        "username": "romer",
        "conda_env": "oneformer",
        "oneformer_path": Path("/home/romer/umut/segmentation/OneFormer"),
    }
    local_object_path = Path(f"./semantic_map_{args.env}.pkl")
    remote_script = CONFIG["oneformer_path"]  / "oneformer_infere.py"
    local_file_path = Path("./rgb_observations.pkl.gz")
    # convert the local_file_path to full path
    local_file_path = local_file_path.resolve()
    print(f"Local file path: {local_file_path}")
    remote_file_path = CONFIG["oneformer_path"] / "rgb_observations.pkl.gz"
    local_predictions_path = Path("./predictions.pkl.gz")
    remote_predictions_path = CONFIG["oneformer_path"] / "outputs/predictions.pkl.gz"

    with open(local_object_path, "rb") as f:
        semantic_map = pickle.load(f)

    with gzip.open(local_file_path, "wb") as f:
        pickle.dump(semantic_map.rgb_observations, f)

    # Wait for the file to be created
    while not local_file_path.exists():
        pass
    print("Local file created.")

    client = get_ssh_client(CONFIG["hostname"], CONFIG["port"], CONFIG["username"])

    # Send the local file to the remote server
    with client.open_sftp() as sftp:
        send_local_file(sftp, local_file_path, remote_file_path.as_posix())
    print("Local file sent to remote server.")

    print("Running remote script, this may take few minutes...")
    status, error_msg = run_remote_script(client, CONFIG["conda_env"], remote_script.as_posix())

    if status != 0:
        print("Error running the remote script:", error_msg)
    else:       
        print("Remote script executed successfully.")

    print("Fetching remote map, this may take few minutes...")
    # Fetch the map using SFTP
    with client.open_sftp() as sftp:
        fetch_remote_file(sftp, remote_predictions_path.as_posix(), local_predictions_path.as_posix()) 

    # Remove the local file after fetching the predictions
    os.remove(local_file_path)
    # Remove the remote file after fetching the predictions
    with client.open_sftp() as sftp:
        sftp.remove(remote_file_path.as_posix())
    
    with gzip.open(local_predictions_path, "rb") as f:
        predictions = pickle.load(f)
    semantic_map.semantic_predictions = predictions

    with gzip.open(local_object_path, "wb") as f:
        pickle.dump(semantic_map, f)

    client.close()

if __name__ == "__main__":
    main()
