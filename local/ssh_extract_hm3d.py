import paramiko
import numpy as np
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
import os


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

    parser.add_argument(
        "--mode",
        type=str,
        default="grid",
        help="Mode of operation: 'grid' or 'random'."
    )
    return parser.parse_args()


def get_ssh_client(hostname, port, username):
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(hostname, port, username)
    return client


def fetch_remote_file(sftp, remote_path, local_path):
    sftp.get(str(remote_path), local_path)

def run_remote_script(client, conda_env, script_path, mode, env):
    conda_init = "source /home/romer/miniconda3/etc/profile.d/conda.sh"
    run_cmd = f"conda activate {conda_env} && python {script_path} --mode {mode} --env {env}"
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
        "conda_env": "habitat",
        "habitat_path": Path("/home/romer/umut/GibsonData/habitat-lab/my_scripts")
    }

    remote_script = CONFIG["habitat_path"]  / "extract.py"
    remote_map_path = CONFIG["habitat_path"] / "observations" / ("semantic_map_" + args.env + ".pkl")

    local_map_path = "./semantic_map_" + args.env + ".pkl"

    client = get_ssh_client(CONFIG["hostname"], CONFIG["port"], CONFIG["username"])

    status, error_msg = run_remote_script(client, CONFIG["conda_env"], remote_script.as_posix(), args.mode, args.env)

    if status != 0:
        print("Error running the remote script:", error_msg)
    else:       
        print("Remote script executed successfully.")

    print("Fetching remote map, this may take few minutes...")
    # Fetch the map using SFTP
    with client.open_sftp() as sftp:
        fetch_remote_file(sftp, remote_map_path.as_posix() , local_map_path)
    client.close()


if __name__ == "__main__":
    main()


