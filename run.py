import os
import time
import subprocess
import requests


if __name__ == "__main__":
    num_nodes = 4
    cwd = os.path.dirname(__file__)

    print("Bringing up the network...")
    os.chdir(os.path.join(cwd, "test-network"))
    os.system("sh ./start.sh")

    print("Bringing up the express applications...")
    os.chdir(os.path.join(cwd, "express-application"))
    with open("../logs/app1.txt", "w") as f:
        subprocess.Popen(
            ["node", f"./app1.js"],
            stdout=f,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            preexec_fn=os.setsid
        )
    time.sleep(1)

    os.chdir(os.path.join(cwd, "test-network"))
    os.system("sh ./req.sh")