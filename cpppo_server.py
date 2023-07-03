import shlex
import subprocess

cmd = shlex.split("enip_server --print FIT201_2@22/1/3=REAL")

client = subprocess.run("echo hello!", shell=False,
                stdout=subprocess.PIPE)

