import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from io import StringIO
import os
import time

# Find latest file
path = '/datasets'

files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]

latest_file = None
latest_time = 0

for file in files:
    file_path = os.path.join(path, file)
    file_mod_time = os.path.getmtime(file_path)  # last modified time
    if file_mod_time > latest_time:
        latest_time = file_mod_time
        latest_file = file

# If file is not found, data logger is broken so exit
if not latest_file:
    print("No files found in the directory.")
    exit(1)
else: latest_file_path = os.path.join(path, latest_file)

while True:

    # Collect latest changes to dataset
    with open(latest_file_path, 'r') as f:
        data = f.read()
        f.close()

    # Parse data
    df = pd.read_csv(StringIO(data), parse_dates=["time"])

    # Format time to display 70 latest seconds
    latest_time = df["time"].max()
    time_threshold = latest_time - pd.Timedelta(seconds=70)
    df_recent = df[df["time"] >= time_threshold].copy()
    df_recent["seconds"] = (df_recent["time"] - df_recent["time"].min()).dt.total_seconds().astype(int)
    all_seconds = list(range(71))
    df_recent = df_recent.set_index("seconds").reindex(all_seconds).reset_index()
    df_recent.rename(columns={"index": "seconds"}, inplace=True)

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(df_recent["seconds"], df_recent["LIT101"], label="T101 (LIT101)", color="green", linestyle="-")
    plt.plot(df_recent["seconds"], df_recent["LIT301"], label="T301 (LIT301)", color="blue", linestyle="-")

    # Format plot
    plt.xlabel("Seconds")
    plt.ylabel("Water level")
    plt.title("T101 and T301 Water level")
    plt.xlim(0, 70)  # Fixed range 0 to 70 seconds
    plt.ylim(0.0, 1.2)
    plt.yticks(np.arange(0.0, 1.3, 0.2), labels=[f"{x:.1f}" for x in np.arange(0.0, 1.3, 0.2)])  # Y-axis labels rounded
    plt.xticks(np.arange(0, 71, 10))  # X-axis ticks every 10 seconds
    plt.legend()
    plt.grid(True)
    plt.show()

    time.sleep(1)
