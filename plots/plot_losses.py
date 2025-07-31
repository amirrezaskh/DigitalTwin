import matplotlib.pyplot as plt
from glob import glob
import json
import seaborn as sns
import pandas as pd
import os

# Set seaborn style for academic figures
sns.set(style="whitegrid", context="paper", font_scale=1.4)

# Output directory
os.makedirs("figures", exist_ok=True)

global_loss_file = "../nodes/losses/res.json"
files = glob("../nodes/losses/node_*.json")

# Room IDs
rooms = [352, 354, 434, 433, 353, 124, 429, 430]

# Metric dictionaries
temp = {}
hum = {}
co2 = {}

# Load local loss logs
for i in range(len(files)):
    with open(files[i], "r") as file:
        local_losses = json.load(file)
    node_temp = []
    node_hum = []
    node_co2 = []
    for losses in local_losses:
        node_temp.append(losses['temprature_value'])
        node_hum.append(losses['humidity_value'])
        node_co2.append(losses['co2_value'])
    temp[f"Room {rooms[i]}"] = node_temp
    hum[f"Room {rooms[i]}"] = node_hum
    co2[f"Room {rooms[i]}"] = node_co2

# Load global losses
with open(global_loss_file, "r") as file:
    global_losses = json.load(file)

global_temp = [entry['temprature_value'] for entry in global_losses]
global_hum = [entry['humidity_value'] for entry in global_losses]
global_co2 = [entry['co2_value'] for entry in global_losses]

temp["Global Model"] = global_temp
hum["Global Model"] = global_hum
co2["Global Model"] = global_co2

# Plotting functions
def plot_metric(data_dict, ylabel, title, filename):
    df = pd.DataFrame(data_dict)
    df["Round"] = df.index
    df_melt = df.melt(id_vars=["Round"], var_name="Room", value_name="Loss")

    plt.figure(figsize=(10, 5))
    sns.lineplot(data=df_melt, x="Round", y="Loss", hue="Room", marker="o", palette="tab10")
    plt.title(title, fontsize=14)
    plt.xlabel("Federated Round")
    plt.ylabel(ylabel)
    plt.legend(title="Room", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f"figures/{filename}.png", dpi=600, bbox_inches='tight')
    plt.close()

def plot_final_round_bar(data_dict, ylabel, title, filename):
    final_values = {k: v[-1] for k, v in data_dict.items()}
    df = pd.DataFrame(list(final_values.items()), columns=["Room", "Loss"])

    plt.figure(figsize=(8, 4))
    sns.barplot(data=df, x="Room", y="Loss", palette="muted")
    plt.title(title, fontsize=14)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"figures/{filename}.png", dpi=600, bbox_inches='tight')
    plt.close()

# Save plots
plot_final_round_bar(temp, "Final RMSE (°C)", "Final Temperature Error per Room", "final_temp_bar")
plot_final_round_bar(hum, "Final RMSE (%RH)", "Final Humidity Error per Room", "final_hum_bar")
plot_final_round_bar(co2, "Final RMSE (ppm)", "Final CO₂ Error per Room", "final_co2_bar")

plot_metric(temp, "RMSE (°C)", "Temperature Prediction Error per Room", "temp_line")
plot_metric(hum, "RMSE (%RH)", "Humidity Prediction Error per Room", "hum_line")
plot_metric(co2, "RMSE (ppm)", "CO₂ Prediction Error per Room", "co2_line")
