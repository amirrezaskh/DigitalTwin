import matplotlib.pyplot as plt
from glob import glob
import json
import seaborn as sns
import pandas as pd

sns.set(style="whitegrid", context="paper", font_scale=1.2)

global_loss_file = "../nodes/losses/res.json"
files = glob("../nodes/losses/node_*.json")

temp = {}
hum = {}
co2 = {}
for i in range(len(files)):
    file = open(files[i], "r")
    local_losses = json.loads(file.read())
    node_temp = []
    node_hum = []
    node_co2 = []
    for j in range(len(local_losses)):
        losses = local_losses[j]
        
        node_temp.append(losses['temprature_value'])
        node_hum.append(losses['humidity_value'])
        node_co2.append(losses['co2_value'])
    temp[f"Node {i}"] = node_temp
    hum[f"Node {i}"] = node_hum
    co2[f"Node {i}"] = node_co2

def plot_metric(data_dict, ylabel, title):
    df = pd.DataFrame(data_dict)
    df["Round"] = df.index
    df_melt = df.melt(id_vars=["Round"], var_name="Node", value_name="Loss")
    
    plt.figure(figsize=(10, 5))
    sns.lineplot(data=df_melt, x="Round", y="Loss", hue="Node", marker="o", palette="tab10")
    
    plt.title(title, fontsize=14)
    plt.xlabel("Federated Round")
    plt.ylabel(ylabel)
    plt.legend(title="Node", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

def plot_final_round_bar(data_dict, ylabel, title):
    final_values = {k: v[-1] for k, v in data_dict.items()}
    df = pd.DataFrame(list(final_values.items()), columns=["Node", "Loss"])
    
    plt.figure(figsize=(8, 4))
    sns.barplot(data=df, x="Node", y="Loss", palette="muted")
    plt.title(title, fontsize=14)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

plot_final_round_bar(temp, "Final RMSE (°C)", "Final Temperature Error per Node")
plot_final_round_bar(hum, "Final RMSE (%RH)", "Final Humidity Error per Node")
plot_final_round_bar(co2, "Final RMSE (ppm)", "Final CO₂ Error per Node")


# Example usage
plot_metric(temp, "RMSE (°C)", "Temperature Prediction Error per Node")
plot_metric(hum, "RMSE (%RH)", "Humidity Prediction Error per Node")
plot_metric(co2, "RMSE (ppm)", "CO2 Prediction Error per Node")

