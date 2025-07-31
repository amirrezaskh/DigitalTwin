import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pytorch_forecasting.data import MultiNormalizer, EncoderNormalizer
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import NaNLabelEncoder
from pytorch_forecasting.metrics import SMAPE
from pytorch_lightning import Trainer
from pathlib import Path

sns.set(style="whitegrid")

# ---------- Configuration ----------
clients_best_model_paths = [
    "path/to/client_0.ckpt",
    "path/to/client_1.ckpt",
    # ...
]
global_model_path = "../nodes/models/global_model.pth"

client_test_datapaths = [
    "../nodes/data/DTFL - Data/352-Table 1.csv",
    "../nodes/data/DTFL - Data/354-Table 1.csv",
    "../nodes/data/DTFL - Data/434-Table 1.csv",
    "../nodes/data/DTFL - Data/433-Table 1.csv",
    "../nodes/data/DTFL - Data/353-Table 1.csv",
    "../nodes/data/DTFL - Data/124-Table 1.csv",
    "../nodes/data/DTFL - Data/429-Table 1.csv",
    "../nodes/data/DTFL - Data/430-Table 1.csv"
]

max_encoder_length = 4 * 24
max_prediction_length = 4 * 3
TARGETS = ["temprature_value", "humidity_value", "co2_value"]
NUM_CLIENTS = len(clients_best_model_paths)
N_SAMPLES = 100  # Number of examples to plot per client
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------- Load Model + Data ----------
def load_model_and_data(model_path, test_data_path):
    model = TemporalFusionTransformer.load_from_checkpoint(model_path)
    model.to(DEVICE).eval()

    df = pd.read_csv(test_data_path)

    df["room_id"] = df["room_id"].astype(str)
    df["hours"] = df["hours"].astype(str)
    df["days"] = df["days"].astype(str)
    df["months"] = df["months"].astype(str)
    for col in TARGETS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.sort_values(["room_id", "timestamp"]).reset_index(drop=True)
    df["time_idx"] = df.groupby("room_id").cumcount()
    training_cutoff = df["time_idx"].max() - max_prediction_length

    df = df.replace([np.inf, -np.inf], np.nan)
    for col in TARGETS:
        df[col] = df[col].interpolate(method="linear")


    train_dataset = TimeSeriesDataSet(
        df[lambda x: x.time_idx <= training_cutoff],
        time_idx="time_idx",
        target=TARGETS,
        group_ids=["room_id"],
        min_encoder_length=max_encoder_length,
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        static_categoricals=["room_id"],
        static_reals=["area", "doors_area", "panels_area"],
        time_varying_known_categoricals=["hours", "days", "months"],
        time_varying_known_reals=["time_idx"],
        time_varying_unknown_reals=TARGETS,
        target_normalizer=MultiNormalizer([
            EncoderNormalizer(method='standard', center=True,
                            max_length=None, transformation=None, method_kwargs={}),
            EncoderNormalizer(method='standard', center=True,
                            max_length=None, transformation=None, method_kwargs={}),
            EncoderNormalizer(method='standard', center=True,
                            max_length=None, transformation=None, method_kwargs={})
        ]),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )

    val_dataset = TimeSeriesDataSet.from_dataset(
        train_dataset, df, predict=True, stop_randomization=True
    )
    dataloader = val_dataset.to_dataloader(
        train=False, batch_size=1280, num_workers=0
    )

    return model, dataloader


# ---------- Predict ----------
def get_predictions(model, dataloader):
    raw_predictions, x = model.predict(dataloader, mode="raw", return_x=True)
    return raw_predictions, x


# ---------- Plot ----------
def plot_actual_vs_predicted(x, raw_predictions, client_name, num_samples=N_SAMPLES):
    for target_idx, target in enumerate(TARGETS):
        plt.figure(figsize=(10, 4))
        for i in range(min(num_samples, len(raw_predictions["prediction"]))):
            actual = x["decoder_target"][i, :, target_idx].detach().cpu()
            predicted = raw_predictions["prediction"][i, :, target_idx].detach().cpu()

            time = list(range(len(actual)))
            plt.plot(time, actual, label="Actual", color="black", alpha=0.3)
            plt.plot(time, predicted, label="Predicted", color="blue", alpha=0.3)

        plt.title(f"{client_name} — {target.replace('_', ' ').title()}")
        plt.xlabel("Time steps")
        plt.ylabel(target.replace("_", " ").title())
        plt.tight_layout()
        if i == 0:
            plt.legend()
        plt.savefig(f"./plots/{client_name}_{target}.png", dpi=300)
        plt.close()


# ---------- Main ----------
Path("./figures").mkdir(parents=True, exist_ok=True)

for i in range(NUM_CLIENTS):
    model_path = clients_best_model_paths[i]
    test_data_path = client_test_datapaths[i]
    client_name = f"Client_{i+1}"

    print(f"Processing {client_name}...")
    model, dataloader = load_model_and_data(model_path, test_data_path)
    raw_preds, x = get_predictions(model, dataloader)
    plot_actual_vs_predicted(x, raw_preds, client_name)

# ---------- Global Model ----------
print("Processing Global Model...")
global_model, global_dataloader = load_model_and_data(global_model_path, client_test_datapaths[0])  # or use merged test set
raw_preds, x = get_predictions(global_model, global_dataloader)
plot_actual_vs_predicted(x, raw_preds, "Global_Model")
