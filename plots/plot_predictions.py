from typing import Any, Dict, Optional, Union
import torch
import numpy as np
import pandas as pd
import matplotlib
from torchmetrics import Metric
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pytorch_forecasting.data import MultiNormalizer, EncoderNormalizer
from pytorch_forecasting import MASE, TemporalFusionTransformer, TimeSeriesDataSet, to_list
from pytorch_lightning import Trainer
from pathlib import Path

sns.set(style="whitegrid")
sns.set_context("paper", font_scale=1.2)          # Larger font for academic papers
sns.set_style("whitegrid", {'axes.grid': True})   # Grid with clean background
sns.set_palette("colorblind")                     # Colorblind-friendly colors

plt.rcParams["figure.dpi"] = 300                  # Higher resolution for display
plt.rcParams["savefig.dpi"] = 600                 # High resolution for saving
# plt.rcParams["font.family"] = "serif"             # Serif font for academic look
plt.rcParams["axes.titlesize"] = 10
plt.rcParams["axes.labelsize"] = 10
plt.rcParams["legend.fontsize"] = 10
plt.rcParams["lines.linewidth"] = 1.3
plt.rcParams["lines.markersize"] = 6
plt.rcParams["grid.alpha"] = 0.3


# ---------- Configuration ----------
clients_best_model_paths = [
    "../nodes/models/node_0/lightning_logs/version_80/checkpoints/epoch=19-step=2960.ckpt",
    "../nodes/models/node_1/lightning_logs/version_76/checkpoints/epoch=10-step=1628.ckpt",
    "../nodes/models/node_2/lightning_logs/version_80/checkpoints/epoch=10-step=1628.ckpt",
    "../nodes/models/node_3/lightning_logs/version_80/checkpoints/epoch=12-step=1924.ckpt",
    "../nodes/models/node_4/lightning_logs/version_77/checkpoints/epoch=19-step=2960.ckpt",
    "../nodes/models/node_5/lightning_logs/version_77/checkpoints/epoch=11-step=1776.ckpt",
    "../nodes/models/node_6/lightning_logs/version_77/checkpoints/epoch=19-step=2960.ckpt",
    "../nodes/models/node_7/lightning_logs/version_77/checkpoints/epoch=19-step=2960.ckpt"
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

rooms = [352, 354, 434, 433, 353, 124, 429, 430]
max_encoder_length = 4 * 24
max_prediction_length = 4 * 3
TARGETS = ["temprature_value", "humidity_value", "co2_value"]
target_names =["Temperature (°C)", "Humidity (%RH)", "CO₂ (ppm)"]
NUM_CLIENTS = len(clients_best_model_paths)
N_SAMPLES = 1  # number of samples to plot per client per target
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
            EncoderNormalizer(method='standard', center=True),
            EncoderNormalizer(method='standard', center=True),
            EncoderNormalizer(method='standard', center=True)
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
    predictions = model.predict(dataloader, mode="raw", return_x=True)
    return predictions.output, predictions.x

# ---------- Plot using pytorch_forecasting's plot_prediction ----------
def plot_samples_with_builtin(model, predictions, target_idx, target_name, x, client_name):
    title_prefix = f"{client_name}"

    figs = plot_prediction(model, x, predictions, idx=0, target_idx=target_idx, add_loss_to_title=False)
    figs[0].set_size_inches(8, 4)
    ax = figs[0].axes[0]
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.set_title(f"{title_prefix}", fontsize=14)
    ax.set_xlabel("Time Index", fontsize=12)
    ax.set_ylabel(target_names[target_idx], fontsize=12)

    plt.tight_layout() # Removes existing legend
    ax.legend(by_label.values(), by_label.keys(), fontsize=10)
    plt.savefig(f"./figures/{client_name} - {target_name}.png",
                dpi=600, bbox_inches='tight', transparent=False)
    plt.close(figs[0])
        
def plot_prediction(
        model,
        x: dict[str, torch.Tensor],
        out: dict[str, torch.Tensor],
        idx: int = 0,
        target_idx: int = 0,
        add_loss_to_title: Union[Metric, torch.Tensor, bool] = False,
        show_future_observed: bool = True,
        ax=None,
        quantiles_kwargs: Optional[dict[str, Any]] = None,
        prediction_kwargs: Optional[dict[str, Any]] = None,
    ):
    """
    Plot prediction of prediction vs actuals

    Args:
        x: network input
        out: network output
        idx: index of prediction to plot
        add_loss_to_title: if to add loss to title or loss function to calculate. Can be either metrics,
            bool indicating if to use loss metric or tensor which contains losses for all samples.
            Calcualted losses are determined without weights. Default to False.
        show_future_observed: if to show actuals for future. Defaults to True.
        ax: matplotlib axes to plot on
        quantiles_kwargs (Dict[str, Any]): parameters for ``to_quantiles()`` of the loss metric.
        prediction_kwargs (Dict[str, Any]): parameters for ``to_prediction()`` of the loss metric.

    Returns:
        matplotlib figure
    """  # noqa: E501
    if quantiles_kwargs is None:
        quantiles_kwargs = {}
    if prediction_kwargs is None:
        prediction_kwargs = {}

    from matplotlib import pyplot as plt

    # all true values for y of the first sample in batch
    encoder_targets = to_list(x["encoder_target"])
    decoder_targets = to_list(x["decoder_target"])

    y_raws = to_list(
        out["prediction"]
    )  # raw predictions - used for calculating loss
    y_hats = to_list(model.to_prediction(out, **prediction_kwargs))
    y_quantiles = to_list(model.to_quantiles(out, **quantiles_kwargs))

    # for each target, plot
    figs = []
    # for y_raw, y_hat, y_quantile, encoder_target, decoder_target in zip(
    #     y_raws, y_hats, y_quantiles, encoder_targets, decoder_targets
    # ):
    y_raw = y_raws[target_idx]
    y_hat = y_hats[target_idx]
    y_quantile = y_quantiles[target_idx]
    encoder_target = encoder_targets[target_idx]
    decoder_target = decoder_targets[target_idx]


    y_all = torch.cat([encoder_target[idx], decoder_target[idx]])
    max_encoder_length = x["encoder_lengths"].max()
    y = torch.cat(
        (
            y_all[: x["encoder_lengths"][idx]],
            y_all[
                max_encoder_length : (
                    max_encoder_length + x["decoder_lengths"][idx]
                )
            ],
        ),
    )
    # move predictions to cpu
    y_hat = y_hat.detach().cpu()[idx, : x["decoder_lengths"][idx]]
    y_quantile = y_quantile.detach().cpu()[idx, : x["decoder_lengths"][idx]]
    y_raw = y_raw.detach().cpu()[idx, : x["decoder_lengths"][idx]]

    # move to cpu
    y = y.detach().cpu()
    # create figure
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
    n_pred = y_hat.shape[0]
    x_obs = np.arange(-(y.shape[0] - n_pred), 0)
    x_pred = np.arange(n_pred)
    prop_cycle = iter(plt.rcParams["axes.prop_cycle"])
    obs_color = next(prop_cycle)["color"]
    pred_color = next(prop_cycle)["color"]
    # plot observed history
    if len(x_obs) > 0:
        if len(x_obs) > 1:
            plotter = ax.plot
        else:
            plotter = ax.scatter
        plotter(x_obs, y[:-n_pred], label="observed", c=obs_color)
    if len(x_pred) > 1:
        plotter = ax.plot
    else:
        plotter = ax.scatter

    # plot observed prediction
    if show_future_observed:
        plotter(x_pred, y[-n_pred:], label=None, c=obs_color)

    # plot prediction
    plotter(x_pred, y_hat, label="predicted", c=pred_color)

    # plot predicted quantiles
    plotter(
        x_pred,
        y_quantile[:, y_quantile.shape[1] // 2],
        c=pred_color,
        alpha=0.15,
    )
    for i in range(y_quantile.shape[1] // 2):
        if len(x_pred) > 1:
            ax.fill_between(
                x_pred,
                y_quantile[:, i],
                y_quantile[:, -i - 1],
                alpha=0.15,
                fc=pred_color,
            )
        else:
            quantiles = torch.tensor(
                [[y_quantile[0, i]], [y_quantile[0, -i - 1]]]
            )
            ax.errorbar(
                x_pred,
                y[[-n_pred]],
                yerr=np.absolute(quantiles - y[-n_pred]),
                c=pred_color,
                capsize=1.0,
            )

    
    ax.set_xlabel("Time index")
    # fig.legend()
    figs.append(fig)

    # return multiple of target is a list, otherwise return single figure
    if isinstance(x["encoder_target"], (tuple, list)):
        return figs
    else:
        return fig


# ---------- Main ----------
Path("./figures").mkdir(parents=True, exist_ok=True)

for i in range(NUM_CLIENTS):
    model_path = clients_best_model_paths[i]
    test_data_path = client_test_datapaths[i]
    client_name = f"Room {rooms[i]}"

    print(f"Processing {client_name}...")
    model, dataloader = load_model_and_data(model_path, test_data_path)
    raw_preds, x = get_predictions(model, dataloader)

    for target_idx, target_name in enumerate(TARGETS):
        plot_samples_with_builtin(model, raw_preds, target_idx, target_name, x, client_name)

# ---------- Global Model ----------
print("Processing Global Model...")
global_model, global_dataloader = load_model_and_data(global_model_path, client_test_datapaths[0])
raw_preds, x = get_predictions(global_model, global_dataloader)
for target_idx, target in enumerate(TARGETS):
    plot_samples_with_builtin(raw_preds, x, target_idx, target, "Global_Model", num_samples=N_SAMPLES)
