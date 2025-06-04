import os
import torch
import requests
import pandas as pd
from pathlib import Path
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer, Trainer
from pytorch_forecasting.data import NaNLabelEncoder


class Node:
    def __init__(self, port):
        self.port = port

        self.max_encoder_length = 48
        self.max_prediction_length = 1

        self.lr = 3e-4
        self.epochs = 5
        self.batch_size = 128

        self.data_path = ""

        cwd = os.path.dirname(__file__)
        model_name = f"node_{self.port-8000}"
        self.root_path = Path(__file__).resolve().parents[1]
        self.model_path = os.path.abspath(
            os.path.join(cwd, f"{self.root_path}/fl/models/{model_name}.pth")
        )

        self.get_data()
        self.model = TemporalFusionTransformer.from_dataset(
            self.train_dataset,
            learning_rate=self.lr,
            hidden_size=16,
            attention_head_size=1,
            dropout=0.1,
            loss=torch.nn.MSELoss(),
            output_size=1,
            log_interval=10,
            reduce_on_plateau_patience=4,
        )

    def get_data(self):
        df = pd.read_csv(self.data_path)
        df['time_idx'] = pd.to_datetime(df['timestamp'])
        df['time_idx'] = df['time_idx'].rank(method="dense").astype(int)

        self.train_dataset = TimeSeriesDataSet(
            df,
            time_idx="time_idx",
            target="humidity",
            group_ids=["room_id"],
            max_encoder_length=self.max_encoder_length,
            max_prediction_length=self.max_prediction_length,
            static_categoricals=["room_id"],
            static_reals=["area", "num_windows", "window_area"],
            time_varying_known_reals=["time_idx", "hour", "day", "month"],
            time_varying_unknown_reals=["humidity", "temperature", "co2", "electricity"],
            target_normalizer=NaNLabelEncoder(),
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
        )
        self.train_dataloader = self.train_dataset.to_dataloader(train=True, batch_size=self.batch_size)


    def load_model(self):
        self.model.load_state_dict(torch.load("./models/global_model.pth"))

    def train(self):
        trainer = Trainer(
            max_epochs=self.epochs,
            gpus=1 if torch.cuda.is_available() else 0
        )
        trainer.fit(
            self.model,
            train_dataloaders=self.train_dataloader,
        )
        torch.save(self.model.state_dict(), f"{self.root_path}/fl/models/node_{self.port-8000}.pth")
        requests.post("http://localhost:3000/api/model/", json={
            "id": f"model_{self.port-8000}",
            "path": self.model_path
        })