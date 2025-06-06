import os
import torch
import warnings
import requests
import matplotlib
matplotlib.use("Agg")
import pandas as pd
from pathlib import Path
import lightning.pytorch as pl
from lightning.pytorch.tuner import Tuner
from lightning.pytorch.loggers import TensorBoardLogger
from pytorch_forecasting.data import MultiNormalizer, EncoderNormalizer
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer


class Node:
    def __init__(self, port, data_path):
        self.port = port

        self.max_encoder_length = 23 * 24
        self.max_prediction_length = 24

        self.lr = 3e-4
        self.epochs = 5
        self.batch_size = 128

        self.data_path = data_path

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
            output_size=[1, 1, 1, 1],
            log_interval=10,
            reduce_on_plateau_patience=4,
        )

    def get_data(self):
        df = pd.read_csv(self.data_path)
        df["hour"] = df["hour"].astype(str)
        df["day"] = df["day"].astype(str)
        df["month"] = df["month"].astype(str)
        df = df.sort_values(["room_id", "timestamp"]).reset_index(drop=True)
        df["time_idx"] = df.groupby("room_id").cumcount()
        training_cutoff = df["time_idx"].max() - self.max_prediction_length

        self.train_dataset = TimeSeriesDataSet(
            df[lambda x: x.time_idx <= training_cutoff],
            time_idx="time_idx",
            target=["humidity", "temperature", "co2", "electricity"],
            group_ids=["room_id"],
            min_encoder_length=self.max_encoder_length,
            max_encoder_length=self.max_encoder_length,
            max_prediction_length=self.max_prediction_length,
            static_categoricals=["room_id"],
            static_reals=["area", "num_windows", "window_area"],
            time_varying_known_categoricals=["hour", "day", "month"],
            time_varying_known_reals=["time_idx"],
            time_varying_unknown_reals=["humidity",
                                        "temperature", "co2", "electricity"],
            target_normalizer=MultiNormalizer([
                EncoderNormalizer(method='standard', center=True,
                                  max_length=None, transformation=None, method_kwargs={}),
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
        self.train_dataloader = self.train_dataset.to_dataloader(
            train=True, batch_size=self.batch_size)

        self.val_dataset = TimeSeriesDataSet.from_dataset(
            self.train_dataset, df, predict=True, stop_randomization=True
        )
        self.val_dataloader = self.val_dataset.to_dataloader(
            train=False, batch_size=self.batch_size * 10, num_workers=0
        )

    def load_model(self):
        self.model.load_state_dict(torch.load("./models/global_model.pth"))

    def train(self):
        warnings.filterwarnings(
            "ignore", message=".*does not have valid feature names.*")
        
        self.load_model()
        early_stop_callback = EarlyStopping(
            monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min"
        )
        lr_logger = LearningRateMonitor()
        logger = TensorBoardLogger(f"models/node {self.port-8000}/")

        trainer = pl.Trainer(
            max_epochs=self.epochs,
            accelerator="auto",
            devices="auto",
            enable_model_summary=True,
            gradient_clip_val=0.1,
            limit_train_batches=50,
            callbacks=[lr_logger, early_stop_callback],
            logger=logger,
        )

        Tuner(trainer).lr_find(
            self.model,
            train_dataloaders=self.train_dataloader,
            val_dataloaders=self.val_dataloader,
            max_lr=10.0,
            min_lr=1e-6,
        )

        trainer.fit(
            self.model,
            train_dataloaders=self.train_dataloader,
            val_dataloaders=self.val_dataloader
        )

        best_model_path = trainer.checkpoint_callback.best_model_path
        # torch.save(self.model.state_dict(), f"{self.root_path}/fl/models/node_{self.port-8000}.pth")
        requests.post("http://localhost:3000/api/model/", json={
            "id": f"model_{self.port-8000}",
            "path": best_model_path
        })
