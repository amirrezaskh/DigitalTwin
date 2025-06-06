from global_var import *


lr = 3e-4
data_path = "./data/rooms/F1_R1.csv"
max_encoder_length = 23 * 24
max_prediction_length = 24

df = pd.read_csv(data_path)
df["hour"] = df["hour"].astype(str)
df["day"] = df["day"].astype(str)
df["month"] = df["month"].astype(str)
df = df.sort_values(["room_id", "timestamp"]).reset_index(drop=True)
df["time_idx"] = df.groupby("room_id").cumcount()
training_cutoff = df["time_idx"].max() - max_prediction_length


train_dataset = TimeSeriesDataSet(
    df[lambda x: x.time_idx <= training_cutoff],
    time_idx="time_idx",
    target=["humidity", "temperature", "co2", "electricity"],
    group_ids=["room_id"],
    min_encoder_length=max_encoder_length,
    max_encoder_length=max_encoder_length,
    max_prediction_length=max_prediction_length,
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

model = TemporalFusionTransformer.from_dataset(
    train_dataset,
    learning_rate=lr,
    hidden_size=16,
    attention_head_size=1,
    dropout=0.1,
    loss=torch.nn.MSELoss(),
    output_size=[1, 1, 1, 1],
    log_interval=10,
    reduce_on_plateau_patience=4,
)

torch.save(model.state_dict(), "./models/global_model.pth")