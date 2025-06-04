from global_var import *


lr = 3e-4
data_path = ""
max_encoder_length = 48
max_prediction_length = 1

df = pd.read_csv(data_path)
df['time_idx'] = pd.to_datetime(df['timestamp'])
df['time_idx'] = df['time_idx'].rank(method="dense").astype(int)


train_dataset = TimeSeriesDataSet(
    df,
    time_idx="time_idx",
    target="humidity",
    group_ids=["room_id"],
    max_encoder_length=max_encoder_length,
    max_prediction_length=max_prediction_length,
    static_categoricals=["room_id"],
    static_reals=["area", "num_windows", "window_area"],
    time_varying_known_reals=["time_idx", "hour", "day", "month"],
    time_varying_unknown_reals=["humidity", "temperature", "co2", "electricity"],
    target_normalizer=NaNLabelEncoder(),
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
    output_size=1,
    log_interval=10,
    reduce_on_plateau_patience=4,
)

torch.save(model.state_dict(), "./models/global_model.pth")