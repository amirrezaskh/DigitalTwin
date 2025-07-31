from global_var import *


lr = 3e-4
data_path = "./data/DTFL - Data/227-Table 1.csv"
max_encoder_length = 4 * 24
max_prediction_length = 4 * 3

df = pd.read_csv(data_path)

df["room_id"] = df["room_id"].astype(str)
df["hours"] = df["hours"].astype(str)
df["days"] = df["days"].astype(str)
df["months"] = df["months"].astype(str)
for col in ["temprature_value", "humidity_value", "co2_value"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")
df = df.sort_values(["room_id", "timestamp"]).reset_index(drop=True)
df["time_idx"] = df.groupby("room_id").cumcount()
training_cutoff = df["time_idx"].max() - max_prediction_length

df = df.replace([np.inf, -np.inf], np.nan)
for col in ["temprature_value", "humidity_value", "co2_value"]:
    df[col] = df[col].interpolate(method="linear")


train_dataset = TimeSeriesDataSet(
    df[lambda x: x.time_idx <= training_cutoff],
    time_idx="time_idx",
    target=["temprature_value", "humidity_value", "co2_value"],
    group_ids=["room_id"],
    min_encoder_length=max_encoder_length,
    max_encoder_length=max_encoder_length,
    max_prediction_length=max_prediction_length,
    static_categoricals=["room_id"],
    static_reals=["area", "doors_area", "panels_area"],
    time_varying_known_categoricals=["hours", "days", "months"],
    time_varying_known_reals=["time_idx"],
    time_varying_unknown_reals=[
        "temprature_value", "humidity_value", "co2_value"],
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

model = TemporalFusionTransformer.from_dataset(
    train_dataset,
    learning_rate=lr,
    hidden_size=32,
    attention_head_size=4,
    dropout=0.2,
    loss=MultiLoss([RMSE(), RMSE(), RMSE()]),
    output_size=[1, 1, 1],
    log_interval=10,
    reduce_on_plateau_patience=4,
)

torch.save(model.state_dict(), "./models/global_model.pth")
