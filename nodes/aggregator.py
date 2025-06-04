from global_var import *


class Aggregator:
    def __init__(self):
        self.data_path = ""
        self.losses = []
        self.root_path = Path(__file__).resolve().parents[1]

        self.lr = 3e-4
        self.batch_size = 128
        self.max_encoder_length = 48
        self.max_prediction_length = 1

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

        self.test_dataset = TimeSeriesDataSet(
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

        self.test_dataloader = self.test_dataset.to_dataloader(train=False, batch_size=self.batch_size)

    def evaluate(self):
        predictions = self.model.predict(self.test_dataloader)
        actuals = torch.cat([y for _, y in iter(self.test_dataloader)])
        predictions = self.model.predict(self.test_dataloader)
        mse = mean_squared_error(actuals.cpu(), predictions.cpu())
        self.losses.append(mse)
    
    def aggregate(self, node_names):
        models = [torch.load(f"{root_path}/fl/models/{node_name}.pth")
                for node_name in node_names]
        self.global_model = {}
        for key in models[0].keys():
            self.global_model[key] = sum(model[key] for model in models) / len(models)
        torch.save(self.global_model, "./models/global_model.pth")
        self.model.load_state_dict(self.global_model)


executer = concurrent.futures.ThreadPoolExecutor(2)
root_path = Path(__file__).resolve().parents[1]
app = Flask(__name__)
aggregator = Aggregator()


@app.route("/losses/")
def save_losses():
    file = open(aggregator.losses_file)
    file.write(json.dumps(aggregator.losses))
    return "Done"

@app.route("/aggregate/", methods=['POST'])
def aggregate():
    nodes = request.get_json()["nodes"]
    aggregator.aggregate(nodes)
    aggregator.evaluate()
    return "Aggregation completed."