from global_var import *


class Aggregator:
    def __init__(self):
        self.data_path = "./data/DTFL - Data/227-Table 1.csv"
        self.targets = ["temprature_value", "humidity_value", "co2_value"]
        self.losses = []
        self.losses_file = "./losses/res.json"
        self.root_path = Path(__file__).resolve().parents[1]

        self.lr = 3e-4
        self.batch_size = 1280
        self.max_encoder_length = 4 * 10
        self.max_prediction_length = 4 * 2

        self.get_data()

    def get_data(self):
        df = pd.read_csv(self.data_path)

        df["room_id"] = df["room_id"].astype(str)
        df["hours"] = df["hours"].astype(str)
        df["days"] = df["days"].astype(str)
        df["months"] = df["months"].astype(str)
        for col in self.targets:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.sort_values(["room_id", "timestamp"]).reset_index(drop=True)
        df["time_idx"] = df.groupby("room_id").cumcount()

        df = df.replace([np.inf, -np.inf], np.nan)
        for col in self.targets:
            df[col] = df[col].interpolate(method="linear")

        self.test_dataset = TimeSeriesDataSet(
            df,
            time_idx="time_idx",
            target=self.targets,
            group_ids=["room_id"],
            min_encoder_length=self.max_encoder_length,
            max_encoder_length=self.max_encoder_length,
            max_prediction_length=self.max_prediction_length,
            static_categoricals=["room_id"],
            static_reals=["area", "doors_area", "panels_area"],
            time_varying_known_categoricals=["hours", "days", "months"],
            time_varying_known_reals=["time_idx"],
            time_varying_unknown_reals=self.targets,
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

        self.test_dataloader = self.test_dataset.to_dataloader(
            train=False, batch_size=len(df), drop_last=True)

    def evaluate(self):
        warnings.filterwarnings(
            "ignore", message=".*does not have valid feature names.*")
        predictions = self.model.predict(self.test_dataloader, return_y=True)
        outputs = predictions.output
        y = predictions.y
        mse = {}
        for i in range(len(self.targets)):
            mse[self.targets[i]] = MAE()(outputs[i], y[0][i]).item()
        self.losses.append(mse)

    def aggregate(self, model_blocks):
        models = [TemporalFusionTransformer.load_from_checkpoint(model_block["path"])
                  for model_block in model_blocks]
        self.global_model = {}
        for key in models[0].state_dict().keys():
            self.global_model[key] = sum(
                model.state_dict()[key] for model in models) / len(models)
        torch.save(self.global_model, "./models/global_model.pth")
        self.model = copy.deepcopy(models[0])
        self.model.load_state_dict(self.global_model)


port = 8080
executer = concurrent.futures.ThreadPoolExecutor(2)
root_path = Path(__file__).resolve().parents[1]
app = Flask(__name__)
aggregator = Aggregator()


@app.route("/losses/")
def save_losses():
    file = open(aggregator.losses_file, "w")
    file.write(json.dumps(aggregator.losses))
    return "Done"


@app.route("/aggregate/", methods=['POST'])
def aggregate():
    model_blocks = request.get_json()["models"]
    aggregator.aggregate(model_blocks)
    aggregator.evaluate()
    return "Aggregation completed."


@app.route("/exit/")
def exit_miner():
    os.kill(os.getpid(), signal.SIGTERM)


if __name__ == '__main__':
    app.run(host="localhost", port=port, debug=True)
