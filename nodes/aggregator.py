from global_var import *


class Aggregator:
    def __init__(self):
        self.data_path = "./data/rooms/F4_R19.csv"
        self.losses = []
        self.root_path = Path(__file__).resolve().parents[1]

        self.lr = 3e-4
        self.batch_size = 1280
        self.max_encoder_length = 23*24
        self.max_prediction_length = 24

        self.get_data()

    def get_data(self):
        df = pd.read_csv(self.data_path)
        df["hour"] = df["hour"].astype(str)
        df["day"] = df["day"].astype(str)
        df["month"] = df["month"].astype(str)
        df = df.sort_values(["room_id", "timestamp"]).reset_index(drop=True)
        df["time_idx"] = df.groupby("room_id").cumcount()

        self.test_dataset = TimeSeriesDataSet(
            df,
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
        
        self.test_dataloader = self.test_dataset.to_dataloader(train=False, batch_size=self.batch_size)

    def evaluate(self):
        warnings.filterwarnings(
            "ignore", message=".*does not have valid feature names.*")
        predictions = self.model.predict(self.test_dataloader, return_y=True)
        targets = ["humidity", "temperature", "co2", "electricity"]
        outputs = predictions.output
        y = predictions.y
        mse = {}
        for i in range(len(targets)):
            mse[targets[i]] = MAE()(outputs[i], y[0][i])
        self.losses.append(mse)
    
    def aggregate(self, model_blocks):
        models = [TemporalFusionTransformer.load_from_checkpoint(model_block["path"])
                for model_block in model_blocks]
        self.global_model = {}
        for key in models[0].state_dict().keys():
            self.global_model[key] = sum(model.state_dict()[key] for model in models) / len(models)
        torch.save(self.global_model, "./models/global_model.pth")
        self.model = copy.deepcopy(models[0])
        self.model.load_state_dict(self.global_model)


port = 5050
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
    model_blocks = request.get_json()["models"]
    aggregator.aggregate(model_blocks)
    aggregator.evaluate()
    return "Aggregation completed."

if __name__ == '__main__':
    app.run(host="localhost", port=port, debug=True)