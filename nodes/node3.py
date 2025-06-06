from global_var import *

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fl.node import Node

port = 8003
data_path = "./data/rooms/F1_R4.csv"
futures = {}
executer = concurrent.futures.ThreadPoolExecutor(2)
node = Node(port, data_path)
app = Flask(__name__)


@app.route("/train/")
def train_client():
    executer.submit(node.train)
    return "The node started training!"


@app.route("/exit/")
def exit_miner():
    os.kill(os.getpid(), signal.SIGTERM)


if __name__ == '__main__':
    app.run(host="localhost", port=port, debug=True)