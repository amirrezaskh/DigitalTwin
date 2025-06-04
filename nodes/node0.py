from global_var import *

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fl.node import Node


port = 8000
futures = {}
executer = concurrent.futures.ThreadPoolExecutor(2)
node = Node()
app = Flask(__name__)


@app.route("/client/train/")
def train_client():
    node.train()
    # executer.submit(node.train)
    return "The node started training!"


@app.route("/exit/")
def exit_miner():
    os.kill(os.getpid(), signal.SIGTERM)


if __name__ == '__main__':
    app.run(host="localhost", port=port, debug=True)