import requests

try:
    requests.get("http://localhost:3000/exit/")
except:
    print("Express app is stopped.")

try:
    requests.get("http://localhost:8080/exit/")
except:
    print("Aggregator is stopped.")

# Nodes
for i in range(4):
    try:
        requests.get(f"http://localhost:{8000 + i}/exit/")
    except:
        print(f"Node{i} is stopped.")