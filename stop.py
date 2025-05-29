import requests

# Express Apps
try:
    requests.get("http://localhost:3000/exit/")
except:
    print("App1 is stopped.")