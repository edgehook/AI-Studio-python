import threading
import queue
import requests
import time
import os
import json
 
data_queue = queue.Queue()

notify_url = "http://172.21.73.126:8080"
if os.getenv("NOTIFY_URL") is not None:
    notify_url = os.getenv("NOTIFY_URL")
 
def notify():
    notify_api = f"{notify_url}/v1/aiStudio/notify"
    while True:
        msg = data_queue.get()
        if msg is None:
            time.sleep(10)
            continue
        try:
            # print(f"notify: {notify_url}, data: {["cmsgontent"]}")
            data = json.dumps(msg)
            response = requests.post(notify_api, json=msg)
            response.raise_for_status()
            print(f"Status Code: {response.status_code}")
            print(f"Response JSON: {response.json()}")
            print(f"Successfully sent data")
        except requests.exceptions.RequestException as e:
            print(f"Failed to send, Error: {e}")
        data_queue.task_done()

worker = threading.Thread(target=notify)
worker.daemon = True 
worker.start()
 
def add_notify_msg(data):
    data_queue.put(data)
 