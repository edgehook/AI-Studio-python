
import queue
import threading
from websockets.sync.server import serve
 
message_queue = queue.Queue()
 
websocket_map = {}
websocket_map_normal_key = "normal"

def push_msg(id, msg):
    message_queue.put({"id": id, "msg": msg})

def websocket_handler(websocket):
    path = websocket.request.path
    if len(path) > 1:
        path = path[1:]
    else:
        path = websocket_map_normal_key
    if path not in websocket_map:
        websocket_map[path] = set()
   
    websocket_map[path].add(websocket)
    try:
        message = websocket.recv()
        print(message)
    except Exception:
        print(f"websocket closed path: {path}")
 
def broadcast_messages():
    while True:
        try:
            message = message_queue.get(timeout=5)
            # print(message)
            if message is None:
                break
            if("id" in message):
                id = message.get("id", None)
                msg = message.get("msg", "")
                if id is not None:
                    websocket_connections = websocket_map.get(id, None) 
                    if websocket_connections:
                        for ws in list(websocket_connections):  
                            try:
                                ws.send(msg)
                            except Exception as e:
                                print(f"Failed to send message to {ws}: {e}")
                                websocket_connections.remove(ws)
        except queue.Empty:
            continue
        except Exception:
            continue

def start_server():
    port = 8765
    with serve(websocket_handler, "0.0.0.0", port, close_timeout=60) as server:
        print(f"start websocket server successful :{port}")
        server.serve_forever()


def run_server():
    websocket_thread = threading.Thread(target=start_server)
    websocket_thread.setDaemon(True)
    websocket_thread.start()
    broadcast_thread = threading.Thread(target=broadcast_messages) 
    broadcast_thread.daemon = True 
    broadcast_thread.start() 
 
