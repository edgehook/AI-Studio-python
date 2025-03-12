import os
import random
import string
from flask import Blueprint, jsonify, request, Flask
import torch
from apiServer.detection import detect
from apiServer.utils import responce
from flask_cors import CORS
bp = Blueprint('main', __name__)
app = Flask(__name__)
cors = CORS(app, resources={
    "/v1/*": {
        "origins": "*",
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})
@bp.route('/v1/detect', methods=["POST"])
def run_detect():
    json = request.get_json()
    weight = json["weight"]
    source = json["source"]
    id = json["id"]
    project = json["project"]
    labels = json["labels"]
    name = json["name"]
    thres = json["thres"]
    tp= json["tp"]
    duration = json["duration"]
    print(f"weight:{weight}, source:{source}, id: {id}, project:{project}, labels:{labels}, thres: {thres}, type: {tp}, duration: {duration}")
    detect_region = json["detectPoints"]
    if not detect.camera_is_available(source=source, tp=tp):
        result = responce.result(500, "error", "Camera open error")
        return jsonify(result), 500

    dt = detect.create_detection(weights=weight, source = source, project = project, labels = labels, name = name, detect_id= id, detect_region= detect_region, thres=thres)
    dt.start_detect(tp, duration=duration)
    result = responce.result(200, "success")
    return jsonify(result), 200

@bp.route('/v1/detect/stop', methods=["POST"])
def stop_detect():
    json = request.get_json()
    id = json["id"]
    dt = detect.get_detection(detect_id=id)
    if dt is not None:
        dt.stop_detect()
    else:
        print(f"detect is None, id={id}")
    
    result = responce.result(200, "success")
    return jsonify(result), 200
@bp.route('/v1/detect/monitor', methods=["POST"])
def get_detect_monitor():
    json = request.get_json()
    detectIds = json["detectIds"]
    
    monitor_array = []
    if detectIds is not None:
        try: 
            # detectIds = json.loads(ids)
            for detectId in detectIds:
                monitor_map = {}
                dt = detect.get_detection(detect_id=detectId)
                if dt is not None:
                    detect_time = dt.detect_time
                    handle_time = dt.handle_time
                    monitor_map["detectId"] = detectId
                    monitor_map["detectTime"] = detect_time
                    monitor_map["handleTime"] = handle_time
                    monitor_map["name"] = dt.name
                    monitor_array.append(monitor_map)
        except json.JSONDecodeError as e:
            result = responce.result(500, "error", "JSON error")
            return jsonify(result), 500
    result = responce.result(200, "success",monitor_array)
    return jsonify(result), 200
@bp.route('/v1/detect/cuda', methods=["GET"])
def get_cuda_version():
    cuda_version = ""
    if torch.cuda.is_available():
        cuda_version = torch.version.cuda
    else:
        print("No available CUDA device was detected。")
    result = responce.result(200, "success",cuda_version)
    return jsonify(result), 200


@bp.route('/v1/camera/screen', methods=["GET"])
def get_camera_screen():
    source = request.args.get("source")
    tp = request.args.get("type")
    base64Img = detect.get_camera_screen(source, tp= tp)
    result = responce.result(200, "success", base64Img)
    return jsonify(result), 200

@bp.route('/v1/camera/video/screen', methods=["POST"])
def get_camera_video_screen():
    if 'video' not in request.files:
        return "No file part", 400

    file = request.files['video']

    if file.filename == '':
        return "No selected file", 400

    random_dir = ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(10))
    save_path = os.path.join(os.getcwd(), random_dir)
    os.makedirs(save_path, exist_ok=True)

    file_path = os.path.join(save_path, file.filename)
    file.save(file_path)
    base64Img = detect.get_camera_screen(file_path, tp="video")
    os.remove(file_path)
    os.rmdir(save_path)
    result = responce.result(200, "success",base64Img)
    return jsonify(result), 200







